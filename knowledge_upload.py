'''保存上传资料，按文件变化增量更新知识库并复用已有向量。'''

import os
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import chroma_index
import document_reader
from filelock import FileLock
from document_monitor import scan_documents
from chroma_index import _get_ready_collection, cleanup_old_indexes, list_indexes
from incremental_index import (
    create_incremental_index, list_index_sources, match_legacy_sources, read_index_manifest,
)

# 同一进程中的多个网页会话不能同时构建，防止选择同一基线后互相覆盖状态。
_UPDATE_LOCK = RLock()


@contextmanager
def knowledge_update_lock():
    '''让同机多个服务进程串行更新与清理，避免新集合被其他更新误删。'''
    with _UPDATE_LOCK:
        # 锁文件独立于Chroma数据库，且跟随测试使用的临时数据库目录。
        lock_path = chroma_index.CHROMA_DIR.parent / "knowledge-update.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_path)):
            yield


def _finish_update(result):
    '''新集合发布或确认未变化后清理旧集合；清理失败单独报告。'''
    try:
        # 清理器会再次校验保留集合完整且是最新版本。
        result["cleanup"] = cleanup_old_indexes(result["name"])
    except Exception as error:
        # 新知识库已成功，不能把清理失败误报为索引构建失败。
        result["cleanup"] = {"kept": result["name"], "deleted": [],
                             "failed": [{"name": "清理操作", "error_type": type(error).__name__}],
                             "skipped_unready": [], "aborted": True}
    return result


class UploadError(Exception):
    '''上传文件或更新知识库时出现可以向用户说明的问题。'''


def save_uploaded_files(uploaded_files):
    '''将上传文件保存到资料/上传，返回相对于资料目录的文件名。'''
    # 先检查所有文件名，避免写入之后才发现有不支持的文件。
    files = list(uploaded_files)
    if not files:
        raise UploadError("请先选择要上传的文件。")
    for uploaded in files:
        name = uploaded.name
        if (not isinstance(name, str) or not name or name in (".", "..")
                or "/" in name or "\\" in name or any(ord(char) < 32 or ord(char) == 127 for char in name)
                or name.startswith(("~$", ".~")) or len(name.encode("utf-8")) > 180):
            raise UploadError("上传文件名无效，请使用不含路径和特殊控制字符的较短文件名。")
        if Path(name).suffix.lower() not in document_reader.SUPPORTED_SUFFIXES:
            raise UploadError(f"不支持文件格式：{name}。请选择 TXT、Markdown、PDF 或 DOCX。")

    # 单独的上传目录便于区分原有资料；拒绝符号链接，避免写入其他位置。
    upload_dir = document_reader.DOCUMENTS_DIR / "上传"
    if upload_dir.is_symlink():
        raise UploadError("资料/上传 不能是符号链接，请检查该目录。")
    upload_dir.mkdir(parents=True, exist_ok=True)
    if upload_dir.is_symlink() or not upload_dir.is_dir():
        raise UploadError("无法使用资料/上传目录，请检查该目录。")

    saved_paths = []
    try:
        for uploaded in files:
            original = Path(uploaded.name)
            number = 0
            while True:
                # 文件同名时增加编号；独占创建保证不会覆盖已有文件。
                name = original.name if number == 0 else f"{original.stem}_{number}{original.suffix}"
                path = upload_dir / name
                try:
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    break
                except FileExistsError:
                    number += 1
            # 先记录已创建文件，发生错误时可以一并清理。
            saved_paths.append(path)
            with os.fdopen(fd, "wb") as file:
                file.write(uploaded.getvalue())
        return [path.relative_to(document_reader.DOCUMENTS_DIR).as_posix() for path in saved_paths]
    except Exception:
        # 如果本批上传没有全部完成，撤销本批已经创建的文件，不影响原有资料。
        for path in saved_paths:
            path.unlink(missing_ok=True)
        raise


def update_knowledge_index(required_sources=(), on_progress=None, base_name=None, preserve_missing=False):
    '''比较资料变化、复用未变向量，并在成功后清理旧的知识库。'''
    with knowledge_update_lock():
        # 只读取文件大小与修改时间，不读取未变文件的全部字节。
        current = scan_documents(document_reader.DOCUMENTS_DIR)
        indexes = list_indexes()
        # 用户已选集合时使用该集合；尚未选择时才使用最新集合。
        if base_name is None:
            base_name = indexes[0] if indexes else None
        elif base_name not in indexes:
            raise UploadError("所选知识库已不可用，请重新选择后再更新。")
        elif base_name != indexes[0]:
            raise UploadError("只能基于最新知识库更新，请先选择最新集合。")

        # 索引清单独立于资料检查清单，只有成功入库后才建立。
        manifest = read_index_manifest(base_name) if base_name else None
        old_sources = list_index_sources(base_name) if base_name else set()
        legacy = base_name is not None and manifest is None

        if manifest is None:
            # 旧集合没有文件版本信息；首次需读取和对比旧片段以确定可复用范围。
            to_read = set(current)
            reused_sources = set()
        else:
            # 清单记录的是每个文件成功入库时的大小和修改时间。
            reused_sources = {source for source in old_sources
                              if source in current and manifest.get(source) == current[source]
                              and manifest[source]["mtime_ns"] != -1}
            # 上传追加时保留旧库里已不在资料目录中的文件；重扫则同步删除。
            if preserve_missing:
                reused_sources |= old_sources - set(current)
            to_read = set(current) - reused_sources

        # 文件删除与文件无法解析是两种情况：前者移除，后者禁止发布不完整结果。
        missing_on_disk = old_sources - set(current)
        deleted_sources = set() if preserve_missing else missing_on_disk
        documents, warnings = document_reader.read_documents(to_read)
        readable = {document["source"] for document in documents}
        missing_required = set(required_sources) - readable - reused_sources
        # 每个应读取文件都必须成功；不能因为新文件无法解析就悄悄漏索引。
        missing = sorted(missing_required | (to_read - readable))
        if missing:
            details = [f"{warning['source']}：{warning['message']}" for warning in warnings if warning["source"] in missing]
            message = "以下资料无法读取正文，未更新知识库：" + "、".join(missing)
            if details:
                message += "；" + "；".join(details)
            raise UploadError(message)

        # 已索引文件如果被编辑为仅含空白，不能静默当成删除。
        # 延续原建库规则：不把纯空白片段写入向量库。
        chunks = [chunk for chunk in document_reader.split_documents(documents) if chunk["text"].strip()]
        if legacy:
            # 对旧集合逐文件比较原文与位置信息；只有完全匹配的来源才复用旧向量。
            reused_sources = match_legacy_sources(base_name, chunks)
            # 上传操作只追加，不因旧来源缺席而丢失原库中的片段。
            if preserve_missing:
                reused_sources |= missing_on_disk
        new_chunks = [chunk for chunk in chunks if chunk["source"] not in reused_sources]
        successful_sources = {chunk["source"] for chunk in chunks} | reused_sources
        # 已缺失的旧来源用-1标记“无法核对”；以后重新出现时强制重新解析。
        signatures = {
            source: (current[source] if source in current else {"size": 0, "mtime_ns": -1})
            for source in sorted(successful_sources)
        }
        if not signatures:
            raise UploadError("没有可建立索引的资料片段，请先准备含文字的资料。")

        def verify_unchanged():
            # 构建期间发生添加、删除或修改时停止发布，避免索引与文件清单不一致。
            if scan_documents(document_reader.DOCUMENTS_DIR) != current:
                raise UploadError("资料在更新过程中发生变化，请重新更新知识库。")

        # 有清单且确实没有任何有效变化时，直接复用原集合，不生成新集合。
        if manifest is not None and not new_chunks and not deleted_sources and signatures == manifest:
            verify_unchanged()
            collection, count = _get_ready_collection(base_name)
            return _finish_update({"name": base_name, "base_name": base_name, "file_count": len(signatures), "chunk_count": count,
                    "warnings": warnings, "unchanged": True, "legacy": False,
                    "reused_file_count": len(reused_sources), "new_file_count": 0,
                    "modified_file_count": 0, "deleted_file_count": 0,
                    "retained_missing_count": len(missing_on_disk) if preserve_missing else 0,
                    "added_sources": [], "modified_sources": [], "deleted_sources": [],
                    "retained_missing_sources": sorted(missing_on_disk) if preserve_missing else [],
                    "reused_chunk_count": count, "new_chunk_count": 0})

        # 新集合复制可复用向量，仅对新增和修改的文件片段调用本地嵌入模型。
        index = create_incremental_index(
            base_name, reused_sources, new_chunks, signatures,
            on_progress=on_progress, validate_before_publish=verify_unchanged,
            missing_sources=missing_on_disk if preserve_missing else (),
        )
        return _finish_update({**index, "base_name": base_name, "warnings": warnings, "unchanged": False, "legacy": legacy,
                "reused_file_count": len(reused_sources),
                "new_file_count": len(successful_sources - old_sources),
                "modified_file_count": len((successful_sources & old_sources) - reused_sources),
                "deleted_file_count": len(deleted_sources),
                "retained_missing_count": len(missing_on_disk) if preserve_missing else 0,
                "added_sources": sorted(successful_sources - old_sources),
                "modified_sources": sorted((successful_sources & old_sources) - reused_sources),
                "deleted_sources": sorted(deleted_sources),
                "retained_missing_sources": sorted(missing_on_disk) if preserve_missing else []})
