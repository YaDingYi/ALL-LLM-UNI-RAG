'''保存网页上传的资料，并根据全部本地资料生成新的索引快照。'''

import os
from pathlib import Path

import document_reader
from index_demo import build_index, save_index


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


def update_knowledge_index(required_sources=(), on_progress=None):
    '''读取全部资料，生成新索引；返回文件名、数量和读取提示。'''
    # 重新读取全部资料，保证新索引不仅包含本次上传的文件。
    documents, warnings = document_reader.read_documents()
    sources = {document["source"] for document in documents}
    missing = [source for source in required_sources if source not in sources]
    if missing:
        # 上传成功不代表文档有可提取的文字，失败时不发布不完整的新索引。
        details = [f"{warning['source']}：{warning['message']}" for warning in warnings if warning["source"] in missing]
        message = "以下上传资料没有读取到正文，未生成新索引：" + "、".join(missing)
        if details:
            message += "；" + "；".join(details)
        raise UploadError(message)

    # 全部片段重新向量化；save_index 创建新快照，不覆盖旧索引。
    chunks = document_reader.split_documents(documents)
    if not chunks:
        raise UploadError("没有可建立索引的资料片段，请先准备含文字的资料。")
    index = build_index(chunks, on_progress=on_progress)
    filename = save_index(index)
    return {
        "filename": filename,
        "file_count": index["file_count"],
        "chunk_count": index["chunk_count"],
        "warnings": warnings,
    }
