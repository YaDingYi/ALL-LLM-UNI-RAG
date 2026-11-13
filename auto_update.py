'''运行一次知识库更新，并将结果保存为本地 Word 更新记录。'''

from datetime import datetime
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo

from knowledge_upload import UploadError, update_knowledge_index


# 更新记录和资料目录分开，避免记录被再次读入知识库。
UPDATE_HISTORY_DIR = Path(__file__).resolve().parent / "更新历史"
# 同一进程内串行执行“更新 + 写记录”，防止启动更新与上传更新交错。
_RUN_LOCK = RLock()


def write_update_record(record, directory=None):
    '''将一次更新结果写入独立的 .docx 文件，返回生成的绝对路径。'''
    # 延迟导入，便于在更新失败或测试时单独检查记录逻辑。
    from docx import Document

    # 测试可以传入临时目录；正常运行使用项目中的“更新历史”。
    target_dir = UPDATE_HISTORY_DIR if directory is None else Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    document = Document()
    document.add_heading("知识库更新记录", level=0)

    # 记录实际开始和结束时间；不要把搜索网页时间混入索引更新记录。
    document.add_paragraph(f"更新方式：{record['trigger']}")
    document.add_paragraph(f"开始时间：{record['started_at'] or '未记录'}")
    document.add_paragraph(f"结束时间：{record['finished_at']}")
    duration = record['duration_seconds']
    document.add_paragraph(f"耗时：{duration:.2f} 秒" if duration is not None else "耗时：未记录")
    document.add_paragraph(f"更新状态：{'成功' if record['ok'] else '失败'}")
    # 自动选择基线时，成功结果会回传实际使用的旧集合名称。
    result = record.get("result")
    base_name = record.get("base_name") or (result or {}).get("base_name") or "首次建库或自动选择未完成"
    document.add_paragraph(f"基线集合：{base_name}")

    if record["ok"] and isinstance(result, dict):
        document.add_paragraph(f"使用集合：{result.get('name', '未知')}")
        document.add_paragraph(f"资料文件：{result.get('file_count', 0)} 个；片段：{result.get('chunk_count', 0)} 个")
        document.add_paragraph(
            f"新增文件：{result.get('new_file_count', 0)} 个；"
            f"修改文件：{result.get('modified_file_count', 0)} 个；"
            f"删除文件：{result.get('deleted_file_count', 0)} 个"
        )
        document.add_paragraph(
            f"复用旧向量：{result.get('reused_chunk_count', 0)} 个片段；"
            f"新生成向量：{result.get('new_chunk_count', 0)} 个片段"
        )
        if result.get("unchanged"):
            document.add_paragraph("资料没有有效变化，本次继续使用原集合。")
        if result.get("legacy"):
            document.add_paragraph("本次为旧集合首次迁移，已读取资料并核对可复用片段。")
        if result.get("retained_missing_count"):
            document.add_paragraph(f"保留旧库中当前缺失的来源：{result['retained_missing_count']} 个")
        # 记录旧集合清理的实际结果，与资料文件删除数量分开。
        cleanup = result.get("cleanup")
        if cleanup is not None:
            document.add_heading("旧知识库清理", level=2)
            document.add_paragraph(f"保留最新集合：{cleanup.get('kept', '未知')}")
            document.add_paragraph(f"已删除旧集合：{len(cleanup.get('deleted', []))} 个")
            for name in cleanup.get("deleted", []):
                document.add_paragraph(str(name), style="List Bullet")
            if cleanup.get("failed") or cleanup.get("aborted"):
                document.add_paragraph("部分旧集合未能清理，后续启动会再次尝试。")
                for item in cleanup.get("failed", []):
                    document.add_paragraph(f"{item.get('name', '未知')}：{item.get('error_type', '未知错误')}")
            if cleanup.get("skipped_unready"):
                document.add_paragraph("未完成的集合可能正在构建，本次没有删除：")
                for name in cleanup["skipped_unready"]:
                    document.add_paragraph(str(name), style="List Bullet")
        # 如果更新函数提供逐文件明细，就记录相对路径；绝不复制资料全文。
        for key, title in (("added_sources", "新增文件"), ("modified_sources", "修改文件"),
                           ("deleted_sources", "删除文件"), ("retained_missing_sources", "保留的缺失来源")):
            items = result.get(key, [])
            if items:
                document.add_heading(title, level=2)
                for source in items:
                    document.add_paragraph(str(source), style="List Bullet")
        if result.get("warnings"):
            document.add_heading("读取提示", level=2)
            for warning in result["warnings"]:
                document.add_paragraph(f"{warning.get('source', '未知来源')}：{warning.get('message', '')}")
    else:
        # 错误内容已经在run_update中转换成可安全显示的提示。
        document.add_paragraph(f"失败原因：{record.get('error', '未知错误')}")
        document.add_paragraph("未发布不完整的新知识库集合；原有可用集合保留。")

    # 时间戳和随机后缀避免重复启动时覆盖旧记录。
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    target = target_dir / f"知识库更新-{stamp}-{uuid4().hex[:8]}.docx"
    temporary = None
    try:
        # 在同一目录先写临时文件；完整保存后再重命名为最终记录。
        with NamedTemporaryFile(dir=target_dir, prefix=".更新记录-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
        document.save(temporary)
        os.replace(temporary, target)
    finally:
        # 异常时不留下无法打开的半成品记录。
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(target.resolve())


def _save_record(record):
    '''为已完成的操作写记录，保留更新本身的成功或失败状态。'''
    try:
        # 成功和失败都尝试写记录；日志不保存密钥、聊天或文档正文。
        record["log_path"] = write_update_record(record)
    except Exception as exc:
        # 索引成功而记录写入失败时要清楚区分，不能误报建库失败。
        record["log_error"] = f"更新记录未能保存（{type(exc).__name__}），请检查“更新历史”目录权限。"
    return record


def run_update(trigger="启动自动更新", base_name=None, required_sources=(),
               preserve_missing=False, on_progress=None):
    '''执行一次增量更新并记录结果；是否只运行一次由调用方的缓存控制。'''
    with _RUN_LOCK:
        started = datetime.now(ZoneInfo("Asia/Shanghai"))
        timer = perf_counter()
        result = None
        error = None
        try:
            # 复用已有增量算法，不读取或传送 API Key，也不调用 DeepSeek。
            result = update_knowledge_index(
                required_sources=required_sources, on_progress=on_progress,
                base_name=base_name, preserve_missing=preserve_missing,
            )
        except UploadError as exc:
            # 已由项目生成的错误可用于说明具体文件问题。
            error = str(exc)
        except Exception as exc:
            # 不暴露不可预期的原始异常或内部配置信息。
            error = f"更新未完成（{type(exc).__name__}），请检查资料、模型和目录权限。"

        record = {
            "trigger": trigger,
            "base_name": base_name,
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "duration_seconds": perf_counter() - timer,
            "ok": error is None,
            "result": result,
            "error": error,
            "log_path": None,
            "log_error": None,
        }
        return _save_record(record)


def record_upload_failure(error, base_name=None):
    '''上传文件保存失败时写入失败记录，不触发知识库更新。'''
    with _RUN_LOCK:
        # 预期内的上传错误由项目生成，可以显示具体的文件提示。
        if isinstance(error, UploadError):
            message = str(error)
        else:
            # 未知异常仅显示类型，避免将原始错误中的路径或密钥写入文档。
            message = f"上传文件保存失败（{type(error).__name__}），请检查文件和资料目录权限。"
        moment = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        return _save_record({
            "trigger": "上传更新",
            "base_name": base_name,
            "started_at": None,
            "finished_at": moment,
            "duration_seconds": None,
            "ok": False,
            "result": None,
            "error": message,
            "log_path": None,
            "log_error": None,
        })
