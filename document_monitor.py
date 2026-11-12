'''保存资料文件清单，比较两次检查之间新增、修改和删除的文档。'''

from datetime import datetime
import json
import os
from pathlib import Path
import stat
from tempfile import NamedTemporaryFile
from threading import RLock
from zoneinfo import ZoneInfo

import document_reader

# 检查记录独立于知识库：只保存文件信息，不代表文件已经完成入库。
SNAPSHOT_PATH = Path(__file__).resolve().parent / ".local" / "document_inventory.json"
_CHECK_LOCK = RLock()


def scan_documents(documents_dir):
    '''递归收集支持的资料文件，只读取文件信息，不提取正文或生成向量。'''
    root = Path(documents_dir)
    if not root.is_dir():
        raise FileNotFoundError("资料目录不存在或无法访问。")
    files = {}

    def fail_on_error(error):
        # 遇到无法访问的子目录就中止，防止把漏扫的文件误判为删除。
        raise error

    for folder, _, names in os.walk(root, onerror=fail_on_error):
        for name in names:
            path = Path(folder) / name
            # 与资料读取模块保持相同的格式和 Office 临时文件过滤规则。
            if name.startswith(("~$", ".~")) or path.suffix.lower() not in document_reader.SUPPORTED_SUFFIXES:
                continue
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                continue
            source = path.relative_to(root).as_posix()
            # 修改时间和大小用于快速判断文件变化，无须重新读取整个 PDF。
            files[source] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
    return dict(sorted(files.items()))


def _load_snapshot(path):
    '''读取并校验旧清单；只有首次没有清单时才返回 None。'''
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("files"), dict):
        raise ValueError("资料检查记录格式不正确。")
    for source, info in data["files"].items():
        if (not isinstance(source, str) or not source or not isinstance(info, dict)
                or any(type(info.get(key)) is not int for key in ("size", "mtime_ns"))
                or info["size"] < 0):
            raise ValueError("资料检查记录中的文件信息不正确。")
    return data


def _save_snapshot(path, data):
    '''先写入临时文件，再原子替换清单，避免写入中断损坏原记录。'''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix="document_inventory-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_document_changes(documents_dir=None, snapshot_path=None):
    '''比较当前文件与上次检查清单；完整扫描并保存成功后，返回变化结果。'''
    root = document_reader.DOCUMENTS_DIR if documents_dir is None else Path(documents_dir)
    path = SNAPSHOT_PATH if snapshot_path is None else Path(snapshot_path)
    # 同一服务中的多个浏览器共用清单，避免同时比较和写入产生冲突。
    with _CHECK_LOCK:
        previous = _load_snapshot(path)
        current = scan_documents(root)
        checked_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        # 首次没有历史基准，不能把所有现有资料声称为“刚刚新增”。
        old_files = previous["files"] if previous is not None else current
        result = {
            "first_check": previous is None,
            "checked_at": checked_at,
            "file_count": len(current),
            "added": sorted(current.keys() - old_files.keys()),
            "modified": sorted(name for name in current.keys() & old_files.keys()
                               if current[name] != old_files[name]),
            "deleted": sorted(old_files.keys() - current.keys()),
        }
        _save_snapshot(path, {"version": 1, "checked_at": checked_at, "files": current})
        return result
