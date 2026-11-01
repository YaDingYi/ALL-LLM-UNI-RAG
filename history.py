'''在本地 Histor 文件夹中读取和保存对话记录。'''

import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

# 所有对话文件都保存在项目目录下的 Histor 文件夹中。
HISTORY_DIR = Path(__file__).resolve().parent / "Histor"


def list_history_files():
    '''列出可选的 JSON 历史文件，包含原来的 conversation.json。'''
    if not HISTORY_DIR.exists():
        return []
    return sorted(
        (path.name for path in HISTORY_DIR.iterdir()
         if path.suffix.lower() == ".json" and path.is_file() and not path.is_symlink()),
        reverse=True,
    )


def _history_path(filename):
    '''只允许使用 Histor 文件夹中已有的普通 JSON 文件。'''
    if not filename or Path(filename).name != filename or not filename.lower().endswith(".json"):
        raise ValueError("历史文件名无效")
    path = HISTORY_DIR / filename
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("历史文件不存在")
    return path


def load_history(filename):
    '''读取指定对话文件中的消息。'''
    path = _history_path(filename)

    # JSON 中保存消息列表，每条包含 role 和 content；助手消息也可以包含 usage。
    with path.open("r", encoding="utf-8") as file:
        messages = json.load(file)
    if not isinstance(messages, list):
        raise ValueError("历史记录必须是列表")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant") or not isinstance(message.get("content"), str):
            raise ValueError("历史记录包含无效消息")
    return messages


def _write_temporary(messages):
    '''先将完整消息列表写入同目录的临时文件。'''
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".history-", suffix=".tmp", dir=HISTORY_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(messages, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return temporary_path


def _question_filename(question):
    '''把首个问题转换为安全且不过长的文件名片段。'''
    # 替换路径分隔符、特殊符号和控制字符，避免生成路径或隐藏文件。
    safe = "".join(" " if unicodedata.category(char).startswith("C") else char for char in question)
    safe = re.sub(r'[\\/:*?"<>|]', " ", safe)
    safe = " ".join(safe.split()).strip(" .")
    # 截断到 120 字节，防止中文或较长问题超过文件名限制。
    safe = safe.encode("utf-8")[:120].decode("utf-8", errors="ignore").strip(" .")
    return safe or "未命名问题"


def create_history(question, messages):
    '''首次提问时创建日期和问题命名的文件，重复名称自动编号。'''
    date = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    stem = f"{date}_{_question_filename(question)}"
    temporary_path = _write_temporary(messages)
    try:
        number = 1
        while True:
            suffix = "" if number == 1 else f"_{number}"
            filename = f"{stem}{suffix}.json"
            try:
                # 硬链接只在目标不存在时成功，避免覆盖已有对话。
                os.link(temporary_path, HISTORY_DIR / filename)
                return filename
            except FileExistsError:
                number += 1
    finally:
        os.unlink(temporary_path)


def save_history(filename, messages):
    '''先写入临时文件，再替换历史文件，避免写入中断破坏旧记录。'''
    path = _history_path(filename)
    temporary_path = _write_temporary(messages)
    try:
        # 只有临时文件完整写入后，才替换原历史文件。
        os.replace(temporary_path, path)
    finally:
        # 如果写入失败，清理本次创建的临时文件。
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
