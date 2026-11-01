'''将资料片段和向量保存为可查看的本地学习索引。'''

from datetime import datetime
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from uuid import uuid4
from zoneinfo import ZoneInfo

from embedding_demo import MODEL_NAME, embed_texts

# 学习索引存放在项目目录下；一个 JSON 文件是一份独立快照。
INDEX_DIR = Path(__file__).resolve().parent / "vector_index"
BATCH_SIZE = 16


def build_index(chunks, on_progress=None):
    '''生成片段向量，并整理成可以保存的索引数据。'''
    # 只处理包含文字的片段。
    usable = [chunk for chunk in chunks if chunk["text"].strip()]
    if not usable:
        raise ValueError("没有可建立索引的资料片段。")

    records = []
    dimension = None

    # 分批生成向量，保持片段与向量的顺序一致。
    for start in range(0, len(usable), BATCH_SIZE):
        batch = usable[start:start + BATCH_SIZE]
        vectors = embed_texts([chunk["text"] for chunk in batch])

        if len(vectors) != len(batch):
            raise ValueError("片段数量与向量数量不一致。")

        for chunk, vector in zip(batch, vectors):
            if not vector or not all(isfinite(value) for value in vector):
                raise ValueError("模型返回了无效向量。")
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                raise ValueError("模型返回的向量维度不一致。")

            # 一条记录同时保存原文、来源位置和对应向量。
            records.append({
                "id": f"chunk-{len(records) + 1:06d}",
                "source": chunk["source"],
                "page": chunk["page"],
                "number": chunk["number"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "vector": vector,
            })

        if on_progress is not None:
            on_progress(min(start + len(batch), len(usable)), len(usable))

    # 保存模型信息，便于以后检查问题向量是否使用了相同模型。
    return {
        "version": 1,
        "model": MODEL_NAME,
        "preprocessing": "ascii-lower-v1",
        "dimension": dimension,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "file_count": len({record["source"] for record in records}),
        "chunk_count": len(records),
        "records": records,
    }


def save_index(index):
    '''完整写入临时文件后，再创建独立索引快照，避免覆盖旧索引。'''
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".index-", suffix=".tmp", dir=INDEX_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(index, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        # 日期加随机编号确保每次构建是独立文件；链接不会覆盖已有文件。
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
        filename = f"index-{timestamp}-{uuid4().hex}.json"
        os.link(temporary_path, INDEX_DIR / filename)
        return filename
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def list_indexes():
    '''列出本项目已经保存的索引快照。'''
    if not INDEX_DIR.exists():
        return []
    return sorted(
        (path.name for path in INDEX_DIR.glob("index-*.json")
         if path.is_file() and not path.is_symlink()),
        reverse=True,
    )


def load_index(filename):
    '''从本地读取指定索引；只允许读取索引目录中的普通文件。'''
    if not filename or Path(filename).name != filename or not filename.startswith("index-") or not filename.endswith(".json"):
        raise ValueError("索引文件名无效。")
    path = INDEX_DIR / filename
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("索引文件不存在。")
    with path.open("r", encoding="utf-8") as file:
        index = json.load(file)
    if not isinstance(index, dict) or not isinstance(index.get("records"), list):
        raise ValueError("索引格式不正确。")
    if not index["records"] or index.get("chunk_count") != len(index["records"]):
        raise ValueError("索引记录数量不正确。")
    dimension = index.get("dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("索引向量维度无效。")
    identifiers = set()
    for record in index["records"]:
        if not isinstance(record, dict) or not all(key in record for key in ("id", "source", "page", "number", "start", "end", "text", "vector")):
            raise ValueError("索引片段记录不完整。")
        if not isinstance(record["id"], str) or record["id"] in identifiers:
            raise ValueError("索引片段标识无效或重复。")
        identifiers.add(record["id"])
        if not isinstance(record["source"], str) or not isinstance(record["text"], str):
            raise ValueError("索引片段文字或来源无效。")
        vector = record["vector"]
        if not isinstance(vector, list) or len(vector) != dimension or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
            for value in vector
        ):
            raise ValueError("索引片段向量无效。")
    return index


def render_index_demo():
    '''显示构建索引和查看已保存索引的网页。'''
    import streamlit as st

    with st.expander("💾 学习：建立本地向量索引"):
        st.caption("先读取并切分资料。每次构建创建一个独立 JSON 快照，不调用 DeepSeek；资料变动后请重新读取并构建。")
        preview = st.session_state.get("document_preview")

        if not preview or not preview["chunks"]:
            st.info("请先点击“读取并切分资料”。")
        else:
            choices = ["全部资料"] + preview["sources"]

            # 优先使用单个学习文件，避免初次演示误处理全部资料。
            if "学习示例.txt" in choices:
                default_index = choices.index("学习示例.txt")
            elif "学习笔记.docx" in choices:
                default_index = choices.index("学习笔记.docx")
            else:
                default_index = 1

            with st.form("index_demo_form"):
                scope = st.selectbox("选择要建立索引的资料", choices, index=default_index)
                build = st.form_submit_button("生成并保存本地索引")

            if build:
                # 新构建开始时移除旧的成功提示，避免错误时显示旧结果。
                st.session_state.pop("last_created_index", None)
                selected = [chunk for chunk in preview["chunks"] if scope == "全部资料" or chunk["source"] == scope]
                progress = st.progress(0, text="准备生成本地向量…")
                try:
                    def update_progress(done, total):
                        progress.progress(done / total, text=f"已向量化 {done}/{total} 个片段")

                    index = build_index(selected, update_progress)
                    filename = save_index(index)
                    st.session_state["last_created_index"] = {
                        "filename": filename, "chunk_count": index["chunk_count"],
                        "dimension": index["dimension"],
                    }
                except ValueError as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(f"索引构建失败（{type(error).__name__}），已有索引未被覆盖。")
                finally:
                    progress.empty()

        created = st.session_state.get("last_created_index")
        if created:
            st.success(f"已保存 {created['chunk_count']} 个片段，向量维度 {created['dimension']}。文件：vector_index/{created['filename']}")

        try:
            filenames = list_indexes()
        except OSError:
            st.error("无法读取 vector_index 文件夹，请检查权限。")
            return
        if not filenames:
            st.info("还没有已保存的索引。")
        else:
            selected_file = st.selectbox("选择已保存的索引", filenames)
            if st.button("查看索引内容"):
                try:
                    index = load_index(selected_file)
                    first = index["records"][0]
                    st.session_state["index_demo_preview"] = {
                        "filename": selected_file, "model": index.get("model"),
                        "chunk_count": index["chunk_count"], "dimension": index.get("dimension"),
                        "first": first,
                    }
                except (OSError, ValueError, KeyError, TypeError):
                    st.session_state.pop("index_demo_preview", None)
                    st.error("索引文件无法读取或格式不正确；原文件未被修改。")

            saved = st.session_state.get("index_demo_preview")

            if saved is not None and saved["filename"] == selected_file:
                first = saved["first"]
                st.text(f"索引文件：vector_index/{saved['filename']}")
                st.text(f"模型：{saved['model']} · 片段：{saved['chunk_count']} · 向量维度：{saved['dimension']}")

                page_label = f" · PDF 第 {first['page']} 页" if first.get("page") is not None else ""

                st.text(f"第一条记录：{first['id']} · {first['source']}{page_label}")
                st.text(first["text"])
                st.text(f"向量前 8 个数字：{first['vector'][:8]}")
                st.text_area("索引中第一条记录的完整向量", value=str(first["vector"]), height=120, disabled=True)
