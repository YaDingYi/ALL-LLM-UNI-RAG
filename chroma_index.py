'''使用本地 Chroma 保存资料片段、向量和索引快照。'''

from datetime import datetime
import hashlib
import json
from math import isfinite
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from embedding_demo import MODEL_NAME, QUERY_PREFIX, embed_texts

# 所有 Chroma 数据位于本项目，和旧的 JSON 索引目录相互独立。
CHROMA_DIR = Path(__file__).resolve().parent / ".local" / "chroma"
DIMENSION = 512
BATCH_SIZE = 16
PREPROCESSING = "ascii-lower-v1"


def _client():
    '''连接本地持久化数据库；不使用 Chroma 的默认嵌入模型。'''
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )


def _new_name():
    '''为每次建立的知识库生成不会覆盖旧集合的名称。'''
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    return f"kb-{timestamp}-{uuid4().hex}"


def _metadata(chunks, source_filename=None, source_digest=None):
    '''记录模型、向量维度和来源，方便查询前检查兼容性。'''
    data = {
        "ready": False,
        "version": 1,
        "model": MODEL_NAME,
        "preprocessing": PREPROCESSING,
        "dimension": DIMENSION,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "file_count": len({chunk["source"] for chunk in chunks}),
        "chunk_count": len(chunks),
    }
    if source_filename is not None:
        data["source_filename"] = source_filename
        data["source_digest"] = source_digest
    return data


def _validate_vector(vector):
    '''只接受有限、非零的 512 维向量。'''
    if not isinstance(vector, list) or len(vector) != DIMENSION:
        raise ValueError("向量维度与当前模型不一致。")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) for value in vector):
        raise ValueError("向量包含无效数字。")
    if not any(vector):
        raise ValueError("零向量无法用于余弦检索。")


def _validate_chunk(chunk):
    '''检查需要写入数据库的片段和来源位置。'''
    if not isinstance(chunk, dict):
        raise ValueError("资料片段格式错误。")
    if not isinstance(chunk.get("source"), str) or not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
        raise ValueError("资料片段缺少来源或正文。")
    for field in ("number", "start", "end"):
        if not isinstance(chunk.get(field), int) or isinstance(chunk[field], bool):
            raise ValueError("资料片段的位置格式错误。")
    page = chunk.get("page")
    if page is not None and (not isinstance(page, int) or isinstance(page, bool)):
        raise ValueError("PDF 页码格式错误。")


def _write_snapshot(chunks, vectors_for_batch, on_progress=None, source_filename=None, source_digest=None):
    '''分批写入一个新集合；完整写入后才标记为可用。'''
    if not chunks:
        raise ValueError("没有可建立索引的资料片段。")
    for chunk in chunks:
        _validate_chunk(chunk)

    client = _client()
    name = _new_name()
    metadata = _metadata(chunks, source_filename, source_digest)
    collection = client.create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata=metadata,
        embedding_function=None,
    )
    try:
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            vectors = vectors_for_batch(start, batch)
            if len(vectors) != len(batch):
                raise ValueError("片段和向量数量不一致。")
            for vector in vectors:
                _validate_vector(vector)

            # 导入旧索引时保留原 id；新片段则按顺序分配 id。
            ids = [chunk.get("id", f"chunk-{number:06d}")
                   for number, chunk in enumerate(batch, start=start + 1)]
            documents = [chunk["text"] for chunk in batch]
            metadatas = []
            for chunk in batch:
                # Chroma 的 metadata 值不能为 None，非 PDF 片段省略 page。
                item = {key: chunk[key] for key in ("source", "number", "start", "end")}
                if chunk.get("page") is not None:
                    item["page"] = chunk["page"]
                metadatas.append(item)

            # 显式提供向量，避免 Chroma 使用其他默认模型。
            collection.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
            if on_progress is not None:
                on_progress(start + len(batch), len(chunks))

        if collection.count() != len(chunks):
            raise ValueError("集合中的片段数量与原资料不一致。")
        # 只有全部写入并确认数量后，界面才能列出这个集合。
        metadata["ready"] = True
        collection.modify(metadata=metadata)
    except BaseException:
        # 清理本次未完成集合；清理失败时 ready=False 仍可防止误选。
        try:
            client.delete_collection(name)
        except Exception:
            pass
        raise
    return {"name": name, "file_count": metadata["file_count"],
            "chunk_count": metadata["chunk_count"], "dimension": DIMENSION}


def create_index(chunks, on_progress=None):
    '''使用现有本地模型生成向量，建立一份新的 Chroma 快照。'''
    usable = []
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
            raise ValueError("资料片段格式错误。")
        if chunk["text"].strip():
            usable.append(chunk)
    return _write_snapshot(
        usable, lambda start, batch: embed_texts([chunk["text"] for chunk in batch]),
        on_progress=on_progress,
    )


def list_indexes():
    '''只列出写入完成的 Chroma 集合，按创建时间从新到旧排列。'''
    if not CHROMA_DIR.exists():
        return []
    client = _client()
    names = []
    for item in client.list_collections():
        name = item if isinstance(item, str) else item.name
        collection = client.get_collection(name=name, embedding_function=None)
        if name.startswith("kb-") and (collection.metadata or {}).get("ready") is True:
            names.append(((collection.metadata or {}).get("created_at", ""), name))
    return [name for _, name in sorted(names, reverse=True)]


def _get_ready_collection(name):
    '''检查集合状态和模型配置，再允许检索。'''
    if not isinstance(name, str) or not name.startswith("kb-"):
        raise ValueError("知识库集合名称无效。")
    collection = _client().get_collection(name=name, embedding_function=None)
    metadata = collection.metadata or {}
    if metadata.get("ready") is not True or metadata.get("version") != 1:
        raise ValueError("知识库尚未构建完成或格式不兼容。")
    if (metadata.get("model") != MODEL_NAME or metadata.get("preprocessing") != PREPROCESSING
            or metadata.get("dimension") != DIMENSION):
        raise ValueError("知识库模型、预处理或向量维度不兼容。")
    # 转换分数依赖余弦距离，不能把其他距离误当成余弦距离。
    config = collection.configuration or {}
    if (config.get("hnsw") or {}).get("space") != "cosine":
        raise ValueError("知识库没有使用余弦距离，请重新建立。")
    count = collection.count()
    if count < 1 or metadata.get("chunk_count") != count:
        raise ValueError("知识库片段数量不一致，请重新建立。")
    return collection, count


def cleanup_old_indexes(keep_name):
    '''保留最新的可用知识库，删除其他已完成的知识库集合。

    保留集合在删除前必须通过模型、维度、距离及片段数校验，而且必须是
    最新可用集合。未完成集合可能正在由其他进程构建，不在这里删除。
    删除结果逐项返回；删除失败时调用者应向用户报告，不能声称清理完成。
    调用方需保证本进程的更新和清理操作串行执行。
    '''
    # 保留对象不可用时直接终止，确保不会删除唯一可查询的集合。
    collection, _ = _get_ready_collection(keep_name)
    ready = list_indexes()
    # 旧版本不应清掉更新版本；清理前再次检查当前最新可用集合。
    if not ready or ready[0] != keep_name:
        raise ValueError("保留的知识库不是最新可用集合，已停止清理。")

    # 从保留集合读出一个已有向量，并直接查询，验证向量索引确实可用。
    # 全程只使用本地已有向量，不加载嵌入模型、不发送任何数据到网络。
    sample = collection.get(limit=1, include=["embeddings"])
    vectors = sample.get("embeddings")
    if len(sample.get("ids", [])) != 1 or vectors is None or len(vectors) != 1:
        raise ValueError("最新知识库无法读取已有向量，已停止清理。")
    vector = vectors[0].tolist() if hasattr(vectors[0], "tolist") else list(vectors[0])
    _validate_vector(vector)
    probe = collection.query(query_embeddings=[vector], n_results=1, include=["distances"])
    distances = probe.get("distances")
    if (not probe.get("ids") or not probe["ids"][0] or distances is None
            or not distances or not distances[0] or not isfinite(distances[0][0])):
        raise ValueError("最新知识库无法完成向量检索，已停止清理。")

    # 先建立待清理名称快照，明确只处理本项目的kb-命名空间。
    client = _client()
    old_names = [name for name in ready if name != keep_name]
    skipped_unready = []
    for item in client.list_collections():
        name = item if isinstance(item, str) else item.name
        if isinstance(name, str) and name.startswith("kb-") and name not in ready:
            # 未完成的集合可能仍在创建中，不能当作旧版本删除。
            skipped_unready.append(name)
    result = {"kept": keep_name, "deleted": [], "failed": [],
              "skipped_unready": sorted(skipped_unready), "aborted": False}

    # 每次删除前重新确认保留对象仍完整、仍是最新，减小并发更新风险。
    for name in old_names:
        try:
            _get_ready_collection(keep_name)
            current = list_indexes()
            if not current or current[0] != keep_name:
                raise ValueError("最新知识库发生变化。")
        except Exception as error:
            # 已删除的项保留在结果里；尚未执行的删除全部停止。
            result["aborted"] = True
            result["failed"].append({"name": name, "error_type": type(error).__name__})
            break
        try:
            # Chroma删除的是该集合，不删除原始资料、历史和其他命名空间。
            client.delete_collection(name)
            result["deleted"].append(name)
        except Exception as error:
            # 一个集合删除失败不阻止其他旧集合继续清理。
            result["failed"].append({"name": name, "error_type": type(error).__name__})
    return result


def search_index(question, index_name, top_k=3):
    '''将问题向量交给 Chroma 查询，返回片段及余弦相似度。'''
    if not isinstance(question, str) or not question.strip():
        raise ValueError("请输入要检索的问题。")
    if len(question) > 200:
        raise ValueError("请将问题控制在 200 个字符以内。")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("返回数量必须大于 0。")
    collection, count = _get_ready_collection(index_name)

    # 沿用原项目的中文模型及问题前缀，查询时不重新生成资料向量。
    vectors = embed_texts([QUERY_PREFIX + question])
    if len(vectors) != 1:
        raise ValueError("模型未返回有效的问题向量。")
    _validate_vector(vectors[0])
    result = collection.query(
        query_embeddings=vectors, n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    references = []
    for identifier, document, metadata, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        if not isinstance(metadata, dict) or not isinstance(document, str) or not isfinite(distance):
            raise ValueError("知识库返回的片段格式错误。")
        # 显式使用 cosine 时，余弦相似度 = 1 - 余弦距离。
        references.append({
            "id": identifier, "source": metadata["source"], "page": metadata.get("page"),
            "number": metadata["number"], "text": document, "score": 1.0 - float(distance),
        })
    return references, count


def import_index(index_data, source_filename, on_progress=None):
    '''校验旧 JSON 索引并复用其中的向量，重复导入同一快照时复用集合。'''
    if (not isinstance(source_filename, str) or Path(source_filename).name != source_filename
            or not source_filename.startswith("index-") or not source_filename.endswith(".json")):
        raise ValueError("旧索引文件名无效。")
    if not isinstance(index_data, dict) or index_data.get("version") != 1:
        raise ValueError("旧索引格式不兼容。")
    if (index_data.get("model") != MODEL_NAME or index_data.get("preprocessing") != PREPROCESSING
            or index_data.get("dimension") != DIMENSION):
        raise ValueError("旧索引使用的模型、预处理或向量维度不兼容。")
    records = index_data.get("records")
    if not isinstance(records, list) or not records or index_data.get("chunk_count") != len(records):
        raise ValueError("旧索引片段数量不一致。")
    identifiers = set()
    for record in records:
        _validate_chunk(record)
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("旧索引片段标识无效或重复。")
        identifiers.add(identifier)
        _validate_vector(record.get("vector"))
    if index_data.get("file_count") != len({record["source"] for record in records}):
        raise ValueError("旧索引资料数量不一致。")

    # 用内容哈希识别同一版本；不依赖文件名判断内容是否相同。
    encoded = json.dumps(index_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    client = _client()
    for item in client.list_collections():
        name = item if isinstance(item, str) else item.name
        collection = client.get_collection(name=name, embedding_function=None)
        metadata = collection.metadata or {}
        if (name.startswith("kb-") and metadata.get("ready") is True
                and metadata.get("source_filename") == source_filename and metadata.get("source_digest") == digest):
            _get_ready_collection(name)
            return {"name": name, "file_count": metadata["file_count"],
                    "chunk_count": metadata["chunk_count"], "dimension": DIMENSION, "already_exists": True}

    result = _write_snapshot(
        records, lambda start, batch: [record["vector"] for record in batch],
        on_progress=on_progress, source_filename=source_filename, source_digest=digest,
    )
    result["already_exists"] = False
    return result
