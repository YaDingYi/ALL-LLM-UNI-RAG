'''复用旧集合的向量，只为新增或变化的资料片段生成向量。'''

from datetime import datetime
import hashlib
import inspect
import json
from zoneinfo import ZoneInfo

import chroma_index
import document_reader

# 给切分规则一个明确版本；修改切分语义时应同时更新此值。
CHUNKING_VERSION = "character-pages-v1"
# 读取旧集合时按批取向量，避免把所有旧向量一次性装入内存。
COPY_BATCH_SIZE = 512


def _chunking_metadata():
    '''记录当前默认切分长度、重叠和规则版本。'''
    parameters = inspect.signature(document_reader.split_documents).parameters
    return {
        "chunking_version": CHUNKING_VERSION,
        "chunk_size": parameters["chunk_size"].default,
        "chunk_overlap": parameters["chunk_overlap"].default,
    }


def _validate_signatures(signatures):
    '''检查文件快照：只接受文件大小和纳秒修改时间组成的简单字典。'''
    if not isinstance(signatures, dict):
        raise ValueError("知识库文件清单格式错误。")
    for source, signature in signatures.items():
        if (not isinstance(source, str) or not source or not isinstance(signature, dict)
                or set(signature) != {"size", "mtime_ns"}
                or any(type(signature.get(key)) is not int for key in ("size", "mtime_ns"))
                or signature["size"] < 0):
            raise ValueError("知识库文件清单包含无效的文件信息。")


def read_index_manifest(base_name):
    '''读取旧集合对应的文件快照；旧格式或切分规则变化时返回None。'''
    if not base_name:
        return None
    collection, _ = chroma_index._get_ready_collection(base_name)
    metadata = collection.metadata or {}
    raw = metadata.get("source_signatures")
    if raw is None:
        return None
    # 策略改变时，需要重新对照文档片段，不能直接根据旧清单复用。
    if any(metadata.get(key) != value for key, value in _chunking_metadata().items()):
        return None
    if not isinstance(raw, str):
        raise ValueError("知识库文件清单格式错误。")
    try:
        signatures = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise ValueError("知识库文件清单无法读取。") from error
    _validate_signatures(signatures)
    return signatures


def _old_batches(collection, count, include):
    '''按固定大小读取旧集合，不改变旧集合。'''
    for offset in range(0, count, COPY_BATCH_SIZE):
        data = collection.get(limit=COPY_BATCH_SIZE, offset=offset, include=include)
        expected = min(COPY_BATCH_SIZE, count - offset)
        if len(data["ids"]) != expected:
            raise ValueError("读取旧集合时片段数量发生变化。")
        yield data


def list_index_sources(base_name):
    '''列出旧集合中的来源，帮助区分文件删除和文件读取失败。'''
    if not base_name:
        return set()
    collection, count = chroma_index._get_ready_collection(base_name)
    sources = set()
    for batch in _old_batches(collection, count, ["metadatas"]):
        for metadata in batch["metadatas"]:
            source = metadata.get("source")
            if not isinstance(source, str) or not source:
                raise ValueError("旧集合包含无效的资料来源。")
            sources.add(source)
    return sources


def _chunk_key(chunk):
    '''比较片段原文、页码和位置，避免把不同版本的资料当作相同。'''
    return (chunk["number"], chunk["start"], chunk["end"], chunk.get("page"), chunk["text"])


def match_legacy_sources(base_name, chunks):
    '''旧集合没有文件清单时，逐片段对照刚读取的文件，找出可复用来源。'''
    if not base_name:
        return set()
    collection, count = chroma_index._get_ready_collection(base_name)
    expected = {}
    for chunk in chunks:
        # 原始create_index会丢弃只有空白的片段，对照旧库时也必须采用同样规则。
        if isinstance(chunk, dict) and isinstance(chunk.get("text"), str) and not chunk["text"].strip():
            continue
        chroma_index._validate_chunk(chunk)
        expected.setdefault(chunk["source"], []).append(_chunk_key(chunk))
    old = {}
    for batch in _old_batches(collection, count, ["documents", "metadatas"]):
        for text, metadata in zip(batch["documents"], batch["metadatas"]):
            source = metadata["source"]
            if source in expected:
                old.setdefault(source, []).append(_chunk_key({**metadata, "text": text}))
    # 排序后同时比较所有片段、正文和位置；完全相同才可复用旧向量。
    return {source for source, values in expected.items()
            if source in old and sorted(values) == sorted(old[source])}


def _new_chunk_id(source, number):
    '''由来源及片段编号生成新集合内部ID，避免继承旧集合的ID冲突。'''
    value = f"{source}\0{number}".encode("utf-8")
    return "chunk-" + hashlib.sha256(value).hexdigest()


def create_incremental_index(base_name, reused_sources, new_chunks, source_signatures,
                             on_progress=None, validate_before_publish=None, missing_sources=()):
    '''创建可回滚的新快照：复制旧向量，只为新片段计算向量。

    reused_sources必须已由调用方通过文件清单或旧索引逐片段对照确认。
    validate_before_publish在所有写入后、发布前检查资料是否发生变化。
    '''
    _validate_signatures(source_signatures)
    reused_sources = set(reused_sources)
    missing_sources = set(missing_sources)
    if not missing_sources <= reused_sources:
        raise ValueError("缺失文件只能从旧集合复用。")
    if any(source_signatures.get(source) != {"size": 0, "mtime_ns": -1} for source in missing_sources):
        raise ValueError("缺失文件的版本必须标记为未验证。")
    # 与旧版create_index保持一致，空白片段不写入Chroma。
    new_chunks = [chunk for chunk in new_chunks
                  if not (isinstance(chunk, dict) and isinstance(chunk.get("text"), str) and not chunk["text"].strip())]
    new_sources = set()
    for chunk in new_chunks:
        chroma_index._validate_chunk(chunk)
        new_sources.add(chunk["source"])
    if not source_signatures or reused_sources & new_sources:
        raise ValueError("增量更新的资料清单无效。")
    if reused_sources | new_sources != set(source_signatures):
        raise ValueError("增量更新的来源和文件清单不一致。")

    base = None
    base_count = 0
    manifest = None
    if base_name:
        base, base_count = chroma_index._get_ready_collection(base_name)
        manifest = read_index_manifest(base_name)
        if manifest is not None and any(manifest.get(source) != source_signatures[source]
                                        and source not in missing_sources for source in reused_sources):
            raise ValueError("声明复用的文件已发生变化。")
    elif reused_sources:
        raise ValueError("没有旧知识库，不能复用旧片段。")

    # 无任何变化时直接返回旧集合，不重新生成向量也不创建重复快照。
    if base is not None and manifest == source_signatures and not new_chunks and reused_sources == set(manifest):
        if validate_before_publish:
            validate_before_publish()
        return {"name": base_name, "file_count": len(manifest), "chunk_count": base_count,
                "dimension": chroma_index.DIMENSION, "reused_chunk_count": base_count,
                "new_chunk_count": 0, "mode": "unchanged"}

    # 先数出需要复制的旧片段，确认每个声明复用的文件确实在旧库中。
    reused_count = 0
    found_sources = set()
    if base is not None and reused_sources:
        for batch in _old_batches(base, base_count, ["metadatas"]):
            for metadata in batch["metadatas"]:
                if metadata.get("source") in reused_sources:
                    reused_count += 1
                    found_sources.add(metadata["source"])
        if found_sources != reused_sources:
            raise ValueError("旧集合缺少可复用的资料片段。")
    total = reused_count + len(new_chunks)
    if not total:
        raise ValueError("没有可建立索引的资料片段。")

    client = chroma_index._client()
    name = chroma_index._new_name()
    metadata = {
        "ready": False,
        "version": 1,
        "model": chroma_index.MODEL_NAME,
        "preprocessing": chroma_index.PREPROCESSING,
        "dimension": chroma_index.DIMENSION,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "file_count": len(source_signatures),
        "chunk_count": total,
        "source_signatures": json.dumps(source_signatures, ensure_ascii=False, sort_keys=True),
        **_chunking_metadata(),
    }
    collection = client.create_collection(
        name=name, configuration={"hnsw": {"space": "cosine"}},
        metadata=metadata, embedding_function=None,
    )
    done = 0
    identifiers = set()

    def write_batch(chunks, vectors):
        '''按来源和编号重新分配ID，再批量写入片段及向量。'''
        nonlocal done
        if len(chunks) != len(vectors):
            raise ValueError("片段和向量数量不一致。")
        ids, texts, metadatas, checked_vectors = [], [], [], []
        for chunk, vector in zip(chunks, vectors):
            chroma_index._validate_chunk(chunk)
            # Chroma返回的向量可能是NumPy数组；统一转为普通列表。
            vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            chroma_index._validate_vector(vector)
            identifier = _new_chunk_id(chunk["source"], chunk["number"])
            if identifier in identifiers:
                raise ValueError("同一文件的片段编号重复。")
            identifiers.add(identifier)
            item = {key: chunk[key] for key in ("source", "number", "start", "end")}
            if chunk.get("page") is not None:
                item["page"] = chunk["page"]
            ids.append(identifier)
            texts.append(chunk["text"])
            metadatas.append(item)
            checked_vectors.append(vector)
        if ids:
            collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=checked_vectors)
            done += len(ids)
            if on_progress:
                on_progress(done, total)

    try:
        # 复制未修改文件的原始片段和向量；不重新运行嵌入模型。
        if base is not None and reused_sources:
            for batch in _old_batches(base, base_count, ["documents", "metadatas", "embeddings"]):
                selected_chunks, selected_vectors = [], []
                for text, item, vector in zip(batch["documents"], batch["metadatas"], batch["embeddings"]):
                    if item.get("source") in reused_sources:
                        selected_chunks.append({**item, "text": text})
                        selected_vectors.append(vector)
                write_batch(selected_chunks, selected_vectors)
        # 新增及修改的文件才需要重新向量化。
        for start in range(0, len(new_chunks), chroma_index.BATCH_SIZE):
            batch = new_chunks[start:start + chroma_index.BATCH_SIZE]
            vectors = chroma_index.embed_texts([chunk["text"] for chunk in batch])
            write_batch(batch, vectors)
        if done != total or collection.count() != total:
            raise ValueError("新集合中的片段数量不一致。")
        # 用户资料在构建过程中发生变化时，调用方可以阻止发布新快照。
        if validate_before_publish:
            validate_before_publish()
        metadata["ready"] = True
        collection.modify(metadata=metadata)
    except BaseException:
        # 只删除本次未完成的新集合；旧集合仍然可查询。
        try:
            client.delete_collection(name)
        except Exception:
            pass
        raise
    return {"name": name, "file_count": len(source_signatures), "chunk_count": total,
            "dimension": chroma_index.DIMENSION, "reused_chunk_count": reused_count,
            "new_chunk_count": len(new_chunks), "mode": "incremental" if base else "initial"}
