'''读取保存的平面向量索引，查找与问题最相关的资料片段。'''

from embedding_demo import MODEL_NAME, embed_texts
from index_demo import list_indexes, load_index
from retrieval_demo import QUERY_PREFIX
from similarity_demo import cosine_similarity


def search_index(question, index, top_k=3):
    '''将问题向量与索引中的已有向量比较，返回排名靠前的片段。'''
    # 检查输入，避免将空问题或无效数量送入模型。
    if not question.strip():
        raise ValueError("请输入要检索的问题。")
    if len(question) > 200:
        raise ValueError("本次演示请将问题控制在 200 个字符以内。")
    if top_k < 1:
        raise ValueError("返回数量必须大于 0。")

    # 问题向量必须与建库时的模型和预处理方式兼容。
    if index.get("version") != 1:
        raise ValueError("不支持这个索引的格式版本，请重新构建索引。")
    if index.get("model") != MODEL_NAME or index.get("preprocessing") != "ascii-lower-v1":
        raise ValueError("索引使用的向量模型或预处理方式与当前配置不一致，请重新构建索引。")

    # 问题前加上 BGE 检索提示，然后只生成一次问题向量。
    question_vector = embed_texts([QUERY_PREFIX + question])[0]
    if len(question_vector) != index["dimension"]:
        raise ValueError("问题向量的维度与索引不一致，请重新构建索引。")

    scored = []
    for record in index["records"]:
        # 使用索引中已经保存的片段向量，不重新生成片段向量。
        vector = record["vector"]
        if not any(vector):
            raise ValueError("索引中存在零向量，无法计算相似度，请重新构建索引。")
        score = cosine_similarity(question_vector, vector)
        # 结果只保存展示需要的字段，不把完整向量再放入网页会话。
        scored.append({
            "id": record["id"], "source": record["source"],
            "page": record["page"], "number": record["number"],
            "text": record["text"], "score": score,
        })

    # 按分数从高到低排序，并返回前 top_k 条和参与比较的总数。
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k], len(scored)


def render_index_search():
    '''显示索引检索界面，并展示排名和片段来源。'''
    import streamlit as st

    with st.expander("🔎 学习：使用本地索引检索片段"):
        st.caption("选择已保存的 JSON 索引；只在本机生成问题向量并比较已有片段向量，不调用 DeepSeek。")

        try:
            filenames = list_indexes()
        except OSError:
            st.error("无法读取 vector_index 文件夹，请检查权限。")
            return

        if not filenames:
            st.info("还没有索引，请先在上方建立并保存本地向量索引。")
            return

        # 选择索引和问题；提交后才读取文件并执行比较。
        with st.form("index_search_form"):
            filename = st.selectbox("选择要检索的索引", filenames)
            question = st.text_input("输入索引检索问题", value="这些资料主要讲什么？", max_chars=200)
            top_k = st.number_input("索引检索返回数量", min_value=1, max_value=10, value=3, step=1)
            search = st.form_submit_button("使用索引检索")

        if search:
            # 清除旧结果，避免失败后把旧结果误认为本次结果。
            st.session_state.pop("index_search_result", None)

            try:
                with st.spinner("正在读取索引并在本机比较向量…"):
                    index = load_index(filename)
                    results, count = search_index(question, index, top_k)

                st.session_state["index_search_result"] = {
                    "filename": filename, "question": question,
                    "results": results, "count": count,
                }
            except (ValueError, OSError) as error:
                st.error(f"索引检索未完成：{error}")
            except Exception as error:
                st.error(f"索引检索未完成（{type(error).__name__}），请检查模型和运行环境。")

        result = st.session_state.get("index_search_result")
        if result is not None and result["filename"] == filename:
            st.text(f"问题：{result['question']} · 索引：{result['filename']}")
            st.success(f"已比较 {result['count']} 个已保存向量，显示前 {len(result['results'])} 个片段。")
            st.caption("分数不是概率；索引是构建时的资料快照，资料变动后需要重新构建。")
            for rank, item in enumerate(result["results"], start=1):
                page_label = f" · PDF 第 {item['page']} 页" if item["page"] is not None else ""
                st.text(f"排名 {rank} · 相似度 {item['score']:.4f} · 来源：{item['source']}{page_label} · 第 {item['number']} 段")
                st.text(item["text"])
