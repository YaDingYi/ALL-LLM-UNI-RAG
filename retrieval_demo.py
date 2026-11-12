'''在本机逐段比较问题和资料，找出相似度最高的片段。'''

import embedding_demo
from embedding_demo import QUERY_PREFIX, embed_texts
from similarity_demo import cosine_similarity

# QUERY_PREFIX 从嵌入模块导入，也保留给此前引用此名称的学习代码使用。
BATCH_SIZE = 16


def search_chunks(question, chunks, top_k=3, on_progress=None):
    '''比较问题和各个片段，按分数从高到低返回结果。'''
    if not question.strip():
        raise ValueError("请输入要检索的问题。")
    if len(question) > 200:
        raise ValueError("本次演示请将问题控制在 200 个字符以内。")
    if top_k < 1:
        raise ValueError("返回数量必须大于 0。")

    # 空白片段没有可比较的内容，将它们排除。
    usable_chunks = [chunk for chunk in chunks if chunk["text"].strip()]
    if not usable_chunks:
        raise ValueError("没有可比较的资料片段，请先读取资料。")

    # 问题只转换一次，后续每个片段都与这个问题向量比较。
    question_vector = embed_texts([QUERY_PREFIX + question])[0]
    scored = []

    # 分批转换片段，避免一次性生成过多向量。
    for start in range(0, len(usable_chunks), BATCH_SIZE):
        batch = usable_chunks[start:start + BATCH_SIZE]
        vectors = embed_texts([chunk["text"] for chunk in batch])

        if len(vectors) != len(batch):
            raise RuntimeError("模型返回的向量数量与片段数量不一致。")

        # 为每个片段计算一个分数，并保留原片段的来源信息。
        for chunk, vector in zip(batch, vectors):
            # 根据嵌入流程的输出标记，决定直接使用点积还是计算完整余弦公式。
            score = cosine_similarity(question_vector, vector, already_normalized=embedding_demo.EMBEDDING_OUTPUT_NORMALIZED)
            scored.append({"chunk": chunk, "score": score})

        # 如果调用者提供进度函数，就报告已完成的片段数量。
        if on_progress is not None:
            on_progress(min(start + len(batch), len(usable_chunks)), len(usable_chunks))

    # 分数越高，向量方向越接近；只取排名靠前的结果。
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k], len(usable_chunks)


def render_retrieval_demo():
    '''显示资料片段比较界面，并处理输入、进度和结果展示。'''
    # 只有显示网页时才导入 Streamlit，检索函数仍可单独使用。
    import streamlit as st

    with st.expander("🔎 学习：从多段资料中找到相关片段"):
        st.caption("先在上方读取资料，再选择比较范围。每次检索都在本机重新生成所选片段的向量；选择全部资料可能较慢，不会调用 DeepSeek。")
        # 从当前会话状态中取出之前保存的资料预览结果，赋值给 loaded，供后续选择和检索使用。
        # 如果 document_preview 不存在，get() 返回 None 而不报错；这一行不会重新读取文件。
        loaded = st.session_state.get("document_preview")

        if not loaded or not loaded["chunks"]:
            st.info("请先展开“读取并切分本地资料”，点击“读取并切分资料”。")
        else:
            # 将“全部资料”与已读取的文件名列表拼接，生成下拉框的完整选项列表。
            choices = ["全部资料"] + loaded["sources"]
            # 如果选项中存在“学习笔记.docx”，就用 index() 获取它的位置，作为下拉框的初始默认位置。
            # 位置编号从 0 开始；如果该文件不存在，就用 0 默认选择第一项“全部资料”。
            default_index = choices.index("学习笔记.docx") if "学习笔记.docx" in choices else 0

            with st.form("retrieval_demo_form"):
                search_question = st.text_input("输入要检索的问题", value="为什么要切分文本？", max_chars=200)
                search_scope = st.selectbox("选择比较范围", choices, index=default_index)
                top_k = st.number_input("显示前几个片段", min_value=1, max_value=10, value=3, step=1)
                search = st.form_submit_button("比较并查找片段")

            # 只有用户提交检索表单时，search 才在本次执行中为 True，才开始下面的检索。
            if search:
                # 删除当前会话中的上次检索结果，避免本次失败时网页仍显示旧结果。
                # pop 的第二个参数 None 表示：如果这个键不存在，返回 None，不会报错。
                st.session_state.pop("retrieval_demo_result", None)

                # 从已读取并切分好的全部片段中，筛选出本次需要比较的片段。
                # 这是列表推导式：依次取出每个 chunk，满足下方条件时加入新列表。
                # 选“全部资料”时保留所有片段；否则只保留来源等于所选文件的片段。
                selected_chunks = [
                    chunk for chunk in loaded["chunks"]
                    if search_scope == "全部资料" or chunk["source"] == search_scope
                ]

                # 在网页上创建初始值为 0 的进度条，并显示准备生成向量的提示。
                # 保存进度条对象，后面可用 progress.progress() 更新，或用 empty() 移除。
                progress = st.progress(0, text="准备生成本地向量…")

                # 尝试执行本地检索；如果其中发生异常，由后面的 except 显示提示。
                try:
                    # 定义进度更新函数：done 是已比较数量，total 是要比较的总数量。
                    # 这里只定义函数；稍后 search_chunks 每完成一批片段时会调用它。
                    def update_progress(done, total):
                        # done / total 得到完成比例；f 字符串将数量填入网页提示文字。
                        # 例如 done=4、total=10 时，进度为 0.4，文字为“已比较 4/10 个片段”。
                        progress.progress(done / total, text=f"已比较 {done}/{total} 个片段")

                    # 在本机将问题和所选片段转换为向量，计算相似度，并按分数排序。
                    # 四个参数依次是问题、所选片段、最多返回数量、用于更新进度的函数。
                    # 传入 update_progress 本身，由 search_chunks 在处理过程中调用它。
                    # results 保存排名靠前的片段及分数；count 保存实际参与比较的片段数量。
                    results, count = search_chunks(search_question, selected_chunks, top_k, update_progress)
                    # 将问题、范围、检索结果和比较数量保存到当前会话，供后面的网页代码显示。
                    # 这里保存到 session_state，不会将结果写入 Histor 文件。
                    st.session_state["retrieval_demo_result"] = {
                        "question": search_question, "scope": search_scope,
                        "results": results, "count": count,
                    }

                # 输入无效或没有可比较的片段时，显示 ValueError 中的具体提示文字。
                except ValueError as error:
                    st.error(str(error))
                # 找不到所需模型文件时，提示检查本地模型目录。
                except FileNotFoundError:
                    st.error("本地向量模型文件不完整，请检查 models/bge-small-zh-v1.5。")
                # 其他常见异常显示异常类型，方便定位问题，同时不展示原始异常内容。
                except Exception as error:
                    st.error(f"片段比较失败（{type(error).__name__}），请检查模型和运行环境。")
                # 无论检索成功还是发生异常，结束时都移除网页上的进度条。
                finally:
                    progress.empty()

        result = st.session_state.get("retrieval_demo_result")

        if result is not None:
            st.text(f"问题：{result['question']} · 比较范围：{result['scope']}")
            st.success(f"已比较 {result['count']} 个片段，按相似度从高到低显示前 {len(result['results'])} 个。")
            st.caption("分数不是概率；排名靠前也不保证片段一定能回答问题。")

            for rank, item in enumerate(result["results"], start=1):
                chunk = item["chunk"]
                page_label = f" · PDF 第 {chunk['page']} 页" if chunk["page"] is not None else ""
                st.text(f"排名 {rank} · 相似度 {item['score']:.4f} · 来源：{chunk['source']}{page_label} · 第 {chunk['number']} 段")
                st.text(chunk["text"])
