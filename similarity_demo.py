'''在本机比较两段文字的向量相似度。'''

from math import fsum, isfinite, sqrt

# 复用上一步的本地向量模型，不重复加载另一套模型。
import embedding_demo
from embedding_demo import MODEL_NAME, text_to_vector


def cosine_similarity(vector_a, vector_b, *, already_normalized=False):
    '''计算两个非零、同维度向量的余弦相似度。

    余弦相似度公式：

                        A · B                  Σ(aᵢ × bᵢ)
        cos(A, B) = ------------- = --------------------------------
                      ||A|| × ||B||    √Σ(aᵢ²) × √Σ(bᵢ²)

    其中 A、B 分别是两个向量，aᵢ、bᵢ 是相同位置上的FastEmbed数，Σ 表示对所有位置求和。
    A · B 是点积，||A|| 和 ||B|| 分别是两个向量的长度。
    对应代码中的 dot_product、length_a 和 length_b，结果理论上在 -1 到 1 之间。

    already_normalized=True 表示调用者确认两个向量均为非零的单位向量，
    此时直接使用点积，不重复计算长度；默认 False，使用完整的余弦公式。
    该参数是调用者提供的信息，函数不会自动验证向量的长度是否为 1。
    '''
    # 空向量或维度不同的向量无法按位置比较。
    if not vector_a or len(vector_a) != len(vector_b):
        raise ValueError("两个向量必须非空且维度相同。")

    # 如果包含无穷大或非数字，计算结果就不可靠。
    if not all(isfinite(value) for value in vector_a + vector_b):
        raise ValueError("向量中的数字必须是有限数值。")

    # 按位置相乘并求和，得到两个向量的点积。
    dot_product = fsum(a * b for a, b in zip(vector_a, vector_b))

    # 如果调用者确认两个向量已做 L2 归一化，长度均为 1，可以直接使用点积。
    if already_normalized:
        # 不计算长度，也要排除没有方向的零向量；any() 判断是否存在非零元素。
        if not any(vector_a) or not any(vector_b):
            raise ValueError("零向量无法计算余弦相似度。")

        score = dot_product
    else:
        # 不确定是否归一化时，按完整公式计算两个向量的长度。
        length_a = sqrt(fsum(value * value for value in vector_a))
        length_b = sqrt(fsum(value * value for value in vector_b))

        # 零向量没有方向，不能计算余弦相似度。
        if length_a == 0 or length_b == 0:
            raise ValueError("零向量无法计算余弦相似度。")

        score = dot_product / (length_a * length_b)

    # 限制浮点舍入误差，保持结果在理论范围 [-1, 1] 内。
    return max(-1.0, min(1.0, score))


def compare_texts(text_a, text_b):
    '''将两段文字转成向量，并返回文字、向量和相似度。'''
    if not text_a.strip() or not text_b.strip():
        raise ValueError("请输入两段有内容的文字。")

    # 两次调用会复用 embedding_demo 中缓存的同一个本地模型。
    vector_a = text_to_vector(text_a)
    vector_b = text_to_vector(text_b)

    # 根据嵌入流程的输出标记，决定直接使用点积还是计算完整余弦公式。
    score = cosine_similarity(vector_a, vector_b, already_normalized=embedding_demo.EMBEDDING_OUTPUT_NORMALIZED)

    return {
        "text_a": text_a,
        "text_b": text_b,
        "vector_a": vector_a,
        "vector_b": vector_b,
        "score": score,
    }


def render_similarity_demo():
    '''显示两段文字的相似度演示，并处理输入和结果展示。'''
    # 仅显示网页时导入 Streamlit；相似度计算函数仍可单独使用。
    import streamlit as st

    with st.expander("📐 学习：比较两段文字的相似度"):
        st.caption("两段文字均在本机处理。分数越高表示向量方向越接近；它不是概率，也不保证事实正确。")

        with st.form("similarity_demo_form"):
            text_a = st.text_area("第一段文字", value="北京是中国的首都。", max_chars=200)
            text_b = st.text_area("第二段文字", value="中国的首都是北京。", max_chars=200)
            compare = st.form_submit_button("计算相似度")

        if compare:
            # 清除上次结果，避免本次出错时误显示旧结果。
            st.session_state.pop("similarity_demo_result", None)
            try:
                with st.spinner("正在本机生成向量并计算相似度…"):
                    comparison = compare_texts(text_a, text_b)
                # 保存在当前会话中，页面重新执行时仍可查看。
                st.session_state["similarity_demo_result"] = comparison
            except ValueError as error:
                st.error(str(error))
            except FileNotFoundError:
                st.error("本地向量模型文件不完整，请检查 models/bge-small-zh-v1.5。")
            except Exception as error:
                st.error(f"相似度计算失败（{type(error).__name__}），请检查模型依赖和运行环境。")

        comparison = st.session_state.get("similarity_demo_result")
        if comparison is not None:
            st.text(f"第一段：{comparison['text_a']}")
            st.text(f"第二段：{comparison['text_b']}")
            st.metric("余弦相似度", f"{comparison['score']:.4f}")
            st.caption(f"模型：{MODEL_NAME}；两个向量均为 {len(comparison['vector_a'])} 维。理论范围是 -1 到 1。")

            # 先显示少量数字；勾选后可查看完整向量。
            st.text(f"向量 A 前 8 个数字：{comparison['vector_a'][:8]}")
            st.text(f"向量 B 前 8 个数字：{comparison['vector_b'][:8]}")
            if st.checkbox("显示两个完整向量"):
                st.text_area("完整向量 A", value=str(comparison["vector_a"]), height=120, disabled=True)
                st.text_area("完整向量 B", value=str(comparison["vector_b"]), height=120, disabled=True)
