'''在本机把一段文字转换成向量，用于学习 Embedding。'''

from functools import lru_cache
from pathlib import Path

# 本地模型名称和模型文件所在目录。
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "bge-small-zh-v1.5"

# 标记当前模型及调用流程输出的向量是否已做 L2 归一化。
# 当前 FastEmbed 流程已归一化，因此为 True；换模型或调用方式时应重新确认。
# 不确定或输出未归一化时设为 False，让相似度函数使用完整公式。
EMBEDDING_OUTPUT_NORMALIZED = True


@lru_cache(maxsize=1)
def load_model():
    '''加载本机模型；重复调用时复用已加载的模型。'''
    # 模型文件不完整时直接报错，不自动下载或请求远程服务。
    required = ("model_optimized.onnx", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")

    if not all((MODEL_DIR / name).is_file() for name in required):
        raise FileNotFoundError("本地向量模型文件不完整，请检查 models/bge-small-zh-v1.5。")

    # 只有首次生成向量时才导入并加载模型。
    import onnxruntime
    from fastembed import TextEmbedding

    # 关闭 ONNX Runtime 的遥测事件，使用本机 CPU 执行推理。
    onnxruntime.disable_telemetry_events()

    model = TextEmbedding(
        model_name=MODEL_NAME,
        specific_model_path=str(MODEL_DIR),
        cache_dir=str(MODEL_DIR.parent),
        local_files_only=True,
        providers=["CPUExecutionProvider"],
        threads=2,
    )
    return model


def embed_texts(texts):
    '''把文字批量转换为向量，按模型返回顺序输出，不额外进行 L2 归一化。'''
    if not texts:
        return []
    # 与原项目保持一致：仅把英文大写字母转成小写，原文不被修改。
    table = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")

    normalized_texts = [text.translate(table) for text in texts]

    # 使用已缓存的本地模型，每批最多处理 16 段文字。
    model = load_model()

    # 将模型返回的数组转为普通列表，保留原始数值，不额外改变向量长度。
    vectors = [vector.tolist() for vector in model.embed(normalized_texts, batch_size=16)]

    # 每段文字应对应一个向量；数量不一致时不能可靠地对应原文。
    if len(vectors) != len(texts):
        raise ValueError("模型返回的向量数量与输入文字数量不一致。")

    # 按输入顺序返回模型输出的向量。
    return vectors


def text_to_vector(text):
    '''将一段非空文字转换为浮点数列表；不会调用 DeepSeek。'''
    if not text.strip():
        raise ValueError("请输入有内容的文字。")
    if len(text) > 200:
        raise ValueError("本次演示请将文字控制在 200 个字符以内。")

    # 复用批量函数，取出第一段文字对应的向量。
    return embed_texts([text])[0]


def render_embedding_demo():
    '''显示单段文字向量化演示，并处理输入和结果展示。'''
    # 只有显示网页时才需要 Streamlit；模型函数仍可单独使用。
    import streamlit as st

    with st.expander("🔢 学习：把一段文字转换为向量"):
        st.caption("使用本地中文模型，在 CPU 上生成向量。文字仅在本机处理，本次演示最多 200 个字符。")

        # 表单可以先输入文字，再通过按钮一次性提交。
        with st.form("embedding_demo_form"):
            embedding_text = st.text_area("输入要转换的文字", value="北京是中国的首都。", max_chars=200)
            generate_vector = st.form_submit_button("生成向量")

        if generate_vector:
            # 清除上一次结果，避免本次出错时误显示旧结果。
            st.session_state.pop("embedding_demo_result", None)

            try:
                with st.spinner("正在本机生成向量，首次加载模型可能稍慢…"):
                    vector = text_to_vector(embedding_text)

                # 保留输入文字和结果，避免其他网页操作后立即消失。
                st.session_state["embedding_demo_result"] = {"text": embedding_text, "vector": vector}
            except ValueError as error:
                st.error(str(error))
            except FileNotFoundError:
                st.error("本地向量模型文件不完整，请检查项目中的 models/bge-small-zh-v1.5。")
            except Exception as error:
                st.error(f"本地向量生成失败（{type(error).__name__}），请检查模型依赖和运行环境。")

        result = st.session_state.get("embedding_demo_result")
        if result is not None:
            st.text(f"本次文字：{result['text']}")
            st.text(f"本地模型：{MODEL_NAME}")
            st.success(f"向量维度：{len(result['vector'])}（即 {len(result['vector'])} 个数字）")
            # 文本框允许滚动查看全部数字，不把向量写入聊天记录。
            st.text_area("完整向量（可滚动查看）", value=str(result["vector"]), height=180, disabled=True)
