# 导入定位配置文件、读取配置和调用模型所需的工具。
from pathlib import Path
from copy import deepcopy
from dotenv import dotenv_values
from openai import OpenAI
import httpx
import streamlit as st
from history import create_history, list_history_files, load_history, save_history
from document_reader import render_document_preview
from embedding_demo import render_embedding_demo
from similarity_demo import render_similarity_demo
from retrieval_demo import render_retrieval_demo
from index_demo import list_indexes, render_index_demo
from index_search import render_index_search
from rag_demo import RetrievalError, prepare_rag_messages
from upload_ui import render_upload_ui
from rag_trace import render_rag_trace

# 对可重试的请求错误，除首次请求外最多再尝试 5 次。
MAX_RETRIES = 5

# 设置浏览器标签页和页面布局。
st.set_page_config(page_title="知识库助手 · 学习版", page_icon="📚", layout="wide")

# 显示网页标题与说明。
st.title("📚 我的知识库助手")
st.caption("第七步：上传资料并更新知识库")

# 显示配置的重试上限；SDK 在后台重试，不逐次显示当前次数。
st.caption(f"请求遇到可重试错误时，最多自动重试 {MAX_RETRIES} 次。")

# 在页面顶部提供上传入口；具体处理位于 upload_ui.py。
render_upload_ui()

# 显示单段文字向量化演示；具体实现位于 embedding_demo.py。
render_embedding_demo()

# 显示两段文字的相似度演示；具体实现位于 similarity_demo.py。
render_similarity_demo()

# 显示本地资料读取和切分预览；具体实现位于 document_reader.py。
render_document_preview()

# 显示资料片段比较演示；具体实现位于 retrieval_demo.py。
render_retrieval_demo()

# 显示本地向量索引构建与查看界面。
render_index_demo()

# 使用已保存的本地索引查找相关片段。
render_index_search()

# 读取当前项目目录中的配置文件；Key 不显示在页面上。
config_path = Path(__file__).resolve().parent / ".env.local"
config = dotenv_values(config_path)
api_key = (config.get("DEEPSEEK_API_KEY") or "").strip()

# 新的浏览器会话先从空对话开始，首次提问时才创建历史文件。
if "active_history_file" not in st.session_state:
    st.session_state["active_history_file"] = ""
    st.session_state["conversation"] = []

# 创建新文件后，在下一次运行中选中它，保持下拉框与当前对话一致。
pending_file = st.session_state.pop("pending_history_file", None)
if pending_file is not None:
    st.session_state["history_choice"] = pending_file

try:
    history_files = list_history_files()
except OSError:
    st.error("无法读取 Histor 文件夹，请检查目录权限。")
    st.stop()

# 空字符串表示新对话；其他选项是 Histor 中的 JSON 文件名。
selected_file = st.selectbox(
    "选择对话历史",
    options=[""] + history_files,
    format_func=lambda name: "🆕 新对话" if not name else name,
    key="history_choice",
)

# 比较下拉框选中的文件和当前正在使用的文件。
# 只有选项发生变化时才重新加载，避免每次页面重新执行都覆盖当前会话。
if selected_file != st.session_state["active_history_file"]:
    # 尝试读取所选历史文件，读取出错时交给下方 except 处理。
    try:
        # 有文件名时读取对应历史；空字符串表示选择“新对话”，使用空列表。
        # 等价于：if selected_file: messages = load_history(selected_file)
        #         else: messages = []
        messages = load_history(selected_file) if selected_file else []

    # 捕获文件操作错误或历史内容格式错误，例如文件不存在或 JSON 无效。
    except (OSError, ValueError):
        # 在网页显示错误提示，不修改原历史文件。
        st.error("所选历史记录无法读取；原文件未被修改，请检查文件内容和权限。")

        # 停止本次脚本执行，避免使用错误数据继续显示；不会关闭网页服务。
        st.stop()

    # 把成功读取的消息列表（或新对话的空列表）设为当前对话内容。
    st.session_state["conversation"] = messages

    # 记录当前选中的文件名，下次运行时用它判断选项是否发生变化。
    st.session_state["active_history_file"] = selected_file

    # 清除上一段对话的提示；如果没有 chat_notice，默认返回 None，不会报错。
    st.session_state.pop("chat_notice", None)
    # 切换对话后，不能把上一段对话的检索过程当作当前对话。
    st.session_state.pop("rag_trace", None)

if not st.session_state["active_history_file"]:
    st.caption("当前是新对话；提交第一个问题后会创建历史文件。")

if "chat_notice" in st.session_state:
    level, message = st.session_state["chat_notice"]

    if level == "error":
        st.error(message)
    else:
        st.warning(message)

# 如果没有填写 Key，提示用户并暂停输入。
if not api_key:
    st.warning("请先在当前项目的 .env.local 中填写 DEEPSEEK_API_KEY。")

def show_message(message):
    '''显示一条消息；助手回答还显示保存的模型名称和 Token 用量。'''
    with st.chat_message(message["role"]):
        st.text(message["content"])
        if message["role"] == "assistant":
            # 模型名称来自接口返回的 response.model，不是根据回答内容推测的。
            model_name = message.get("model")
            if isinstance(model_name, str) and model_name.strip():
                st.caption(f"返回模型：{model_name}")
            elif "model" in message:
                st.caption("本次请求未返回模型名称。")
            # 旧历史可能没有 usage；新回答在接口未返回用量时会保存 None。
            usage = message.get("usage")
            if isinstance(usage, dict) and all(
                key in usage for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                st.caption(
                    f"本次 Token 用量：输入 {usage['prompt_tokens']}，"
                    f"输出 {usage['completion_tokens']}，总计 {usage['total_tokens']}"
                )
            elif "usage" not in message:
                st.caption("历史记录未保存 Token 用量。")
            else:
                st.caption("本次请求的 Token 用量不可用。")

            # 参考片段保存在助手消息中，重新加载历史时也能查看。
            sources = message.get("sources")
            if isinstance(sources, list) and sources:
                with st.expander("查看本次参考片段"):
                    for rank, source in enumerate(sources, start=1):
                        if not isinstance(source, dict):
                            continue
                        page = f" · PDF 第 {source['page']} 页" if source.get("page") is not None else ""
                        score = source.get("score")
                        score_text = f" · 相似度 {score:.4f}" if isinstance(score, (int, float)) else ""
                        st.text(f"[{rank}] 来源：{source.get('source', '未知')}{page}{score_text}")
                        st.text(source.get("text", ""))


# 按顺序显示已保存的用户消息和助手回答。
for message in st.session_state["conversation"]:
    show_message(message)

# 查找已经保存的索引；目录无法读取时，普通对话仍可使用。
try:
    available_indexes = list_indexes()
except OSError:
    available_indexes = []
    st.warning("无法读取向量索引目录，请检查目录权限。")

# 已有索引时默认进入知识库问答，也可以切换回普通对话。
chat_mode = st.radio(
    "对话模式", ["普通对话", "知识库问答"],
    index=1 if available_indexes else 0, horizontal=True, key="chat_mode_choice",
)
rag_index = None
if chat_mode == "知识库问答":
    if available_indexes:
        rag_index = st.selectbox("选择问答使用的索引", available_indexes, key="rag_index_choice")
        st.caption("会将本次检索出的资料片段及来源、当前问题和此前对话发送给 DeepSeek；问题最多 200 字。")
    else:
        st.info("请先在上方建立本地向量索引，再使用知识库问答。")

# 没有 Key 或知识库模式没有索引时，禁用输入框。
question = st.chat_input(
    "请输入你的问题", max_chars=200 if chat_mode == "知识库问答" else None,
    disabled=not api_key or (chat_mode == "知识库问答" and not rag_index),
)

# 用户提交问题后，先保存和显示问题，再请求模型回答。
# 如果 question 为空，跳过这一整段，不发送请求。
if question:
    st.session_state.pop("chat_notice", None)
    # 新问题开始时清除上一次过程，检索失败时也不会显示旧结果。
    st.session_state.pop("rag_trace", None)

    created_new_file = False
    # 把问题追加到当前会话：role 标记消息来自用户，content 保存问题文字。
    st.session_state["conversation"].append({"role": "user", "content": question})

    # 先把问题保存到本地；保存失败时撤回本次内存中的追加并停止请求。
    try:
        if st.session_state["active_history_file"]:
            save_history(st.session_state["active_history_file"], st.session_state["conversation"])
        else:
            # 文件名由本地日期和首个问题组成，已有同名文件时不会覆盖。
            filename = create_history(question, st.session_state["conversation"])
            st.session_state["active_history_file"] = filename
            st.session_state["pending_history_file"] = filename
            created_new_file = True
    except (OSError, ValueError):
        st.session_state["conversation"].pop()
        st.error("问题未能保存到 Histor，请检查目录权限和磁盘空间后重试。")
        st.stop()

    # 创建用户消息容器，并在其中显示刚刚提交的问题。
    with st.chat_message("user"):
        st.text(question)

    # 尝试创建客户端并请求回答；发生错误时由下面的 except 处理。
    try:
        # 知识库模式先查找资料；普通模式只整理原有对话。
        references = None
        if chat_mode == "知识库问答":
            with st.spinner("正在本地索引中查找资料…"):
                api_messages, references = prepare_rag_messages(
                    question, st.session_state["conversation"], rag_index
                )
            # 保存本次实际准备的内容，而不是根据之后的界面选择重新计算。
            st.session_state["rag_trace"] = {
                "question": question, "index_filename": rag_index,
                "references": deepcopy(references), "messages": deepcopy(api_messages),
                "request_status": "已准备消息，尚未调用模型接口",
            }
        else:
            # 模型接口只接受角色和内容，不发送本地保存的用量等字段。
            api_messages = [
                {"role": message["role"], "content": message["content"]}
                for message in st.session_state["conversation"]
            ]

        # 创建 DeepSeek 客户端；with 在离开代码块时关闭客户端并释放连接。
        with OpenAI(
            # 使用从当前项目 .env.local 中读取到的 API Key。
            api_key=api_key,
            # 指定请求发往 DeepSeek 的接口，而不是默认接口。
            base_url="https://api.deepseek.com",
            # 设置请求的超时时间为 60 秒。
            timeout=60,
            # 可重试的请求错误最多自动重试 MAX_RETRIES 次。
            max_retries=MAX_RETRIES,
            # 显式提供 HTTP 客户端，避免当前环境中 openai 与 httpx 的版本兼容问题。
            http_client=httpx.Client(timeout=60),
        ) as client:
            # 等待模型回答时，在网页上显示提示。
            with st.spinner("DeepSeek 正在回答…"):
                if chat_mode == "知识库问答":
                    st.session_state["rag_trace"]["request_status"] = "正在尝试调用模型接口"
                # 发送对话记录并等待模型返回结果，保存到 response。
                response = client.chat.completions.create(
                    # 选择 DeepSeek 的 deepseek-flash 模型。
                    model="deepseek-flash",
                    # 发送整理后的消息，使模型能参考之前的对话。
                    messages=api_messages,
                    # 限制本次回答最多生成 512 个 token。
                    max_tokens=512,
                    # 控制非思考模式下的随机程度；思考模式开启时此参数不生效。
                    temperature=0.1,
                    # 通过 extra_body 传入 DeepSeek 参数，兼容当前较旧的 OpenAI SDK。
                    extra_body={
                        # 保持思考模式开启；可改为 disabled 关闭。
                        "thinking": {"type": "enabled"},
                        # 思考强度可设置为 low、high 或 max；当前使用 max。
                        "reasoning_effort": "max",
                    },
                )
                if chat_mode == "知识库问答":
                    st.session_state["rag_trace"]["request_status"] = "已收到模型响应"

        # response 是本次非流式 Chat Completions 返回的对象，而不是请求参数。
        # 顶层字段：
        #   id                 本次回答的唯一标识。
        #   object             返回对象类型，通常为 "chat.completion"。
        #   created            回答创建时间，Unix 时间戳（秒）。
        #   model              服务端返回的模型名称。
        #   system_fingerprint 模型运行时的后端配置指纹。
        #   choices            候选回答列表；本程序使用第一项 choices[0]。
        #   usage              本次请求的 token 用量；可能为空或不存在。
        #
        # choices 中每项的字段：
        #   index              该候选回答在列表中的序号。
        #   finish_reason      停止原因，如 stop（正常结束）、length（达到长度限制）、
        #                      content_filter、tool_calls、insufficient_system_resource 或 aborted。
        #   message            助手生成的消息对象。
        #   logprobs           可选的 token 对数概率信息，未提供时可能为 None；
        #                      其 content / reasoning_content 包含 token、logprob、bytes、
        #                      top_logprobs 等信息，当前程序不使用。
        #
        # message 中的字段：
        #   role               消息角色，助手回答通常是 "assistant"。
        #   content            最终回答文本；可能为空或 None。
        #   reasoning_content  思考模式返回的思考内容；未提供时可能为空或不存在。
        #   tool_calls         可选的工具调用列表；每项包含 id、type，以及 function 的
        #                      name 和 arguments（JSON 格式的字符串）。
        #
        # usage 中的字段：
        #   prompt_tokens                   输入 token 数。
        #   completion_tokens               生成的 token 数。
        #   total_tokens                    输入和生成 token 的总数。
        #   prompt_tokens_details.cached_tokens  输入中命中缓存的 token 数。
        #   prompt_cache_hit_tokens         输入中命中缓存的 token 数。
        #   prompt_cache_miss_tokens        输入中未命中缓存的 token 数。
        #   completion_tokens_details.reasoning_tokens  思考过程使用的 token 数。
        #
        # SDK 还可能定义 service_tier、message.refusal、message.audio、
        # message.function_call（旧版函数调用）等兼容字段；DeepSeek 不一定返回。
        # 可选字段和详情对象可能不存在或为 None，不能假设每次都能直接访问。
        # 具体结构以 DeepSeek 文档和当前 SDK 版本为准：
        # https://api-docs.deepseek.com/api/create-chat-completion/
        # 从第一条候选回答的 message 中取出最终回答文本。

        answer = response.choices[0].message.content

        # 读取 API 返回的模型名称；如果没有返回，则记为 None。
        response_model = getattr(response, "model", None)
        if not isinstance(response_model, str):
            response_model = None

        # 如果模型没有返回文本，主动抛出异常，交给下方 except 处理。
        if not answer:
            raise ValueError("模型没有返回文本")

        # 将接口返回的用量对象转换为可以写入 JSON 的字典。
        usage = getattr(response, "usage", None)

        usage_data = None

        if usage is not None:
            usage_data = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

        # 将回答、模型名称和用量放在同一条助手消息中，方便以后一起读取。
        answer_message = {
            "role": "assistant", "content": answer,
            "model": response_model, "usage": usage_data,
        }

        # 保存实际提供给模型的资料片段，方便以后查看当时的依据。
        if references is not None:
            answer_message["sources"] = references
        st.session_state["conversation"].append(answer_message)

        # 将包含新回答的完整对话保存到本地；失败时仍显示已生成的回答。
        try:
            save_history(st.session_state["active_history_file"], st.session_state["conversation"])
        except (OSError, ValueError):
            notice = "回答已生成，但未能保存到 Histor；页面关闭或会话重建后可能丢失。"
            st.session_state["chat_notice"] = ("warning", notice)
            st.warning(notice)

        # 与加载历史时使用同一函数，显示回答和对应的用量。
        show_message(answer_message)

    # 检索失败时不发送模型请求，在网页上显示本地错误提示。
    except RetrievalError as error:
        notice = str(error)
        st.session_state["chat_notice"] = ("error", notice)
        st.error(notice)

    # 捕获请求或处理回答时发生的异常，避免网页因异常中断。
    except Exception as error:
        trace = st.session_state.get("rag_trace")
        if trace and trace["request_status"] == "正在尝试调用模型接口":
            trace["request_status"] = "模型请求未完成，无法确定服务端是否收到"
        # 只显示异常类型，不展示原始错误详情；请求失败时用户问题仍保留在会话中。
        notice = f"本次回答未完成（{type(error).__name__}），请检查配置、网络和账户额度后重试。"
        st.session_state["chat_notice"] = ("error", notice)
        st.error(notice)

    # 新文件创建后刷新下拉框；对话已保存，刷新不会重复请求模型。
    if created_new_file:
        st.rerun()

# 显示最近一次知识库问答的过程；普通重跑不会重新发送请求。
render_rag_trace()
