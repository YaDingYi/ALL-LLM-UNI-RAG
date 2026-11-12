'''Streamlit 入口：上传并更新知识库、管理对话历史、调用 DeepSeek 回答。

页面每次交互会重新执行本文件；对话和选择状态由 st.session_state 保存。
知识库构建、检索和历史文件读写分别委托给对应模块，完整流程见 README。
'''

# command line: python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8503
# 导入定位配置文件、读取配置和调用模型所需的工具。
from pathlib import Path
from dotenv import dotenv_values
from openai import NOT_GIVEN, OpenAI
import httpx
import streamlit as st
from history import create_history, list_history_files, load_history, save_history
from chat_ui import show_message
from chroma_index import list_indexes
from rag_demo import RetrievalError, prepare_rag_messages
from upload_ui import render_upload_ui

# 对可重试的请求错误，除首次请求外最多再尝试 5 次。
MAX_RETRIES = 5
# 思考过程也占用输出 Token；知识库问答通常更复杂，给予更高的输出上限。
CHAT_MAX_TOKENS = 512
RAG_MAX_TOKENS = 8192


class EmptyModelAnswerError(Exception):
    '''模型没有返回可显示的最终回答。'''

# 一、设置网页并显示上传与知识库更新入口。
# 设置浏览器标签页和页面布局。
st.set_page_config(page_title="知识库助手 · 学习版", page_icon="📚", layout="wide")

# 显示网页标题与说明。
st.title("📚 我的知识库助手")
st.caption("上传资料，选择对话模式，开始提问。")

# 在页面顶部提供上传入口；具体处理位于 upload_ui.py。
render_upload_ui()

# 二、读取 API Key，并初始化或切换本地对话历史。
# 读取当前项目目录中的配置文件；Key 不显示在页面上。
config_path = Pat
h(__file__).resolve().parent / ".env.local"
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

# 三、使用统一显示函数呈现历史消息及回答。
# 按顺序显示已保存的用户消息和助手回答。
for message in st.session_state["conversation"]:
    show_message(message)

# 四、选择普通对话或知识库问答，以及本次要使用的知识库。
# 查找已经完成的 Chroma 集合；数据库不可用时，普通对话仍可使用。
try:
    available_indexes = list_indexes()
except Exception:
    available_indexes = []
    st.warning("无法读取本地 Chroma 知识库，请检查依赖和目录权限。")

# 已有索引时默认进入知识库问答，也可以切换回普通对话。
chat_mode = st.radio(
    "对话模式", ["普通对话", "知识库问答"],
    index=1 if available_indexes else 0, horizontal=True, key="chat_mode_choice",
)
rag_index = None
if chat_mode == "知识库问答":
    if available_indexes:
        rag_index = st.selectbox("选择问答使用的知识库", available_indexes, key="rag_index_choice")
        st.caption("会将本次检索出的资料片段及来源、当前问题和此前对话发送给 DeepSeek；问题最多 200 字。")
    else:
        st.info("请先在上方上传资料并更新知识库，再使用知识库问答。")

# 没有 Key 或知识库模式没有索引时，禁用输入框。
question = st.chat_input(
    "请输入你的问题", max_chars=200 if chat_mode == "知识库问答" else None,
    disabled=not api_key or (chat_mode == "知识库问答" and not rag_index),
)

# 五、提交问题后保存历史；知识库模式先检索，再调用 DeepSeek 并保存回答。
# 用户提交问题后，先保存和显示问题，再请求模型回答。
# 如果 question 为空，跳过这一整段，不发送请求。
if question:
    st.session_state.pop("chat_notice", None)

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
                # 发送对话记录并等待模型返回结果，保存到 response。
                response = client.chat.completions.create(
                    # model 和 messages 是必填参数，其余是 DeepSeek 的可选参数。
                    model="deepseek-flash",
                    messages=api_messages,

                    # 思考内容和最终回答共用输出额度；知识库问答使用较高上限。
                    max_tokens=RAG_MAX_TOKENS if chat_mode == "知识库问答" else CHAT_MAX_TOKENS,
                    # 温度范围 0～2；思考模式下不生效。
                    temperature=0.1,

                    # NOT_GIVEN 表示 SDK 不发送此参数；需要时可替换为有效值。
                    # 核采样，范围大于 0 且不超过 1；实际效果取决于模型和模式。
                    top_p=NOT_GIVEN,
                    # 输出格式：例如 {"type": "text"} 或 {"type": "json_object"}。
                    # JSON 模式还需要在消息中要求输出 JSON。
                    response_format=NOT_GIVEN,
                    # 停止序列：一个字符串或最多 16 个字符串；遇到时停止生成。
                    stop=NOT_GIVEN,

                    # 是否返回流式数据。当前后续代码按非流式对象处理，不要直接设为 True。
                    stream=False,
                    # 流式选项例如 {"include_usage": True}；仅能与 stream=True 配合，
                    # 启用流式还需要改写响应读取和 Token 用量处理。
                    stream_options=NOT_GIVEN,

                    # 工具定义和调用方式。启用前需实现工具执行、结果回传及思考内容处理。
                    tools=NOT_GIVEN,
                    # 可选 none、auto、required 或指定函数；思考模式不支持后两种强制调用。
                    tool_choice=NOT_GIVEN,

                    # 是否返回输出 token 的对数概率；当前网页不展示这些数据。
                    logprobs=NOT_GIVEN,
                    # 每个位置返回的候选 token 数量 0～20；设置时必须同时 logprobs=True。
                    top_logprobs=NOT_GIVEN,

                    # 这两项在 DeepSeek 当前接口中已弃用，发送也不生效。
                    frequency_penalty=NOT_GIVEN,
                    presence_penalty=NOT_GIVEN,

                    # DeepSeek 专有字段通过 extra_body 传入请求体。
                    extra_body={
                        # 思考模式开关：enabled 或 disabled；显式设置时决定开关。
                        "thinking": {"type": "enabled"},
                        # 可选 none、low、high、max；当前 max。
                        "reasoning_effort": "max",
                        # 可选自定义用户标识，例如 "user_001"；不要填写私人信息。
                        "user_id": NOT_GIVEN,
                    },
                )

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
        finish_reason = response.choices[0].finish_reason

        # 读取 API 返回的模型名称；如果没有返回，则记为 None。
        response_model = getattr(response, "model", None)
        if not isinstance(response_model, str):
            response_model = None

        # 未返回正文时，根据结束原因给出明确提示，不把它误报为网络或 Key 错误。
        if not answer:
            if finish_reason == "length":
                raise EmptyModelAnswerError("模型达到输出长度或上下文限制，尚未生成最终回答。请缩短问题或对话历史后重试，或调整 max_tokens。")
            raise EmptyModelAnswerError("模型未返回最终回答，请重试或检查模型配置。")

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
            "model": response_model, "usage": usage_data, "finish_reason": finish_reason,
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

    # 只展示程序自行生成的空回答提示，不暴露请求或配置的原始内容。
    except EmptyModelAnswerError as error:
        notice = str(error)
        st.session_state["chat_notice"] = ("error", notice)
        st.error(notice)

    # 捕获请求或处理回答时发生的异常，避免网页因异常中断。
    except Exception as error:
        # 只显示异常类型，不展示原始错误详情；请求失败时用户问题仍保留在会话中。
        notice = f"本次回答未完成（{type(error).__name__}），请检查配置、网络和账户额度后重试。"
        st.session_state["chat_notice"] = ("error", notice)
        st.error(notice)

    # 新文件创建后刷新下拉框；对话已保存，刷新不会重复请求模型。
    if created_new_file:
        st.rerun()

