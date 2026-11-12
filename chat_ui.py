'''显示聊天消息、模型用量及知识库参考片段。'''

import streamlit as st


def show_message(message):
    '''显示一条消息；助手回答还显示保存的模型名称和 Token 用量。'''
    with st.chat_message(message["role"]):
        st.text(message["content"])
        if message["role"] == "assistant":
            # 达到长度限制时保留已有正文，并提醒回答可能尚未完成；历史也能显示此提示。
            if message.get("finish_reason") == "length":
                st.warning("回答因长度限制提前结束，内容可能不完整；可缩短问题或对话历史后重试。")
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


