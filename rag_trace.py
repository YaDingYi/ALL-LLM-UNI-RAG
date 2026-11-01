'''展示最近一次知识库问答的检索结果和模型消息。'''

import json
import streamlit as st


def render_rag_trace():
    '''从当前网页会话读取过程快照，不重新检索或请求模型。'''
    with st.expander("🔬 学习：观察本次知识库问答过程"):
        trace = st.session_state.get("rag_trace")

        if trace is None:
            st.info("提交一个知识库问题后，可在这里查看本次检索和模型消息。")
            return

        st.caption("此过程仅保存在当前网页会话中；新提问或切换对话后会清除。回答在对应聊天消息中查看。")
        st.text(f"本次问题：{trace['question']}")
        st.text(f"本次使用的索引：{trace['index_filename']}")
        st.text(f"模型请求状态：{trace['request_status']}")

        st.write("**第一步：检索到的资料片段**")
        st.caption("按相似度从高到低排列；分数不是概率，也不随片段发送给模型。")
        for rank, reference in enumerate(trace["references"], start=1):
            page = f" · PDF 第 {reference['page']} 页" if reference["page"] is not None else ""
            st.text(f"[{rank}] 相似度 {reference['score']:.4f} · 来源：{reference['source']}{page} · 第 {reference['number']} 段")
            st.text(reference["text"])

        st.write("**第二步：准备给模型的消息**")
        st.caption("messages 包含 system 要求、此前对话及本次资料和问题，可能包含私人内容；不会附带配置的 API Key。")
        if st.checkbox("显示本次模型消息的完整内容"):
            # 直接显示保存的 messages 快照；不会显示请求凭证或重新调用模型。
            st.code(json.dumps(trace["messages"], ensure_ascii=False, indent=2), language="json")
