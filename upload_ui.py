'''在网页中上传资料，并将当前资料重新建立为本地索引。'''

import streamlit as st

from knowledge_upload import UploadError, save_uploaded_files, update_knowledge_index


def clear_old_previews():
    '''资料变化后移除当前网页会话中的旧预览和检索结果。'''
    for key in ("document_preview", "retrieval_demo_result", "index_search_result",
                "index_demo_preview", "last_created_index", "preview_source"):
        st.session_state.pop(key, None)

    for key in list(st.session_state):
        if key.startswith("chunk_page::"):
            del st.session_state[key]


def render_upload_ui():
    '''显示上传和更新按钮，成功后选择新索引用于知识库问答。'''
    with st.expander("📤 上传资料并更新知识库", expanded=True):
        st.caption("支持 TXT、Markdown、PDF 和 Word（.docx）。选择文件后点击上传按钮；文件保存到资料/上传。")
        st.caption("更新会重新读取全部资料并生成所有片段的向量，资料较多时可能耗时；上传和建索引都在本机完成。")

        files = st.file_uploader(
            "选择要上传的资料", type=["txt", "md", "markdown", "pdf", "docx"],
            accept_multiple_files=True,
        )

        upload = st.button("上传并更新知识库")
        rebuild = st.button("重新读取资料并更新知识库")

        if upload or rebuild:
            st.session_state.pop("upload_update_status", None)
            saved = []
            progress = None

            try:
                # 只有点击上传按钮时才保存文件；同名文件不会被覆盖。
                if upload:
                    saved = save_uploaded_files(files)

                # 资料可能已经改变，旧预览不再代表当前资料。
                clear_old_previews()
                progress = st.progress(0, text="正在读取资料，准备生成向量…")

                def update_progress(done, total):
                    progress.progress(done / total, text=f"已向量化 {done}/{total} 个片段")

                # 上传的文件必须能读取正文，否则不生成新的索引。
                result = update_knowledge_index(required_sources=saved, on_progress=update_progress)

                # 在下面的聊天控件创建之前，切换模式并选中新索引。
                st.session_state["chat_mode_choice"] = "知识库问答"
                st.session_state["rag_index_choice"] = result["filename"]
                st.session_state["upload_update_status"] = {
                    "ok": True, "saved": saved, "result": result,
                }
            except UploadError as error:
                st.session_state["upload_update_status"] = {
                    "ok": False, "saved": saved, "message": str(error),
                }

            except Exception as error:
                st.session_state["upload_update_status"] = {
                    "ok": False, "saved": saved,
                    "message": f"更新未完成（{type(error).__name__}），请检查文件、模型和目录权限。",
                }
            finally:
                if progress is not None:
                    progress.empty()

        status = st.session_state.get("upload_update_status")

        if status:
            if status["saved"]:
                st.text("本次已保存：" + "、".join(status["saved"]))
            if status["ok"]:
                result = status["result"]
                st.success(f"知识库已更新：读取 {result['file_count']} 个文件，生成 {result['chunk_count']} 个片段；更新时自动选用新索引 {result['filename']}。")
                for warning in result["warnings"]:
                    st.warning(f"{warning['source']}：{warning['message']}")
            else:
                st.error(status["message"])
                if status["saved"]:
                    st.info("以上文件已保存，旧索引未改变；处理问题后可点击“重新读取资料并更新知识库”。")
