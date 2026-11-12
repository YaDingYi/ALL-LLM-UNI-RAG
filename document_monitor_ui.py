'''在服务首次打开网页时检查资料，并保留结果供后续页面交互显示。'''

import streamlit as st

from document_monitor import check_document_changes


@st.cache_resource(show_spinner=False)
def _startup_document_check():
    '''同一服务进程共用一次检查；重启服务会清空缓存并重新扫描。'''
    return check_document_changes()


def render_document_check():
    '''显示启动检查结果，也允许在服务运行期间手动重新检查。'''
    st.markdown("**本地资料检查**")
    if st.button("重新检查资料文件", key="recheck_documents"):
        # 用户运行期间手动添加文件后，可主动清除缓存并立即检查。
        _startup_document_check.clear()

    try:
        result = _startup_document_check()
    except (OSError, ValueError):
        st.warning("资料检查未完成，请检查“资料”目录权限及 .local/document_inventory.json 的格式和权限。原检查记录已保留，其他功能仍可使用。")
        return

    if result["first_check"]:
        st.info(f"首次检查：已建立 {result['file_count']} 个资料文件的检查基准。下次启动或点击重新检查时，将显示相较本次的变化。")
    elif any(result[key] for key in ("added", "modified", "deleted")):
        st.warning(f"相较上次检查：新增 {len(result['added'])} 个、修改 {len(result['modified'])} 个、删除 {len(result['deleted'])} 个资料文件。")
        with st.expander("查看变化的文件", expanded=True):
            for key, label in (("added", "新增"), ("modified", "修改"), ("deleted", "删除")):
                if result[key]:
                    st.text(f"{label}：\n" + "\n".join(result[key]))
        st.info("如需将当前资料用于问答，请点击下方“重新读取资料并更新知识库”。")
    else:
        st.success(f"资料检查完成：相较上次检查没有文件变化，共 {result['file_count']} 个支持格式文件。")

    st.caption(f"检查时间：{result['checked_at']}（北京时间）。递归检查资料及子文件夹中的 TXT、Markdown、PDF、DOCX；结果与上次检查比较，不代表所选知识库的入库状态。")
