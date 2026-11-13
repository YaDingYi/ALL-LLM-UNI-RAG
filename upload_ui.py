'''在网页显示启动自动更新的结果；资料由用户放入本地文件夹。'''

import streamlit as st

from auto_update import run_update


@st.cache_resource(show_spinner="服务启动后正在自动更新知识库，请稍候…")
def _startup_update():
    '''每个服务进程首次打开网页时执行一次；后续页面重运行共享结果。'''
    return run_update(trigger="启动自动更新")


def render_update_status():
    '''显示启动更新及Word记录状态，不提供网页上传入口。'''
    # 同一进程只更新一次；重新运行页面时复用已缓存的结果。
    startup = _startup_update()
    if startup["ok"]:
        result = startup["result"]
        if result.get("unchanged"):
            st.caption("启动自动更新完成：资料没有有效变化，继续使用已有知识库。")
        else:
            st.caption(f"启动自动更新完成：已生成知识库 {result['name']}。")
        cleanup = result.get("cleanup")
        if cleanup:
            if cleanup.get("failed") or cleanup.get("aborted"):
                st.warning("最新知识库已可用，但部分旧知识库清理失败；请查看更新记录及目录权限。下次启动时会再次尝试。")
            else:
                st.caption(f"已清理 {len(cleanup.get('deleted', []))} 个旧知识库。")
            if cleanup.get("skipped_unready"):
                st.info("存在未完成的知识库集合，可能仍在构建，本次没有删除。")
        # 文件读取提示仍显示，便于用户发现不能解析的资料。
        for warning in result.get("warnings", []):
            st.warning(f"{warning['source']}：{warning['message']}")
    else:
        st.warning(f"启动自动更新未完成：{startup['error']}")
    # 索引成功但记录写入失败时，单独提示，不误报知识库失败。
    if startup.get("log_error"):
        st.warning(startup["log_error"])
