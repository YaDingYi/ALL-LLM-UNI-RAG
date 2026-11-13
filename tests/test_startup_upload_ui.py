'''隔离测试自动更新状态界面，不读取或修改用户资料。'''

import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import upload_ui


INDEX_RESULT = {
    "name": "kb-start", "file_count": 1, "chunk_count": 2, "warnings": [],
    "unchanged": False, "legacy": False, "new_file_count": 1,
    "modified_file_count": 0, "deleted_file_count": 0,
    "reused_file_count": 0, "reused_chunk_count": 0, "new_chunk_count": 2,
}


def outcome(ok=True, result=None, error=None, log_error=None):
    return {"ok": ok, "result": INDEX_RESULT.copy() if result is None and ok else result,
            "error": error, "log_path": None, "log_error": log_error}


SCRIPT = '''import streamlit as st
from upload_ui import render_update_status
render_update_status()
st.selectbox("其他选择", ["kb-start", "kb-upload", "kb-older"], key="other_choice")
'''


class StartupUpdateUITests(unittest.TestCase):
    def setUp(self):
        # 使用显式清理模拟不同服务进程，防止测试之间共享启动缓存。
        upload_ui._startup_update.clear()
        self.addCleanup(upload_ui._startup_update.clear)

    def run_page(self):
        page = AppTest.from_string(SCRIPT, default_timeout=10).run()
        self.assertFalse(page.exception, [str(item.value) for item in page.exception])
        return page

    def test_once_per_service_across_reruns_sessions_and_restart(self):
        with patch.object(upload_ui, "run_update", return_value=outcome()) as run_update:
            page = self.run_page()
            self.assertEqual(len(page.button), 0)
            self.assertEqual(len(page.get("file_uploader")), 0)
            page.run()
            another_session = self.run_page()
            self.assertFalse(another_session.exception)
            self.assertEqual(run_update.call_count, 1)
            self.assertEqual(run_update.call_args.kwargs["trigger"], "启动自动更新")
            # 清理全局缓存模拟服务重启，下一连接才运行下一次更新。
            upload_ui._startup_update.clear()
            self.run_page()
            self.assertEqual(run_update.call_count, 2)

    def test_user_selection_is_not_overwritten_on_rerun(self):
        with patch.object(upload_ui, "run_update", return_value=outcome()):
            page = self.run_page()
            self.assertEqual(page.selectbox[0].value, "kb-start")
            page.selectbox[0].set_value("kb-older").run()
            self.assertEqual(page.selectbox[0].value, "kb-older")
            page.run()
            self.assertEqual(page.selectbox[0].value, "kb-older")

    def test_failure_is_visible_and_cached_until_restart(self):
        failed = outcome(False, error="测试专用启动失败")
        with patch.object(upload_ui, "run_update", return_value=failed) as run_update:
            page = self.run_page()
            notices = [str(item.value) for item in [*page.error, *page.warning, *page.info]]
            self.assertTrue(any("测试专用启动失败" in notice for notice in notices))
            page.run()
            self.run_page()
            self.assertEqual(run_update.call_count, 1)
            self.assertEqual(len(page.button), 0)
            self.assertEqual(len(page.get("file_uploader")), 0)

    def test_log_failure_does_not_misreport_update_or_repeat_it(self):
        status = outcome(log_error="测试专用日志失败")
        with patch.object(upload_ui, "run_update", return_value=status) as run_update:
            page = self.run_page()
            notices = [str(item.value) for item in [*page.error, *page.warning, *page.info]]
            self.assertTrue(any("测试专用日志失败" in notice for notice in notices))
            page.run()
            self.assertEqual(run_update.call_count, 1)

    def test_startup_success_does_not_show_upload_controls(self):
        with patch.object(upload_ui, "run_update", return_value=outcome()) as update:
            page = self.run_page()
            self.assertEqual(len(page.get("file_uploader")), 0)
            self.assertEqual(len(page.button), 0)
            page.run()
            self.assertEqual(update.call_count, 1)


if __name__ == "__main__":
    unittest.main()
