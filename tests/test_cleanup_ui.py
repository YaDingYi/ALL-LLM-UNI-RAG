'''隔离验证知识库清理提示与旧选择失效后的页面行为。'''

from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import chroma_index
import dotenv
import history
import upload_ui


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
STATUS_SCRIPT = "from upload_ui import render_update_status\nrender_update_status()"


def status(cleanup):
    '''构造模拟的启动更新结果，不触及实际集合与更新历史。'''
    return {
        "ok": True,
        "result": {"name": "kb-new", "unchanged": False, "warnings": [], "cleanup": cleanup},
        "error": None,
        "log_path": None,
        "log_error": None,
    }


class CleanupStatusUITests(unittest.TestCase):
    def render(self, outcome):
        with patch.object(upload_ui, "_startup_update", return_value=outcome):
            page = AppTest.from_string(STATUS_SCRIPT, default_timeout=10).run()
        self.assertFalse(page.exception, [str(item.value) for item in page.exception])
        return page

    def test_success_shows_deleted_collection_count(self):
        page = self.render(status({"kept": "kb-new", "deleted": ["kb-a", "kb-b"],
                                   "failed": [], "skipped_unready": [], "aborted": False}))
        self.assertTrue(any("已清理 2 个旧知识库" in item.value for item in page.caption))
        self.assertFalse(page.warning)

    def test_failure_reports_partial_cleanup_and_preserves_log_warning(self):
        result = status({"kept": "kb-new", "deleted": ["kb-a"],
                         "failed": [{"name": "kb-b", "error_type": "PermissionError"}],
                         "skipped_unready": [], "aborted": False})
        result["log_error"] = "测试用记录保存失败"
        page = self.render(result)
        self.assertTrue(any("部分旧知识库清理失败" in item.value for item in page.warning))
        self.assertTrue(any("测试用记录保存失败" in item.value for item in page.warning))
        self.assertFalse(any("已清理 1 个旧知识库" in item.value for item in page.caption))

    def test_unfinished_collection_is_reported_as_not_deleted(self):
        page = self.render(status({"kept": "kb-new", "deleted": [], "failed": [],
                                   "skipped_unready": ["kb-in-progress"], "aborted": False}))
        self.assertTrue(any("未完成的知识库集合" in item.value for item in page.info))


class DeletedSelectionUITests(unittest.TestCase):
    def test_deleted_selection_leaves_no_index_selector(self):
        # 模拟旧会话状态与集合变化，页面不能重新出现知识库选择框。
        with patch.object(dotenv, "dotenv_values", return_value={"DEEPSEEK_API_KEY": "test-key"}), \
                patch.object(history, "list_history_files", return_value=[]), \
                patch.object(upload_ui, "render_update_status"), \
                patch.object(chroma_index, "list_indexes", return_value=["kb-old", "kb-new"]) as indexes:
            page = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
            self.assertFalse(page.exception, [str(item.value) for item in page.exception])
            self.assertFalse(any(item.label == "选择问答使用的知识库" for item in page.selectbox))
            page.session_state["rag_index_choice"] = "kb-old"
            indexes.return_value = ["kb-new"]
            page.run()
            self.assertFalse(page.exception, [str(item.value) for item in page.exception])
            self.assertFalse(any(item.label == "选择问答使用的知识库" for item in page.selectbox))


if __name__ == "__main__":
    unittest.main()
