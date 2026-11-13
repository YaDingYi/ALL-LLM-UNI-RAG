'''隔离验证联合检索网页；不连接模型、搜索服务或用户数据库。'''

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import chroma_index
import dotenv
import history
import openai
import rag_demo
import upload_ui
import web_search


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
LOCAL = {"id": "chunk-1", "source": "guide.txt", "page": None,
         "number": 1, "text": "本地资料", "score": 0.8}
WEB = {"type": "web", "source": "网络页面", "url": "https://example.org/source",
       "text": "网络摘要", "page": None, "date": None,
       "retrieved_at": "2026-11-12T12:00:00+08:00"}
WEB2 = {**WEB, "source": "第二页面", "url": "https://example.org/second", "text": "第二条摘要"}


class HybridUITests(unittest.TestCase):
    '''检验联合模式调用、展示、历史和失败处理。'''

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.history_dir = Path(self.tmp.name) / "Histor"
        self.requests = []
        self.real_update_status = upload_ui.render_update_status
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="基于两种来源的回答 [1][2]"), finish_reason="stop")],
            model="fake-model", usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )
        requests = self.requests

        class FakeClient:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **kwargs):
                requests.append(kwargs)
                return response

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        replacements = [
            patch.object(history, "HISTORY_DIR", self.history_dir),
            patch.object(dotenv, "dotenv_values", return_value={"DEEPSEEK_API_KEY": "test-only-key"}),
            patch.object(chroma_index, "list_indexes", return_value=["kb-test"]),
            patch.object(rag_demo, "search_index", return_value=([LOCAL.copy()], 1)),
            patch.object(upload_ui, "render_update_status"),
            patch.object(openai, "OpenAI", FakeClient),
            patch.object(web_search, "search_web", return_value=[WEB.copy(), WEB2.copy()]),
        ]
        self.web_mock = None
        self.index_mock = None
        self.local_mock = None
        for replacement in replacements:
            result = replacement.start()
            self.addCleanup(replacement.stop)
            if replacement.attribute == "search_web":
                self.web_mock = result
            elif replacement.attribute == "list_indexes":
                self.index_mock = result
            elif replacement.attribute == "search_index":
                self.local_mock = result

    def run_app(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        return app

    def test_hybrid_full_flow_and_history_reload(self):
        app = self.run_app()
        self.assertEqual(app.radio[0].value, "本地+互联网联合检索")
        self.assertEqual(app.chat_input[0].proto.max_chars, 128)
        app.chat_input[0].set_value("示范问题").run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.web_mock.assert_called_once_with("示范问题", "test-only-key", count=3)
        self.assertEqual(len(self.requests), 1)
        content = self.requests[0]["messages"][-1]["content"]
        self.assertIn("[1] 本地资料：guide.txt", content)
        self.assertIn("[2] 互联网搜索摘要：网络页面", content)
        self.assertIn("[3] 互联网搜索摘要：第二页面", content)
        self.assertIn("https://example.org/source", content)
        files = list(self.history_dir.glob("*.json"))
        self.assertEqual(len(files), 1)
        messages = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(messages[-1]["sources"][0]["source"], "guide.txt")
        self.assertEqual(messages[-1]["sources"][1]["url"], "https://example.org/source")
        self.assertEqual(messages[-1]["sources"][2]["url"], "https://example.org/second")
        self.assertNotIn("test-only-key", json.dumps(messages, ensure_ascii=False))
        texts = [item.value for item in app.text]
        self.assertTrue(any("[1] 来源：guide.txt" in text for text in texts))
        self.assertTrue(any("[2] 互联网搜索摘要：网络页面" in text for text in texts))
        self.assertEqual(len(app.get("link_button")), 2)
        # 新会话加载同一历史，仍可显示原有编号和网络链接。
        fresh = self.run_app()
        fresh.selectbox[0].set_value(files[0].name).run()
        self.assertFalse(fresh.exception, [str(item.value) for item in fresh.exception])
        self.assertTrue(any("[2] 互联网搜索摘要：网络页面" in item.value for item in fresh.text))
        self.assertEqual(len(fresh.get("link_button")), 2)
        self.web_mock.assert_called_once()
        self.assertEqual(len(self.requests), 1)

    def test_search_failure_does_not_call_model(self):
        self.web_mock.side_effect = web_search.WebSearchError("联网搜索暂时不可用")
        app = self.run_app()
        app.chat_input[0].set_value("示范问题").run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.assertTrue(any("联网搜索暂时不可用" in item.value for item in app.error))
        self.assertEqual(self.requests, [])
        messages = json.loads(next(self.history_dir.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual([item["role"] for item in messages], ["user"])

    def test_local_mode_does_not_search_web(self):
        # 即使遗留状态指向旧集合，只使用最新集合且不发起互联网搜索。
        self.index_mock.return_value = ["kb-new", "kb-old"]
        app = self.run_app()
        app.session_state["rag_index_choice"] = "kb-old"
        app.radio[0].set_value("知识库问答").run()
        self.assertEqual(app.chat_input[0].proto.max_chars, 200)
        app.chat_input[0].set_value("本地问题").run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.web_mock.assert_not_called()
        self.assertEqual(len(self.requests), 1)
        self.local_mock.assert_called_once_with("本地问题", "kb-new", 3)
        self.assertIn("[1] 本地资料：guide.txt", self.requests[0]["messages"][-1]["content"])

    def test_ordinary_chat_does_not_search_local_or_web(self):
        app = self.run_app()
        app.radio[0].set_value("普通对话").run()
        app.chat_input[0].set_value("普通问题").run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.local_mock.assert_not_called()
        self.web_mock.assert_not_called()
        self.assertEqual(len(self.requests), 1)

    def test_empty_index_disables_hybrid_input(self):
        self.index_mock.return_value = []
        app = self.run_app()
        self.assertEqual(app.radio[0].value, "普通对话")
        app.radio[0].set_value("本地+互联网联合检索").run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.assertTrue(app.chat_input[0].proto.disabled)
        self.assertTrue(any("资料" in item.value for item in app.info))
        self.web_mock.assert_not_called()

    def test_startup_selects_updated_index_and_preserves_hybrid_chat(self):
        # 自动更新使用桩，不写入真实资料、Word记录或Chroma。
        self.index_mock.return_value = ["kb-new", "kb-test"]
        startup = {"ok": True, "result": {"name": "kb-new", "file_count": 1, "chunk_count": 2, "warnings": []},
                   "error": None, "log_path": None, "log_error": None}

        with patch.object(upload_ui, "render_update_status", self.real_update_status), \
                patch.object(upload_ui, "_startup_update", return_value=startup):
            app = self.run_app()
            self.assertEqual(app.radio[0].value, "本地+互联网联合检索")
            self.assertEqual(len(app.get("file_uploader")), 0)
            self.assertEqual(len(app.button), 0)
            self.assertFalse(any(item.label == "选择问答使用的知识库" for item in app.selectbox))
            # 旧会话遗留选择不会影响新集合的使用。
            app.session_state["rag_index_choice"] = "kb-test"
            app.chat_input[0].set_value("更新后的问题").run()
            self.assertFalse(app.exception, [str(item.value) for item in app.exception])
            self.local_mock.assert_called_once_with("更新后的问题", "kb-new", 3)
            self.web_mock.assert_called_once_with("更新后的问题", "test-only-key", count=3)


if __name__ == "__main__":
    unittest.main()
