'''使用模拟 HTTP 响应验证联网搜索，不联网、不消耗用户账户。'''

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import httpx

from web_search import SEARCH_URL, WebSearchError, search_web


class WebSearchTests(unittest.TestCase):
    def response(self, payload=None, status=200):
        response = Mock(status_code=status)
        response.json.return_value = payload
        return response

    @patch("web_search.httpx.post")
    def test_request_and_result_normalization(self, post):
        post.return_value = self.response({"results": [{
            "title": "  页面标题\n示例 ", "url": "https://example.com/news",
            "summary": "  第一行\n第二行 ", "date": "2026-11-12T00:00:00Z",
        }]})
        result = search_web("  测试问题  ", " test-key ", count=2)
        args, kwargs = post.call_args
        self.assertEqual(args, (SEARCH_URL,))
        self.assertEqual(kwargs["json"], {"query": "测试问题", "count": 2, "need_content": False})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        retrieved_at = result[0].pop("retrieved_at")
        self.assertEqual(datetime.fromisoformat(retrieved_at).utcoffset().total_seconds(), 8 * 3600)
        self.assertEqual(result, [{"type": "web", "source": "页面标题 示例", "url": "https://example.com/news", "text": "第一行 第二行", "page": None, "date": "2026-11-12T00:00:00Z"}])
        self.assertNotIn("test-key", repr(result))

    @patch("web_search.httpx.post")
    def test_filters_invalid_urls_and_deduplicates(self, post):
        post.return_value = self.response({"results": [
            {"url": "javascript:alert(1)", "summary": "不允许"},
            {"url": "https://user:pass@example.com", "summary": "不允许"},
            {"url": "https://example.com", "snippet": "有效摘要"},
            {"url": "https://example.com", "summary": "重复"},
            {"url": "https://example.org", "summary": "第二篇"},
        ]})
        result = search_web("问题", "key", count=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["url"], "https://example.com")
        self.assertEqual(result[0]["text"], "有效摘要")

    @patch("web_search.httpx.post")
    def test_empty_search_is_distinct_from_bad_results(self, post):
        post.return_value = self.response({"results": []})
        self.assertEqual(search_web("问题", "key"), [])
        post.return_value = self.response({"results": [{"url": "javascript:bad", "summary": "无效"}]})
        with self.assertRaisesRegex(WebSearchError, "缺少有效链接或摘要"):
            search_web("问题", "key")

    @patch("web_search.httpx.post")
    def test_bad_query_and_count_never_issue_requests(self, post):
        for question in ("", "   ", "问" * 129):
            with self.subTest(question=question), self.assertRaises(WebSearchError):
                search_web(question, "key")
        for count in (0, 51, True, 1.5):
            with self.subTest(count=count), self.assertRaises(WebSearchError):
                search_web("问题", "key", count)
        with self.assertRaises(WebSearchError):
            search_web("问题", "")
        post.assert_not_called()

    @patch("web_search.httpx.post")
    def test_status_errors_are_safe(self, post):
        for status, message in ((401, "认证"), (402, "余额"), (429, "频繁"), (500, "HTTP 500")):
            post.return_value = self.response(status=status)
            with self.subTest(status=status), self.assertRaisesRegex(WebSearchError, message) as raised:
                search_web("问题", "secret-key")
            self.assertNotIn("secret-key", str(raised.exception))
            post.return_value.json.assert_not_called()

    @patch("web_search.httpx.post")
    def test_timeout_and_connection_errors(self, post):
        for error in (httpx.TimeoutException("secret-key"), httpx.ConnectError("secret-key")):
            post.side_effect = error
            with self.subTest(error=error), self.assertRaises(WebSearchError) as raised:
                search_web("问题", "secret-key")
            self.assertNotIn("secret-key", str(raised.exception))
            self.assertIsNone(raised.exception.__cause__)

    @patch("web_search.httpx.post")
    def test_invalid_response_is_not_treated_as_empty(self, post):
        for payload in ({}, {"results": None}, [], {"results": {}}):
            post.return_value = self.response(payload)
            with self.subTest(payload=payload), self.assertRaisesRegex(WebSearchError, "格式"):
                search_web("问题", "key")
        post.return_value = self.response()
        post.return_value.json.side_effect = ValueError("bad JSON")
        with self.assertRaisesRegex(WebSearchError, "无法解析"):
            search_web("问题", "key")


if __name__ == "__main__":
    unittest.main()
