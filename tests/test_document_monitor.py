'''使用临时资料目录验证检查结果、失败保护和 Streamlit 重跑行为。'''

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from document_monitor import check_document_changes


class DocumentMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "资料"
        self.root.mkdir()
        self.snapshot = Path(self.temp.name) / ".local" / "document_inventory.json"

    def check(self):
        return check_document_changes(self.root, self.snapshot)

    def test_first_check_and_unchanged_restart(self):
        (self.root / "旧资料.txt").write_text("现有资料", encoding="utf-8")
        first = self.check()
        self.assertTrue(first["first_check"])
        self.assertEqual(first["file_count"], 1)
        self.assertEqual(first["added"], [])
        saved = json.loads(self.snapshot.read_text(encoding="utf-8"))
        self.assertEqual(set(saved["files"]), {"旧资料.txt"})
        second = self.check()
        self.assertFalse(second["first_check"])
        for key in ("added", "modified", "deleted"):
            self.assertEqual(second[key], [])

    def test_nested_additions_and_ignored_files(self):
        self.check()
        folder = self.root / "RRAM"
        folder.mkdir()
        # 文件内容故意不可解析：扫描不应尝试读取正文。
        names = ["论文.PDF", "笔记.md", "补充.markdown", "记录.txt", "说明.docx"]
        for name in names + ["~$说明.docx", ".~说明.docx", "旧版.doc", "图片.png"]:
            (folder / name).write_bytes(b"not a document")
        result = self.check()
        self.assertEqual(result["added"], sorted(f"RRAM/{name}" for name in names))
        self.assertEqual(result["file_count"], 5)

    def test_modification_deletion_and_rename(self):
        for name in ("修改.txt", "删除.md", "旧名.txt"):
            (self.root / name).write_text("旧内容", encoding="utf-8")
        self.check()
        (self.root / "修改.txt").write_text("修改后的内容", encoding="utf-8")
        (self.root / "删除.md").unlink()
        (self.root / "旧名.txt").rename(self.root / "新名.txt")
        result = self.check()
        self.assertEqual(result["modified"], ["修改.txt"])
        self.assertEqual(result["added"], ["新名.txt"])
        self.assertEqual(result["deleted"], sorted(["删除.md", "旧名.txt"]))

    def test_mtime_detects_same_size_edit(self):
        document = self.root / "笔记.txt"
        document.write_text("old", encoding="utf-8")
        self.check()
        before = document.stat()
        document.write_text("new", encoding="utf-8")
        os.utime(document, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        self.assertEqual(self.check()["modified"], ["笔记.txt"])

    def test_corrupt_snapshot_is_not_overwritten(self):
        self.snapshot.parent.mkdir()
        for data in (b"invalid json", b'{"version":1,"files":[]}',
                     b'{"version":1,"files":{"a.pdf":{"size":false,"mtime_ns":1}}}'):
            with self.subTest(data=data):
                self.snapshot.write_bytes(data)
                with self.assertRaises(ValueError):
                    self.check()
                self.assertEqual(self.snapshot.read_bytes(), data)

    def test_missing_directory_preserves_snapshot(self):
        self.check()
        original = self.snapshot.read_bytes()
        self.root.rmdir()
        with self.assertRaises(FileNotFoundError):
            self.check()
        self.assertEqual(self.snapshot.read_bytes(), original)

    def test_inaccessible_subdirectory_preserves_snapshot(self):
        self.check()
        original = self.snapshot.read_bytes()

        def broken_walk(root, onerror):
            onerror(PermissionError("cannot scan child directory"))

        with patch("document_monitor.os.walk", side_effect=broken_walk):
            with self.assertRaises(PermissionError):
                self.check()
        self.assertEqual(self.snapshot.read_bytes(), original)

    def test_failed_save_preserves_snapshot_and_cleans_tempfile(self):
        self.check()
        original = self.snapshot.read_bytes()
        (self.root / "新增.txt").write_text("new", encoding="utf-8")
        with patch("document_monitor.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.check()
        self.assertEqual(self.snapshot.read_bytes(), original)
        self.assertEqual(list(self.snapshot.parent.glob("*.tmp")), [])
        self.assertEqual(self.check()["added"], ["新增.txt"])

    def test_ui_checks_once_then_explicit_recheck(self):
        from streamlit.testing.v1 import AppTest
        from document_monitor_ui import _startup_document_check

        _startup_document_check.clear()
        self.addCleanup(_startup_document_check.clear)
        script = "from document_monitor_ui import render_document_check\nrender_document_check()"
        with patch("document_monitor_ui.check_document_changes", side_effect=self.check) as check:
            page = AppTest.from_string(script).run()
            self.assertEqual(len(page.exception), 0)
            self.assertIn("首次检查", page.info[0].value)
            (self.root / "新增.txt").write_text("new", encoding="utf-8")
            page.run()
            another_session = AppTest.from_string(script).run()
            self.assertEqual(len(another_session.exception), 0)
            self.assertEqual(check.call_count, 1)
            page.button(key="recheck_documents").click().run()
            self.assertEqual(check.call_count, 2)
            self.assertIn("新增 1 个", page.warning[0].value)
            self.assertIn("新增.txt", page.text[0].value)
            # 清空进程级缓存模拟服务重启，使用磁盘上的基准继续比较。
            (self.root / "再次新增.txt").write_text("new", encoding="utf-8")
            _startup_document_check.clear()
            restarted = AppTest.from_string(script).run()
            self.assertEqual(check.call_count, 3)
            self.assertIn("再次新增.txt", restarted.text[0].value)
            self.assertNotIn("\n新增.txt", restarted.text[0].value)

    def test_ui_scan_failure_keeps_other_controls_available(self):
        from streamlit.testing.v1 import AppTest
        from document_monitor_ui import _startup_document_check

        _startup_document_check.clear()
        self.addCleanup(_startup_document_check.clear)
        script = ("import streamlit as st\nfrom document_monitor_ui import render_document_check\n"
                  "render_document_check()\nst.button('其他功能')")
        with patch("document_monitor_ui.check_document_changes", side_effect=OSError("unreadable")):
            page = AppTest.from_string(script).run()
        self.assertEqual(len(page.exception), 0)
        self.assertIn("资料检查未完成", page.warning[0].value)
        self.assertEqual(page.button[-1].label, "其他功能")


if __name__ == "__main__":
    unittest.main()
