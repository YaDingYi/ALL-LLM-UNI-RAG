'''在临时目录测试自动更新记录，不修改用户资料和知识库。'''

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from docx import Document

import auto_update
import chroma_index
import document_reader


def document_text(path):
    '''按段落读取生成的Word文档，便于检查用户能看到的内容。'''
    return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)


class AutoUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.history_dir = Path(self.temporary.name) / "更新历史"
        self.directory_patch = patch.object(auto_update, "UPDATE_HISTORY_DIR", self.history_dir)
        self.directory_patch.start()
        self.addCleanup(self.directory_patch.stop)
        self.update_patch = patch.object(auto_update, "update_knowledge_index")
        self.update = self.update_patch.start()
        self.addCleanup(self.update_patch.stop)
        self.result = {
            "name": "kb-test", "file_count": 3, "chunk_count": 8,
            "new_file_count": 1, "modified_file_count": 1, "deleted_file_count": 1,
            "reused_chunk_count": 6, "new_chunk_count": 2,
            "added_sources": ["上传/新文档.txt"], "modified_sources": ["修改.md"],
            "deleted_sources": ["移除.pdf"], "warnings": [],
            "unchanged": False, "legacy": False, "retained_missing_count": 0,
        }
        self.update.return_value = self.result

    def test_success_records_counters_and_file_names(self):
        progress = Mock()
        result = auto_update.run_update(trigger="上传更新", base_name="kb-old",
                                        required_sources=("上传/新文档.txt",),
                                        preserve_missing=True, on_progress=progress)
        self.update.assert_called_once_with(required_sources=("上传/新文档.txt",), on_progress=progress,
                                            base_name="kb-old", preserve_missing=True)
        self.assertTrue(result["ok"])
        self.assertIsNone(result["log_error"])
        self.assertEqual(result["result"], self.result)
        path = Path(result["log_path"])
        self.assertEqual(path.parent, self.history_dir.resolve())
        self.assertEqual(path.suffix, ".docx")
        text = document_text(path)
        for value in ("上传更新", "更新状态：成功", "kb-old", "kb-test", "上传/新文档.txt", "修改.md", "移除.pdf"):
            self.assertIn(value, text)
        self.assertIn("新生成向量：2 个片段", text)

    def test_identical_second_runs_do_not_overwrite_records(self):
        first = auto_update.run_update()
        second = auto_update.run_update()
        self.assertNotEqual(first["log_path"], second["log_path"])
        self.assertEqual(len(list(self.history_dir.glob("*.docx"))), 2)

    def test_update_failure_is_logged_as_failure(self):
        self.update.side_effect = auto_update.UploadError("资料无法读取：bad.pdf")
        result = auto_update.run_update()
        self.assertFalse(result["ok"])
        self.assertIsNone(result["result"])
        self.assertIn("资料无法读取", result["error"])
        text = document_text(result["log_path"])
        self.assertIn("更新状态：失败", text)
        self.assertIn("bad.pdf", text)

    def test_unexpected_exception_does_not_leak_raw_details(self):
        self.update.side_effect = RuntimeError("test-secret-do-not-output")
        result = auto_update.run_update()
        self.assertFalse(result["ok"])
        self.assertIn("RuntimeError", result["error"])
        self.assertNotIn("test-secret-do-not-output", result["error"])
        self.assertNotIn("test-secret-do-not-output", document_text(result["log_path"]))

    def test_record_failure_keeps_successful_index_result(self):
        with patch.object(auto_update, "write_update_record", side_effect=OSError("private path")):
            result = auto_update.run_update()
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["name"], "kb-test")
        self.assertIsNone(result["log_path"])
        self.assertIn("更新记录未能保存", result["log_error"])
        self.assertNotIn("private path", result["log_error"])

    def test_partial_cleanup_is_recorded_without_hiding_success(self):
        # 清理旧集合失败时，知识库更新仍成功；Word记录必须告知未清理的集合。
        self.result["cleanup"] = {
            "kept": "kb-test", "deleted": ["kb-old-a"],
            "failed": [{"name": "kb-old-b", "error_type": "PermissionError"}],
            "skipped_unready": [], "aborted": False,
        }
        result = auto_update.run_update()
        self.assertTrue(result["ok"])
        text = document_text(result["log_path"])
        for value in ("更新状态：成功", "kb-old-a", "kb-old-b", "PermissionError", "未能清理"):
            self.assertIn(value, text)

    def test_upload_save_failure_writes_record_without_starting_index_update(self):
        result = auto_update.record_upload_failure(auto_update.UploadError("文件名不支持"), base_name="kb-old")
        self.update.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertEqual(result["trigger"], "上传更新")
        self.assertEqual(result["base_name"], "kb-old")
        self.assertIn("文件名不支持", document_text(result["log_path"]))

    def test_unexpected_upload_error_and_log_failure_are_safe(self):
        with patch.object(auto_update, "write_update_record", side_effect=OSError("secret path")):
            result = auto_update.record_upload_failure(RuntimeError("test-secret"))
        self.update.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertIn("RuntimeError", result["error"])
        self.assertNotIn("test-secret", result["error"])
        self.assertNotIn("secret path", result["log_error"])


class AutoUpdateIntegrationTests(unittest.TestCase):
    '''临时真实Chroma、合成文档与假向量模型的端到端检查。'''

    def test_start_update_changed_documents_and_unchanged_restart(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "资料"
            docs.mkdir()
            (docs / "不变.txt").write_text("这是一份保留的测试资料。", encoding="utf-8")
            modified = docs / "修改.txt"
            modified.write_text("这是旧版本。", encoding="utf-8")
            embedding = Mock(side_effect=lambda texts: [[1.0] + [0.0] * 511 for _ in texts])
            with patch.object(document_reader, "DOCUMENTS_DIR", docs), \
                    patch.object(chroma_index, "CHROMA_DIR", root / "chroma"), \
                    patch.object(chroma_index, "embed_texts", embedding), \
                    patch.object(auto_update, "UPDATE_HISTORY_DIR", root / "更新历史"):
                # 首次启动建立基线并记录两个新增文件。
                first = auto_update.run_update()
                self.assertTrue(first["ok"], first["error"])
                self.assertIsNone(first["log_error"])
                first_name = first["result"]["name"]
                first_doc = document_text(first["log_path"])
                self.assertIn("更新状态：成功", first_doc)
                self.assertIn("不变.txt", first_doc)
                self.assertIn("修改.txt", first_doc)
                self.assertEqual(first["result"]["new_chunk_count"], 2)

                # 第二次启动只为新增和修改的两份资料生成向量。
                modified.write_text("修改后是一段长度不同的资料。", encoding="utf-8")
                (docs / "新增.txt").write_text("这是一份新增资料。", encoding="utf-8")
                embedding.reset_mock()
                second = auto_update.run_update()
                self.assertTrue(second["ok"], second["error"])
                self.assertIsNone(second["log_error"])
                second_name = second["result"]["name"]
                self.assertNotEqual(first_name, second_name)
                self.assertEqual(second["result"]["base_name"], first_name)
                self.assertEqual(second["result"]["reused_chunk_count"], 1)
                self.assertEqual(second["result"]["new_chunk_count"], 2)
                embedding.assert_called_once()
                self.assertEqual(set(embedding.call_args.args[0]), {"修改后是一段长度不同的资料。", "这是一份新增资料。"})
                # 新集合发布后只保留最新集合，Word记录仍保留旧集合名称。
                self.assertNotIn(first_name, chroma_index.list_indexes())
                self.assertEqual(chroma_index.list_indexes(), [second_name])
                self.assertEqual(chroma_index._get_ready_collection(second_name)[1], 3)
                second_doc = document_text(second["log_path"])
                for expected in ("更新状态：成功", "新增.txt", "修改.txt", "复用旧向量：1 个片段", first_name, second_name):
                    self.assertIn(expected, second_doc)

                # 无变化的下一次启动复用新集合，不再次生成向量。
                embedding.reset_mock()
                third = auto_update.run_update()
                self.assertTrue(third["ok"], third["error"])
                self.assertIsNone(third["log_error"])
                self.assertEqual(third["result"]["name"], second_name)
                self.assertTrue(third["result"]["unchanged"])
                embedding.assert_not_called()
                third_doc = document_text(third["log_path"])
                self.assertIn("更新状态：成功", third_doc)
                self.assertIn("资料没有有效变化", third_doc)
                self.assertEqual(len(list((root / "更新历史").glob("*.docx"))), 3)
                self.assertNotEqual(first["log_path"], second["log_path"])
                self.assertNotEqual(second["log_path"], third["log_path"])


if __name__ == "__main__":
    unittest.main()
