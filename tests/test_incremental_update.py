'''用临时资料和模拟索引验证增量流程，不读取用户文件。'''

import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import document_reader
import knowledge_upload
import chroma_index
from document_monitor import scan_documents


class IncrementalUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old = self.root / "old.txt"
        self.old.write_text("旧资料的内容", encoding="utf-8")
        self.manifest = scan_documents(self.root)
        self.patch_root = patch.object(document_reader, "DOCUMENTS_DIR", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        # 将跨进程锁的目录也指向临时目录，绝不接触用户的真实数据库。
        directory_patch = patch.object(chroma_index, "CHROMA_DIR", self.root / "chroma")
        directory_patch.start()
        self.addCleanup(directory_patch.stop)
        self.mocks = {}
        for name, value in (
            ("list_indexes", ["kb-old"]),
            ("read_index_manifest", self.manifest),
            ("list_index_sources", {"old.txt"}),
        ):
            mocked = patch.object(knowledge_upload, name, return_value=value)
            self.mocks[name] = mocked.start()
            self.addCleanup(mocked.stop)
        self.create = patch.object(knowledge_upload, "create_incremental_index")
        self.create_mock = self.create.start()
        self.addCleanup(self.create.stop)
        self.create_mock.side_effect = self.finish
        # 本类只测试调度，使用清理桩；真实Chroma清理由下方集成测试覆盖。
        self.cleanup_patch = patch.object(knowledge_upload, "cleanup_old_indexes")
        self.cleanup = self.cleanup_patch.start()
        self.addCleanup(self.cleanup_patch.stop)
        self.cleanup.side_effect = lambda name: {"kept": name, "deleted": [], "failed": [],
                                                 "skipped_unready": [], "aborted": False}

    def finish(self, base, reused, chunks, signatures, on_progress=None, validate_before_publish=None, missing_sources=()):
        validate_before_publish()
        return {"name": "kb-new", "file_count": len(signatures), "chunk_count": len(chunks) + len(reused),
                "reused_chunk_count": len(reused), "new_chunk_count": len(chunks)}

    def test_added_file_is_only_file_parsed_and_vectorized(self):
        (self.root / "new.txt").write_text("新资料", encoding="utf-8")
        with patch.object(document_reader, "read_documents", wraps=document_reader.read_documents) as read:
            result = knowledge_upload.update_knowledge_index(base_name="kb-old")
        read.assert_called_once_with({"new.txt"})
        args = self.create_mock.call_args.args
        self.assertEqual(args[1], {"old.txt"})
        self.assertEqual({chunk["source"] for chunk in args[2]}, {"new.txt"})
        self.assertEqual(result["reused_file_count"], 1)
        self.assertEqual(result["new_file_count"], 1)
        self.cleanup.assert_called_once_with("kb-new")

    def test_no_change_reuses_existing_collection(self):
        with patch.object(knowledge_upload, "_get_ready_collection", return_value=(Mock(), 1)), \
                patch.object(document_reader, "read_documents", wraps=document_reader.read_documents) as read:
            result = knowledge_upload.update_knowledge_index(base_name="kb-old")
        read.assert_called_once_with(set())
        self.create_mock.assert_not_called()
        self.assertEqual(result["name"], "kb-old")
        self.assertTrue(result["unchanged"])
        # 即使资料没有变化，也清理此前已积累的老集合。
        self.cleanup.assert_called_once_with("kb-old")

    def test_modified_file_is_reprocessed(self):
        self.old.write_text("已修改并且长度不同的内容", encoding="utf-8")
        result = knowledge_upload.update_knowledge_index(base_name="kb-old")
        args = self.create_mock.call_args.args
        self.assertEqual(args[1], set())
        self.assertEqual(args[2][0]["text"], "已修改并且长度不同的内容")
        self.assertEqual(result["modified_file_count"], 1)

    def test_an_older_explicit_baseline_is_rejected_before_cleanup(self):
        # 不能用旧基线构建新集合后删除较新的集合，否则会丢失新增资料。
        self.mocks["list_indexes"].return_value = ["kb-newer", "kb-old"]
        with self.assertRaises(knowledge_upload.UploadError):
            knowledge_upload.update_knowledge_index(base_name="kb-old")
        self.create_mock.assert_not_called()
        self.cleanup.assert_not_called()

    def test_cleanup_failure_is_reported_without_losing_updated_result(self):
        (self.root / "new.txt").write_text("新资料", encoding="utf-8")
        self.cleanup.return_value = {"kept": "kb-new", "deleted": [],
                                     "failed": [{"name": "kb-old", "error_type": "PermissionError"}],
                                     "skipped_unready": [], "aborted": False}
        self.cleanup.side_effect = None
        result = knowledge_upload.update_knowledge_index(base_name="kb-old")
        self.assertEqual(result["name"], "kb-new")
        self.assertEqual(result["cleanup"]["failed"][0]["name"], "kb-old")

    def test_cleanup_exception_is_reported_separately_from_successful_update(self):
        (self.root / "new.txt").write_text("新资料", encoding="utf-8")
        self.cleanup.side_effect = PermissionError("敏感内部详情")
        result = knowledge_upload.update_knowledge_index(base_name="kb-old")
        self.assertEqual(result["name"], "kb-new")
        self.assertEqual(result["cleanup"]["failed"][0]["error_type"], "PermissionError")
        self.assertNotIn("敏感内部详情", str(result["cleanup"]))

    def test_update_lock_blocks_another_process(self):
        # 子进程尝试同一锁：主进程持有时超时，释放后能够取得。
        script = (
            "import sys\n"
            "from filelock import FileLock, Timeout\n"
            "try:\n"
            "    with FileLock(sys.argv[1], timeout=0.2):\n"
            "        pass\n"
            "except Timeout:\n"
            "    sys.exit(7)\n"
        )
        lock_path = self.root / "knowledge-update.lock"
        command = [sys.executable, "-B", "-c", script, str(lock_path)]
        with knowledge_upload.knowledge_update_lock():
            blocked = subprocess.run(command, capture_output=True, timeout=5)
            self.assertEqual(blocked.returncode, 7, blocked.stderr.decode())
        available = subprocess.run(command, capture_output=True, timeout=5)
        self.assertEqual(available.returncode, 0, available.stderr.decode())

    def test_deleted_file_not_carried_into_new_snapshot(self):
        (self.root / "new.txt").write_text("新资料", encoding="utf-8")
        self.old.unlink()
        result = knowledge_upload.update_knowledge_index(base_name="kb-old")
        args = self.create_mock.call_args.args
        self.assertEqual(args[1], set())
        self.assertNotIn("old.txt", args[3])
        self.assertEqual(result["deleted_file_count"], 1)

    def test_existing_document_read_failure_does_not_publish(self):
        self.old.write_text("已修改", encoding="utf-8")
        with patch.object(document_reader, "read_documents", return_value=([], [{"source": "old.txt", "message": "读取失败"}])):
            with self.assertRaises(knowledge_upload.UploadError):
                knowledge_upload.update_knowledge_index(base_name="kb-old")
        self.create_mock.assert_not_called()
        self.cleanup.assert_not_called()

    def test_new_document_read_failure_does_not_publish(self):
        (self.root / "new.txt").write_text("新增", encoding="utf-8")
        with patch.object(document_reader, "read_documents", return_value=([], [{"source": "new.txt", "message": "读取失败"}])):
            with self.assertRaisesRegex(knowledge_upload.UploadError, "new.txt"):
                knowledge_upload.update_knowledge_index(base_name="kb-old")
        self.create_mock.assert_not_called()
        self.cleanup.assert_not_called()

    def test_upload_preserves_missing_legacy_source(self):
        self.old.unlink()
        (self.root / "new.txt").write_text("新资料", encoding="utf-8")
        self.mocks["read_index_manifest"].return_value = None
        with patch.object(knowledge_upload, "match_legacy_sources", return_value=set()):
            result = knowledge_upload.update_knowledge_index(base_name="kb-old", preserve_missing=True)
        args = self.create_mock.call_args.args
        self.assertEqual(args[1], {"old.txt"})
        self.assertEqual(args[3]["old.txt"], {"size": 0, "mtime_ns": -1})
        self.assertEqual(result["retained_missing_count"], 1)
        self.assertEqual(result["deleted_file_count"], 0)

    def test_changes_during_build_prevent_publication(self):
        added = self.root / "new.txt"
        added.write_text("新增", encoding="utf-8")
        def change_during_build(*args, **kwargs):
            added.write_text("新增后又被修改", encoding="utf-8")
            kwargs["validate_before_publish"]()
        self.create_mock.side_effect = change_during_build
        with self.assertRaisesRegex(knowledge_upload.UploadError, "更新过程中发生变化"):
            knowledge_upload.update_knowledge_index(base_name="kb-old")
        self.cleanup.assert_not_called()

    def test_legacy_migration_compares_before_reuse(self):
        self.mocks["read_index_manifest"].return_value = None
        with patch.object(knowledge_upload, "match_legacy_sources", return_value={"old.txt"}) as match:
            result = knowledge_upload.update_knowledge_index(base_name="kb-old")
        match.assert_called_once()
        self.assertEqual(self.create_mock.call_args.args[1], {"old.txt"})
        self.assertEqual(self.create_mock.call_args.args[2], [])
        self.assertTrue(result["legacy"])


class TemporaryChromaIntegrationTests(unittest.TestCase):
    def test_real_temporary_chroma_reuses_vectors_and_supports_search(self):
        # 用独立的临时目录和模拟向量模型完整走一遍资料读取、建库与检索。
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "资料"
            docs.mkdir()
            (docs / "old.txt").write_text("第一份旧资料。", encoding="utf-8")
            embed = Mock(side_effect=lambda texts: [[1.0] + [0.0] * 511 for _ in texts])
            with patch.object(document_reader, "DOCUMENTS_DIR", docs), \
                    patch.object(chroma_index, "CHROMA_DIR", root / "chroma"), \
                    patch.object(chroma_index, "embed_texts", embed):
                first = knowledge_upload.update_knowledge_index()
                self.assertEqual(first["new_chunk_count"], 1)
                embed.reset_mock()
                (docs / "new.txt").write_text("第二份新增资料。", encoding="utf-8")
                second = knowledge_upload.update_knowledge_index(base_name=first["name"])
                self.assertNotEqual(first["name"], second["name"])
                self.assertEqual((second["reused_chunk_count"], second["new_chunk_count"]), (1, 1))
                embed.assert_called_once_with(["第二份新增资料。"])
                # 新集合发布成功后，旧集合会被删除。
                self.assertNotIn(first["name"], chroma_index.list_indexes())
                self.assertEqual(chroma_index.list_indexes(), [second["name"]])
                self.assertEqual(chroma_index._get_ready_collection(second["name"])[1], 2)
                results, count = chroma_index.search_index("问题", second["name"], 2)
                self.assertEqual(count, 2)
                self.assertEqual({result["source"] for result in results}, {"old.txt", "new.txt"})

    def test_unchanged_update_cleans_old_collection_without_reembedding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "资料"
            docs.mkdir()
            (docs / "资料.txt").write_text("有效资料。", encoding="utf-8")
            embed = Mock(side_effect=lambda texts: [[1.0] + [0.0] * 511 for _ in texts])
            with patch.object(document_reader, "DOCUMENTS_DIR", docs), \
                    patch.object(chroma_index, "CHROMA_DIR", root / "chroma"), \
                    patch.object(chroma_index, "embed_texts", embed):
                current = knowledge_upload.update_knowledge_index()
                old = chroma_index.create_index([{"source": "old.txt", "number": 1,
                    "start": 0, "end": 2, "page": None, "text": "旧版"}])
                old_collection = chroma_index._get_ready_collection(old["name"])[0]
                old_metadata = dict(old_collection.metadata)
                old_metadata["created_at"] = "2000-01-01T00:00:00+08:00"
                old_collection.modify(metadata=old_metadata)
                embed.reset_mock()
                result = knowledge_upload.update_knowledge_index(base_name=current["name"])
                self.assertTrue(result["unchanged"])
                self.assertEqual(result["name"], current["name"])
                self.assertEqual(chroma_index.list_indexes(), [current["name"]])
                embed.assert_not_called()

    def test_update_failure_leaves_existing_collection_intact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "资料"
            docs.mkdir()
            (docs / "资料.txt").write_text("原资料。", encoding="utf-8")
            embed = Mock(side_effect=lambda texts: [[1.0] + [0.0] * 511 for _ in texts])
            with patch.object(document_reader, "DOCUMENTS_DIR", docs), \
                    patch.object(chroma_index, "CHROMA_DIR", root / "chroma"), \
                    patch.object(chroma_index, "embed_texts", embed):
                first = knowledge_upload.update_knowledge_index()
                (docs / "新资料.txt").write_text("新增资料。", encoding="utf-8")
                embed.side_effect = RuntimeError("模型失败")
                with self.assertRaisesRegex(RuntimeError, "模型失败"):
                    knowledge_upload.update_knowledge_index(base_name=first["name"])
                self.assertEqual(chroma_index.list_indexes(), [first["name"]])
                self.assertEqual(chroma_index._get_ready_collection(first["name"])[1], 1)

    def test_upload_preserves_missing_then_reappeared_file_is_reprocessed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "资料"
            docs.mkdir()
            original = docs / "old.txt"
            original.write_text("第一份旧资料。", encoding="utf-8")
            old_info = original.stat()
            embed = Mock(side_effect=lambda texts: [[1.0] + [0.0] * 511 for _ in texts])
            with patch.object(document_reader, "DOCUMENTS_DIR", docs), \
                    patch.object(chroma_index, "CHROMA_DIR", root / "chroma"), \
                    patch.object(chroma_index, "embed_texts", embed):
                first = knowledge_upload.update_knowledge_index()
                original.unlink()
                (docs / "new.txt").write_text("新增资料。", encoding="utf-8")
                embed.reset_mock()
                second = knowledge_upload.update_knowledge_index(base_name=first["name"], preserve_missing=True)
                self.assertEqual(second["retained_missing_count"], 1)
                self.assertEqual(second["deleted_file_count"], 0)
                self.assertEqual(chroma_index._get_ready_collection(second["name"])[1], 2)
                embed.assert_called_once_with(["新增资料。"])
                # 再次上传时继续保留同一份缺失来源，只处理第二个新文件。
                (docs / "extra.txt").write_text("额外资料。", encoding="utf-8")
                embed.reset_mock()
                second = knowledge_upload.update_knowledge_index(base_name=second["name"], preserve_missing=True)
                self.assertEqual(second["retained_missing_count"], 1)
                self.assertEqual(chroma_index._get_ready_collection(second["name"])[1], 3)
                embed.assert_called_once_with(["额外资料。"])
                # 原文件以相同大小和修改时间重新出现，因缺失标记仍须重新解析。
                original.write_text("不一样的资料。", encoding="utf-8")
                self.assertEqual(original.stat().st_size, old_info.st_size)
                import os
                os.utime(original, ns=(old_info.st_atime_ns, old_info.st_mtime_ns))
                embed.reset_mock()
                third = knowledge_upload.update_knowledge_index(base_name=second["name"], preserve_missing=True)
                embed.assert_called_once_with(["不一样的资料。"])
                self.assertEqual(third["modified_file_count"], 1)
                original.unlink()
                fourth = knowledge_upload.update_knowledge_index(base_name=third["name"])
                self.assertEqual(fourth["deleted_file_count"], 1)
                self.assertEqual(chroma_index._get_ready_collection(fourth["name"])[1], 2)


if __name__ == "__main__":
    unittest.main()
