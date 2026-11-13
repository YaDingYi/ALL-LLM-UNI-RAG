'''在临时Chroma数据库验证增量快照，不读取用户资料或调用嵌入模型。'''

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import chroma_index
import incremental_index


def chunk(source, number, text):
    return {"source": source, "number": number, "start": 0, "end": len(text),
            "page": None, "text": text}


def signature(size=1, mtime=1):
    return {"size": size, "mtime_ns": mtime}


class IncrementalIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path_patch = patch.object(chroma_index, "CHROMA_DIR", Path(self.temporary.name))
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        # 用确定性的512维向量替代模型，让测试能检查调用次数。
        self.embed = Mock(side_effect=lambda texts: [[float(len(text))] + [0.0] * 511 for text in texts])
        self.embed_patch = patch.object(chroma_index, "embed_texts", self.embed)
        self.embed_patch.start()
        self.addCleanup(self.embed_patch.stop)

    def create(self, chunks, signatures):
        return incremental_index.create_incremental_index(None, set(), chunks, signatures)

    def collection(self, name):
        return chroma_index._get_ready_collection(name)[0]

    def test_initial_build_and_manifest(self):
        result = self.create([chunk("a.txt", 1, "资料一")], {"a.txt": signature()})
        self.assertEqual(result["mode"], "initial")
        self.assertEqual(result["new_chunk_count"], 1)
        self.assertEqual(result["reused_chunk_count"], 0)
        self.assertEqual(incremental_index.read_index_manifest(result["name"]), {"a.txt": signature()})
        self.assertEqual(incremental_index.list_index_sources(result["name"]), {"a.txt"})
        self.embed.assert_called_once_with(["资料一"])

    def test_changed_added_deleted_and_unchanged_sources(self):
        initial = self.create([chunk("a.txt", 1, "原始A"), chunk("b.txt", 1, "原始B"),
                               chunk("c.txt", 1, "原始C")],
                              {"a.txt": signature(), "b.txt": signature(), "c.txt": signature()})
        old = self.collection(initial["name"])
        old_a = old.get(where={"source": "a.txt"}, include=["embeddings"])["embeddings"][0].tolist()
        self.embed.reset_mock()
        progress = Mock()
        validate = Mock()
        result = incremental_index.create_incremental_index(
            initial["name"], {"a.txt"}, [chunk("b.txt", 1, "修改后B"), chunk("d.txt", 1, "新增D")],
            {"a.txt": signature(), "b.txt": signature(2, 2), "d.txt": signature()},
            on_progress=progress, validate_before_publish=validate,
        )
        self.assertEqual(result["mode"], "incremental")
        self.assertEqual((result["reused_chunk_count"], result["new_chunk_count"]), (1, 2))
        self.embed.assert_called_once_with(["修改后B", "新增D"])
        self.assertEqual(self.collection(initial["name"]).count(), 3)
        current = self.collection(result["name"])
        self.assertEqual(incremental_index.list_index_sources(result["name"]), {"a.txt", "b.txt", "d.txt"})
        self.assertEqual(current.get(where={"source": "a.txt"}, include=["embeddings"])["embeddings"][0].tolist(), old_a)
        self.assertEqual(current.get(where={"source": "b.txt"}, include=["documents"])["documents"], ["修改后B"])
        self.assertEqual(progress.call_args.args, (3, 3))
        validate.assert_called_once()

    def test_no_changes_returns_original_collection(self):
        initial = self.create([chunk("a.txt", 1, "资料")], {"a.txt": signature()})
        self.embed.reset_mock()
        validate = Mock()
        result = incremental_index.create_incremental_index(initial["name"], {"a.txt"}, [],
                                                            {"a.txt": signature()}, validate_before_publish=validate)
        self.assertEqual(result["name"], initial["name"])
        self.assertEqual(result["mode"], "unchanged")
        self.embed.assert_not_called()
        validate.assert_called_once()

    def test_legacy_snapshot_reuses_only_exactly_matching_sources(self):
        old_chunks = [chunk("a.txt", 1, "旧A"), chunk("b.txt", 1, "旧B")]
        legacy = chroma_index.create_index(old_chunks)
        self.assertIsNone(incremental_index.read_index_manifest(legacy["name"]))
        new_chunks = [chunk("a.txt", 1, "旧A"), chunk("b.txt", 1, "新B")]
        matches = incremental_index.match_legacy_sources(legacy["name"], new_chunks)
        self.assertEqual(matches, {"a.txt"})
        self.embed.reset_mock()
        updated = incremental_index.create_incremental_index(legacy["name"], matches, [new_chunks[1]],
                                                            {"a.txt": signature(), "b.txt": signature(2, 2)})
        self.embed.assert_called_once_with(["新B"])
        self.assertEqual(incremental_index.read_index_manifest(updated["name"])["a.txt"], signature())

    def test_whitespace_chunks_match_legacy_filter(self):
        chunks = [chunk("a.txt", 1, "内容"), chunk("a.txt", 2, "   ")]
        legacy = chroma_index.create_index(chunks)
        self.assertEqual(incremental_index.match_legacy_sources(legacy["name"], chunks), {"a.txt"})
        created = self.create(chunks, {"a.txt": signature()})
        self.assertEqual(created["chunk_count"], 1)
        self.assertEqual(self.collection(created["name"]).count(), 1)

    def test_failed_write_does_not_touch_original_or_publish_new_collection(self):
        original = self.create([chunk("a.txt", 1, "原始")], {"a.txt": signature()})
        self.embed.side_effect = RuntimeError("test failure")
        with self.assertRaises(RuntimeError):
            incremental_index.create_incremental_index(original["name"], {"a.txt"}, [chunk("b.txt", 1, "新增")],
                                                       {"a.txt": signature(), "b.txt": signature()})
        self.assertEqual(chroma_index.list_indexes(), [original["name"]])
        self.assertEqual(self.collection(original["name"]).count(), 1)

    def test_validate_before_publish_failure_removes_new_collection(self):
        original = self.create([chunk("a.txt", 1, "原始")], {"a.txt": signature()})
        def fail():
            raise ValueError("文件在更新期间发生变化")
        with self.assertRaisesRegex(ValueError, "文件在更新期间发生变化"):
            incremental_index.create_incremental_index(original["name"], {"a.txt"}, [chunk("b.txt", 1, "新增")],
                                                       {"a.txt": signature(), "b.txt": signature()},
                                                       validate_before_publish=fail)
        self.assertEqual(chroma_index.list_indexes(), [original["name"]])
        self.assertEqual(self.collection(original["name"]).count(), 1)

    def test_changed_file_cannot_be_declared_reused(self):
        initial = self.create([chunk("a.txt", 1, "资料")], {"a.txt": signature()})
        with self.assertRaisesRegex(ValueError, "已发生变化"):
            incremental_index.create_incremental_index(initial["name"], {"a.txt"}, [], {"a.txt": signature(1, 2)})

    def test_changed_split_configuration_requires_revalidation(self):
        initial = self.create([chunk("a.txt", 1, "资料")], {"a.txt": signature()})
        with patch.object(incremental_index, "CHUNKING_VERSION", "new-rule"):
            self.assertIsNone(incremental_index.read_index_manifest(initial["name"]))
            self.assertEqual(incremental_index.match_legacy_sources(initial["name"], [chunk("a.txt", 1, "资料")]), {"a.txt"})


if __name__ == "__main__":
    unittest.main()
