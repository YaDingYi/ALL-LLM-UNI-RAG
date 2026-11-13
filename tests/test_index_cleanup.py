'''在临时Chroma中检查旧知识库清理；不访问用户数据库。'''

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import chroma_index
from chromadb.api.models.Collection import Collection


class IndexCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory_patch = patch.object(chroma_index, "CHROMA_DIR", Path(self.temporary.name))
        directory_patch.start()
        self.addCleanup(directory_patch.stop)
        self.vector_patch = patch.object(
            chroma_index, "embed_texts",
            side_effect=lambda texts: [[1.0] + [0.0] * 511 for _ in texts],
        )
        self.vector_patch.start()
        self.addCleanup(self.vector_patch.stop)

    def make_index(self, text):
        chunk = {"source": "test.txt", "number": 1, "start": 0,
                 "end": len(text), "page": None, "text": text}
        return chroma_index.create_index([chunk])["name"]

    def test_only_old_ready_knowledge_collections_are_removed(self):
        first = self.make_index("旧一")
        second = self.make_index("旧二")
        latest = self.make_index("最新")
        client = chroma_index._client()
        client.create_collection("other-project", embedding_function=None)
        client.create_collection("kb-in-progress", metadata={"ready": False}, embedding_function=None)

        result = chroma_index.cleanup_old_indexes(latest)

        self.assertEqual(set(result["deleted"]), {first, second})
        self.assertEqual(result["kept"], latest)
        self.assertEqual(result["failed"], [])
        self.assertEqual(result["skipped_unready"], ["kb-in-progress"])
        self.assertEqual(chroma_index.list_indexes(), [latest])
        self.assertEqual(chroma_index._get_ready_collection(latest)[1], 1)
        names = {item if isinstance(item, str) else item.name for item in client.list_collections()}
        self.assertIn("other-project", names)
        self.assertIn("kb-in-progress", names)

    def test_old_or_missing_keep_cannot_delete_anything(self):
        old = self.make_index("旧")
        latest = self.make_index("新")
        with self.assertRaisesRegex(ValueError, "不是最新"):
            chroma_index.cleanup_old_indexes(old)
        with self.assertRaises(Exception):
            chroma_index.cleanup_old_indexes("kb-not-found")
        self.assertEqual(set(chroma_index.list_indexes()), {old, latest})

    def test_failed_delete_does_not_hide_failure_or_stop_other_deletes(self):
        old_one = self.make_index("旧一")
        old_two = self.make_index("旧二")
        latest = self.make_index("最新")
        client = chroma_index._client()
        proxy = Mock(wraps=client)

        def delete(name):
            if name == old_two:
                raise PermissionError("example private details")
            return client.delete_collection(name)

        proxy.delete_collection.side_effect = delete
        with patch.object(chroma_index, "_client", return_value=proxy):
            result = chroma_index.cleanup_old_indexes(latest)
        self.assertEqual(result["deleted"], [old_one])
        self.assertEqual(result["failed"], [{"name": old_two, "error_type": "PermissionError"}])
        self.assertFalse(result["aborted"])
        self.assertEqual(set(chroma_index.list_indexes()), {old_two, latest})

    def test_newer_index_during_cleanup_aborts_remaining_deletes(self):
        old_one = self.make_index("旧一")
        old_two = self.make_index("旧二")
        latest = self.make_index("最新")
        # 模拟一次删除后出现更新的集合；每次继续删除前都重新确认最新集合。
        snapshots = [[latest, old_two, old_one], [latest, old_two, old_one],
                     ["kb-newer", latest, old_one]]
        with patch.object(chroma_index, "list_indexes", side_effect=snapshots):
            result = chroma_index.cleanup_old_indexes(latest)
        self.assertEqual(result["deleted"], [old_two])
        self.assertTrue(result["aborted"])
        self.assertEqual(result["failed"], [{"name": old_one, "error_type": "ValueError"}])
        self.assertEqual(set(chroma_index.list_indexes()), {old_one, latest})

    def test_latest_incompatible_collection_cannot_be_kept(self):
        old = self.make_index("旧")
        newest = self.make_index("新")
        collection, _ = chroma_index._get_ready_collection(newest)
        metadata = dict(collection.metadata)
        metadata["model"] = "incorrect-model"
        collection.modify(metadata=metadata)
        with self.assertRaisesRegex(ValueError, "不兼容"):
            chroma_index.cleanup_old_indexes(newest)
        self.assertEqual(set(chroma_index.list_indexes()), {old, newest})

    def test_probe_failure_preserves_every_collection(self):
        old = self.make_index("旧")
        newest = self.make_index("新")
        with patch.object(Collection, "query", side_effect=RuntimeError("查询失败")):
            with self.assertRaisesRegex(RuntimeError, "查询失败"):
                chroma_index.cleanup_old_indexes(newest)
        self.assertEqual(set(chroma_index.list_indexes()), {old, newest})

    def test_invalid_vector_or_empty_query_preserves_old_collection(self):
        old = self.make_index("旧")
        newest = self.make_index("新")
        with patch.object(Collection, "get", return_value={"ids": ["x"], "embeddings": [[0.0] * 512]}):
            with self.assertRaisesRegex(ValueError, "零向量"):
                chroma_index.cleanup_old_indexes(newest)
        with patch.object(Collection, "query", return_value={"ids": [[]], "distances": [[]]}):
            with self.assertRaisesRegex(ValueError, "无法完成向量检索"):
                chroma_index.cleanup_old_indexes(newest)
        self.assertEqual(set(chroma_index.list_indexes()), {old, newest})


if __name__ == "__main__":
    unittest.main()
