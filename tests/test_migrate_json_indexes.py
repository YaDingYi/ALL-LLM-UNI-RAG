'''旧索引迁移只针对空库的最新文件；测试不会读取用户资料或数据库。'''

from contextlib import contextmanager
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import migrate_json_indexes as migration
import chroma_index
import index_demo


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.locked = False

        @contextmanager
        def fake_lock():
            self.assertFalse(self.locked)
            self.locked = True
            try:
                yield
            finally:
                self.locked = False

        self.patches = [
            patch.object(migration, "knowledge_update_lock", side_effect=fake_lock),
            patch.object(migration, "list_json_indexes", return_value=["index-2.json", "index-1.json"]),
            patch.object(migration, "list_chroma_indexes", side_effect=self.check_chroma),
            patch.object(migration, "load_index", return_value={"records": ["mock"]}),
            patch.object(migration, "import_index", side_effect=self.import_latest),
        ]
        self.lock, self.list_json, self.list_chroma, self.load, self.importer = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)

    def check_chroma(self):
        self.assertTrue(self.locked, "检查Chroma时必须持有更新锁")
        return []

    def import_latest(self, data, name):
        self.assertTrue(self.locked, "导入时必须持有更新锁")
        return {"already_exists": False, "name": "kb-new", "chunk_count": 1}

    def test_empty_library_migrates_only_latest_json(self):
        with redirect_stdout(io.StringIO()):
            result = migration.migrate_json_indexes()
        self.assertEqual(result, {"migrated": 1, "reused": 0, "failed": 0})
        self.load.assert_called_once_with("index-2.json")
        self.importer.assert_called_once_with({"records": ["mock"]}, "index-2.json")
        self.assertFalse(self.locked)

    def test_existing_chroma_blocks_migration_without_reading_json(self):
        self.list_chroma.side_effect = lambda: ["kb-current"] if self.locked else self.fail("未持有锁")
        output = io.StringIO()
        with redirect_stdout(output):
            result = migration.migrate_json_indexes()
        self.assertEqual(result["failed"], 1)
        self.assertIn("已有可用的 Chroma 知识库", output.getvalue())
        self.load.assert_not_called()
        self.importer.assert_not_called()
        self.assertFalse(self.locked)

    def test_invalid_latest_does_not_fall_back_to_older_json(self):
        self.load.side_effect = ValueError("invalid")
        with redirect_stdout(io.StringIO()):
            result = migration.migrate_json_indexes()
        self.assertEqual(result["failed"], 1)
        self.load.assert_called_once_with("index-2.json")
        self.importer.assert_not_called()
        self.assertFalse(self.locked)

    def test_no_json_does_not_touch_chroma(self):
        self.list_json.return_value = []
        with redirect_stdout(io.StringIO()):
            result = migration.migrate_json_indexes()
        self.assertEqual(result, {"migrated": 0, "reused": 0, "failed": 0})
        self.lock.assert_not_called()
        self.list_chroma.assert_not_called()
        self.importer.assert_not_called()

    def test_existing_library_check_failure_does_not_import(self):
        self.list_chroma.side_effect = RuntimeError("database unavailable")
        with self.assertRaises(RuntimeError):
            migration.migrate_json_indexes()
        self.load.assert_not_called()
        self.importer.assert_not_called()
        self.assertFalse(self.locked)


class MigrationIntegrationTests(unittest.TestCase):
    def test_latest_only_in_isolated_chroma_and_original_files_intact(self):
        '''在临时目录使用真实Chroma，确认只迁移最新JSON、保留原文件且拒绝重复导入。'''
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_dir = root / "vector_index"
            old_dir.mkdir()

            def snapshot(text):
                chunk = {"id": "test-1", "source": "demo.txt", "page": None,
                         "number": 1, "start": 0, "end": len(text), "text": text,
                         "vector": [1.0] + [0.0] * (chroma_index.DIMENSION - 1)}
                return {"version": 1, "model": chroma_index.MODEL_NAME,
                        "preprocessing": chroma_index.PREPROCESSING,
                        "dimension": chroma_index.DIMENSION, "records": [chunk],
                        "chunk_count": 1, "file_count": 1}

            filenames = ["index-20260101-000000-test.json", "index-20260102-000000-test.json"]
            for name, text in zip(filenames, ("早期示例", "最新示例")):
                (old_dir / name).write_text(json.dumps(snapshot(text), ensure_ascii=False), encoding="utf-8")
            originals = {name: (old_dir / name).read_bytes() for name in filenames}
            with patch.object(index_demo, "INDEX_DIR", old_dir), \
                    patch.object(chroma_index, "CHROMA_DIR", root / "chroma"), \
                    redirect_stdout(io.StringIO()):
                result = migration.migrate_json_indexes()
                self.assertEqual(result["migrated"], 1)
                names = chroma_index.list_indexes()
                self.assertEqual(len(names), 1)
                collection, _ = chroma_index._get_ready_collection(names[0])
                self.assertEqual(collection.get(include=["documents"])["documents"], ["最新示例"])
                self.assertEqual(migration.migrate_json_indexes()["failed"], 1)
                self.assertEqual(chroma_index.list_indexes(), names)
            self.assertEqual(originals, {name: (old_dir / name).read_bytes() for name in filenames})


if __name__ == "__main__":
    unittest.main()
