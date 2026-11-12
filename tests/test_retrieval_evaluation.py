'''用临时文件和假集合验证评估预检与汇总，不连接Chroma、不调用模型。'''

# 使用标准库生成测试题库和内容摘要。
import hashlib
import json
# 使用临时目录隔离测试，不触碰项目中的原资料。
from pathlib import Path
from tempfile import TemporaryDirectory
# 使用标准库测试框架，无需安装额外依赖。
import unittest

# 导入函数不会执行main，也不会加载向量模型。
from evaluation.run_retrieval import audit_collection, load_dataset, summarize


class FakeCollection:
    '''只提供get方法的假集合；不存在修改数据库的方法。'''

    def __init__(self, text, intervals):
        # 保存测试文字和切分区间。
        self.text = text
        self.intervals = intervals

    def get(self, where, include):
        # 预检应只按指定来源读取已存文本与位置。
        assert where == {"source": "lesson.md"}
        assert include == ["documents", "metadatas"]
        # 返回与Chroma get相同的三个平行列表。
        return {
            "ids": [f"chunk-{i}" for i in range(len(self.intervals))],
            "documents": [self.text[start:end] for start, end in self.intervals],
            "metadatas": [
                {"source": "lesson.md", "start": start, "end": end}
                for start, end in self.intervals
            ],
        }


class DatasetPreflightTests(unittest.TestCase):
    '''题库必须与原文版本、引文和位置一致。'''

    def setUp(self):
        # 每个测试拥有独立的临时目录，结束后自动清理。
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "资料" / "lesson.md"
        self.source.parent.mkdir()
        # UTF-8 BOM不应被当成正文字符计入证据位置。
        self.source.write_text("第一行\n答案保留90天。\n最后一行\n", encoding="utf-8-sig")
        self.path = self.root / "evaluation" / "questions.json"
        self.path.parent.mkdir()
        # 最小题库只保留预检真正需要的字段。
        self.dataset = {
            "source_root": "../资料",
            "source_versions": [{
                "source": "lesson.md",
                "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            }],
            "cases": [{
                "id": "one",
                "answerable": True,
                "gold_evidence": [{
                    "source": "lesson.md", "page": None,
                    "line_start": 2, "line_end": 2, "quote": "保留90天",
                }],
            }],
        }

    def write_dataset(self):
        # 写入临时题库，不修改真实示范题。
        self.path.write_text(json.dumps(self.dataset, ensure_ascii=False), encoding="utf-8")

    def test_valid_quote_maps_to_exact_character_span(self):
        # 正确摘要、行号和引文应得到精确字符区间。
        self.write_dataset()
        _, texts, mapped = load_dataset(self.path)
        evidence = mapped["one"][0]
        text = texts["lesson.md"]
        self.assertEqual(text[evidence["start"]:evidence["end"]], "保留90天")
        self.assertEqual(evidence["start"], len("第一行\n答案"))

    def test_changed_source_digest_is_rejected(self):
        # 原文更新后不能继续用旧标签评分。
        self.write_dataset()
        self.source.write_text("正文已经改变", encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "原文版本变化"):
            load_dataset(self.path)

    def test_quote_at_wrong_line_is_rejected(self):
        # 引文确实存在于文件，但错误行号仍必须被发现。
        self.dataset["cases"][0]["gold_evidence"][0]["line_start"] = 1
        self.dataset["cases"][0]["gold_evidence"][0]["line_end"] = 1
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "引用无法唯一定位"):
            load_dataset(self.path)

    def test_duplicate_case_ids_are_rejected(self):
        # 重复题号不能通过字典覆盖而偷偷减少评估题数。
        self.dataset["cases"].append(self.dataset["cases"][0])
        self.write_dataset()
        with self.assertRaisesRegex(ValueError, "题号重复"):
            load_dataset(self.path)

    def test_out_of_range_lines_are_rejected(self):
        # 结束行越界时Python仍会返回包含引文的切片，必须主动拒绝。
        for start, end in ((0, 2), (-10, 2), (2, 99), (3, 2)):
            with self.subTest(start=start, end=end):
                self.dataset["cases"][0]["gold_evidence"][0]["line_start"] = start
                self.dataset["cases"][0]["gold_evidence"][0]["line_end"] = end
                self.write_dataset()
                with self.assertRaisesRegex(ValueError, "证据行号越界"):
                    load_dataset(self.path)


class CollectionPreflightTests(unittest.TestCase):
    '''不只是有命中证据，原文入库版本及完整性也要通过预检。'''

    def test_overlapping_chunks_cover_full_source(self):
        # 重叠片段仍应共同覆盖全部文字，不要求无重叠切分。
        text = "0123456789"
        records = audit_collection(FakeCollection(text, [(0, 6), (4, 10)]), {"lesson.md": text})
        self.assertEqual(len(records), 2)
        self.assertTrue(all(row["page"] is None for row in records))

    def test_missing_tail_is_rejected(self):
        # 每条现存片段都与原文相符，但遗漏文件末尾也必须失败。
        text = "0123456789"
        with self.assertRaisesRegex(ValueError, "未完整包含"):
            audit_collection(FakeCollection(text, [(0, 6), (4, 9)]), {"lesson.md": text})

    def test_old_database_text_is_rejected(self):
        # 文件长度相同不代表版本一致，要核对实际文字。
        with self.assertRaisesRegex(ValueError, "片段与题库原文不一致"):
            audit_collection(FakeCollection("旧的资料", [(0, 4)]), {"lesson.md": "新的资料"})

    def test_missing_source_is_rejected(self):
        # 没有该来源的任何片段时，不能给出看似合法的检索零分。
        with self.assertRaisesRegex(ValueError, "未完整包含"):
            audit_collection(FakeCollection("资料", []), {"lesson.md": "资料"})


class SummaryTests(unittest.TestCase):
    '''失败题保留在正向指标分母中，无答案题单独保留但不参与均值。'''

    def test_failed_answerable_case_remains_in_denominator(self):
        # 一道成功正向题、一道失败正向题，以及一道成功的无答案题。
        keys = ("evidence_hit_at_1", "evidence_hit_at_k", "evidence_recall_at_k", "evidence_rr_at_k", "all_evidence_at_k")
        rows = [
            {"answerable": True, "status": "ok", "metrics": dict.fromkeys(keys, 1.0), "retrieval_seconds": 0.2},
            {"answerable": True, "status": "error", "metrics": None, "retrieval_seconds": 1.0},
            {"answerable": False, "status": "ok", "metrics": dict.fromkeys(keys, None), "retrieval_seconds": 0.4},
        ]
        result = summarize(rows)
        # 正向分母为2，不能删除失败题后报告100%。
        self.assertEqual(result["answerable_count"], 2)
        self.assertEqual(result["case_count"], 3)
        self.assertEqual(result["completed_count"], 2)
        for key in keys:
            self.assertEqual(result[key], 0.5)
        # 耗时中位数只统计完成的请求，并与错误率分开解释。
        self.assertAlmostEqual(result["median_retrieval_seconds"], 0.3)

    def test_no_answerable_cases_produce_no_positive_score(self):
        # 仅有无答案题时，正向指标没有合法分母。
        result = summarize([{"answerable": False, "status": "ok", "metrics": None, "retrieval_seconds": 0.2}])
        self.assertEqual(result["answerable_count"], 0)
        self.assertIsNone(result["evidence_recall_at_k"])


# 直接运行本文件时执行测试；导入不会启动任何实际检索。
if __name__ == "__main__":
    unittest.main()
