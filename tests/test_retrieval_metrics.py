'''验证证据覆盖指标的边界语义；不读取Chroma、不调用模型。'''

# 使用Python标准库的测试框架，不增加安装依赖。
import unittest

# 导入本阶段新增的纯计算函数。
from evaluation.retrieval_metrics import interval_coverage, score_retrieval


# 为测试生成一个简短的证据或片段对象，避免重复编写相同字典。
def span(start, end, source="docs/C3/C3.md", page=None):
    '''创建包含原文来源、页码和字符范围的测试对象。'''
    # 两个被测函数不需要向量或正文，坐标就足以测试区间计算。
    return {"source": source, "page": page, "start": start, "end": end}


# 将相关测试集中在一个类中，unittest会自动发现test_开头的方法。
class RetrievalMetricsTests(unittest.TestCase):
    '''检查区间合并、跨片段完成、排序、来源隔离和无答案题。'''

    # 检查重叠、重复、越界区间不会造成重复计数。
    def test_interval_union_clips_and_deduplicates(self):
        '''目标只有10个字符，重叠和目标外的字符不能使结果超过10。'''
        # 区间顺序打乱，同时包含重复区间和超出目标范围的部分。
        intervals = [(15, 50), (11, 16), (-100, 12), (11, 16)]
        # 区间并集完整覆盖[10,20)，因此结果为10。
        self.assertEqual(interval_coverage(10, 20, intervals), 10)
        # 没有区间时覆盖字符数为0。
        self.assertEqual(interval_coverage(10, 20, []), 0)

    # 检查紧邻的区间可以拼接，但真实缺口不能被抹去。
    def test_adjacent_intervals_and_real_gap(self):
        '''左闭右开区间[10,15)与[15,20)可以连续覆盖目标。'''
        # 两个区间无缝相接，覆盖10个字符。
        self.assertEqual(interval_coverage(10, 20, [(10, 15), (15, 20)]), 10)
        # [14,15)缺少一个字符，因此只能覆盖9个字符。
        self.assertEqual(interval_coverage(10, 20, [(10, 14), (15, 20)]), 9)
        # 用评分函数确认90%的覆盖不能被当成完整证据。
        result = score_retrieval([span(10, 20)], [span(10, 14), span(15, 20)])
        # 字符覆盖比例为0.9。
        self.assertEqual(result["character_coverage"], [0.9])
        # 完整证据命中仍为0。
        self.assertEqual(result["evidence_hit_at_k"], 0.0)

    # 检查多个片段共同完成一条证据时，记录正确的前缀排名。
    def test_evidence_completed_by_two_chunks(self):
        '''第一片段只覆盖一半，前两片段一起完成证据，倒数排名应为1/2。'''
        # 标准证据跨越两个连续的片段。
        references = [span(10, 15), span(15, 20)]
        # 使用前两条即可完整覆盖。
        result = score_retrieval([span(10, 20)], references)
        # 第一名还不足以完成整条证据。
        self.assertEqual(result["evidence_hit_at_1"], 0.0)
        # 前k条已完成唯一证据。
        self.assertEqual(result["evidence_hit_at_k"], 1.0)
        # 唯一的一条证据被找齐，证据召回为1。
        self.assertEqual(result["evidence_recall_at_k"], 1.0)
        # 最早在前两条完成，因此倒数排名为0.5。
        self.assertEqual(result["evidence_rr_at_k"], 0.5)
        # 保留排名2，以便人工核对。
        self.assertEqual(result["first_completion_rank"], 2)
        # 全部证据已齐全。
        self.assertEqual(result["all_evidence_at_k"], 1.0)
        # 当只看前一条时，仍然只有一半字符被覆盖。
        first_only = score_retrieval([span(10, 20)], references, top_k=1)
        # top_k必须真正限制可使用的片段数。
        self.assertEqual(first_only["character_coverage"], [0.5])
        # 未完成时，不应产生完成排名。
        self.assertIsNone(first_only["first_completion_rank"])

    # 检查第一名、第三名和截断前缀下的排名计算。
    def test_first_third_and_missing_ranks(self):
        '''排序保持输入顺序，不能为提高成绩重新排列检索结果。'''
        # 第一名已完整覆盖证据时，倒数排名为1。
        first = score_retrieval([span(10, 20)], [span(0, 30)])
        # 第一名直接完成证据。
        self.assertEqual(first["evidence_rr_at_k"], 1.0)
        # 先放两个无关片段，再放一个覆盖证据的片段。
        references = [span(30, 40), span(40, 50), span(0, 30)]
        # 默认检查前三条，第三条才完成证据。
        third = score_retrieval([span(10, 20)], references)
        # 倒数排名应是1/3。
        self.assertAlmostEqual(third["evidence_rr_at_k"], 1 / 3)
        # 明确检查完成位置为第三名。
        self.assertEqual(third["first_completion_rank"], 3)
        # 只保留前两条时，完整证据尚未出现。
        missing = score_retrieval([span(10, 20)], references, top_k=2)
        # 未命中的倒数排名为0。
        self.assertEqual(missing["evidence_rr_at_k"], 0.0)
        # 未命中不能用0作为排名，而应使用None。
        self.assertIsNone(missing["first_completion_rank"])

    # 检查文件和页码同时匹配，才允许合并坐标。
    def test_source_and_page_are_isolated(self):
        '''另一文件或另一页的同样坐标，不是当前证据的覆盖。'''
        # 证据位于C3文件的第2页。
        evidence = span(10, 20, page=2)
        # 第一个片段文件错误，第二个页码错误，只有第三个片段部分有效。
        references = [span(10, 20, source="other.pdf", page=2), span(10, 20, page=1), span(10, 15, page=2)]
        # 按来源与页面过滤后，只剩一半字符被覆盖。
        result = score_retrieval([evidence], references)
        # 错误来源不能补齐缺少的一半。
        self.assertEqual(result["character_coverage"], [0.5])
        # 证据尚未完整命中。
        self.assertEqual(result["covered_evidence"], [False])

    # 检查证据召回的分母是证据数量，而不是片段数量或字符数量。
    def test_recall_counts_evidence_items(self):
        '''两条证据只完成一条，即使命中多次，证据召回仍然为1/2。'''
        # 两条证据长度不同，防止实现误用字符长度加权。
        evidences = [span(10, 20), span(100, 140)]
        # 重复找到第一条证据，第二条没有找到。
        result = score_retrieval(evidences, [span(10, 20), span(10, 20)])
        # 至少一条证据已命中。
        self.assertEqual(result["evidence_hit_at_1"], 1.0)
        # 按完成证据条数计算召回，不能因重复片段增加分子。
        self.assertEqual(result["evidence_recall_at_k"], 0.5)
        # 一条完成、一条未完成的状态应保留。
        self.assertEqual(result["covered_evidence"], [True, False])
        # 证据没有全部找齐。
        self.assertEqual(result["all_evidence_at_k"], 0.0)

    # 检查资料不足题不生成误导性的0分或满分。
    def test_no_evidence_is_not_applicable(self):
        '''无正向证据时，即使检索返回片段，五个正向指标仍为None。'''
        # 检索返回了片段，但题库本身没有正向证据。
        result = score_retrieval([], [span(10, 20)])
        # 遍历五项汇总指标，避免只验证其中一个。
        for key in ("evidence_hit_at_1", "evidence_hit_at_k", "evidence_recall_at_k", "evidence_rr_at_k", "all_evidence_at_k"):
            # None表示“不适用”，不是“没有检索成功”。
            self.assertIsNone(result[key])
        # 没有证据时状态列表为空。
        self.assertEqual(result["covered_evidence"], [])
        # 没有证据时字符覆盖列表为空。
        self.assertEqual(result["character_coverage"], [])
        # 没有证据时完成排名也不适用。
        self.assertIsNone(result["first_completion_rank"])

    # 检查正向题完全没有返回片段，与无答案题严格区分。
    def test_no_results_is_zero_for_answerable_case(self):
        '''有证据目标但没有检索结果时，记录未命中与0覆盖。'''
        # 保留一个标准证据，但返回结果为空。
        result = score_retrieval([span(10, 20)], [])
        # 正向题无检索结果的证据召回为0。
        self.assertEqual(result["evidence_recall_at_k"], 0.0)
        # 字符覆盖比例也为0。
        self.assertEqual(result["character_coverage"], [0.0])
        # 未完成任何证据，所以没有完成排名。
        self.assertIsNone(result["first_completion_rank"])

    # 检查会造成无意义分母或切片行为的输入被明确拒绝。
    def test_invalid_parameters_are_rejected(self):
        '''拒绝零长度证据区间和不合理的top_k参数。'''
        # 空目标区间不能计算覆盖比例。
        with self.assertRaises(ValueError):
            # 触发明确的区间长度错误。
            interval_coverage(10, 10, [])
        # 分别检查零、负数、布尔值和非整数数量。
        for top_k in (0, -1, True, 1.5):
            # 每一种无效数量都应明确报错。
            with self.assertRaises(ValueError):
                # 即使没有标准证据，也不能接受无效top_k。
                score_retrieval([], [], top_k=top_k)


# 通过python直接执行本文件时运行测试，导入时不自动运行。
if __name__ == "__main__":
    # 启动标准库测试框架并显示测试结果。
    unittest.main()
