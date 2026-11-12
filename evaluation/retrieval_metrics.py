'''按原文字符区间计算证据覆盖指标，不调用向量模型或大模型。'''


# 将多个片段的覆盖范围合并，防止重叠部分被重复计数。
def interval_coverage(start, end, intervals):
    '''返回目标区间[start, end)被intervals并集覆盖的字符数；结束位置不包含。'''
    # 目标必须有正长度，否则字符覆盖比例会出现除零或无意义的分母。
    if end <= start:
        # 明确拒绝空区间和起止位置颠倒的区间。
        raise ValueError("证据结束位置必须大于起始位置。")
    # covered累计已经覆盖的字符数，初始为0。
    covered = 0
    # cursor记录已经处理到的位置，防止相邻片段的重叠被重复计算。
    cursor = start
    # 按区间起点从小到大处理，输入本身不需要预先排序。
    for left, right in sorted(intervals):
        # 只保留位于目标起点之后的部分。
        left = max(start, left)
        # 只保留位于目标终点之前的部分。
        right = min(end, right)
        # 区间超出目标或与已覆盖部分重叠时，只增加真正新增的非负长度。
        covered += max(0, right - max(cursor, left))
        # 更新已处理的最远位置，保持游标不会向后移动。
        cursor = max(cursor, right)
    # 返回区间并集在目标范围内覆盖的长度，而不是各片段长度之和。
    return covered


# 在给定检索排序上，计算完整证据的命中、覆盖和最早完成位置。
def score_retrieval(evidences, references, top_k=3):
    '''评估前top_k个片段的标注证据覆盖；允许多个片段共同覆盖一条证据。

    每条证据和片段都应提供source、page、start、end；区间为左闭右开。
    evidence_rr_at_k是最早完成一条证据的前缀排名倒数，不是传统MRR。
    无证据的题返回None指标，调用者应将其排除在正向题均值之外。
    '''
    # 返回数量必须是正整数；bool虽属于int的子类，但不是合理的检索数量。
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        # 尽早报告参数错误，避免切片得到不符合预期的结果。
        raise ValueError("top_k必须是正整数。")
    # 没有正向标准证据时，不能把“无法计算”误记为0分。
    if not evidences:
        # None在保存为JSON后会显示为null，表示该指标不适用。
        return {
            # 资料不足题不计算第一名的正向证据命中。
            "evidence_hit_at_1": None,
            # 资料不足题不计算前k名的正向证据命中。
            "evidence_hit_at_k": None,
            # 没有标准证据时，召回比例没有合法分母。
            "evidence_recall_at_k": None,
            # 没有证据目标，就没有完成它的排名。
            "evidence_rr_at_k": None,
            # 不使用all([])把资料不足题误判为所有证据齐全。
            "all_evidence_at_k": None,
            # 没有证据，自然也没有逐条覆盖状态。
            "covered_evidence": [],
            # None表示没有可报告的首次完成排名。
            "first_completion_rank": None,
            # 字符覆盖比例也留空。
            "character_coverage": [],
        }
    # 保留检索器给出的顺序；只评估指定的前k条，不再按相似度重新排序。
    selected = references[:top_k]
    # 没返回任何片段时，每条正向证据都尚未覆盖。
    covered_evidence = [False] * len(evidences)
    # 初始时每条证据的字符覆盖比例均为0。
    character_coverage = [0.0] * len(evidences)
    # 第一名是否完成了至少一条证据，初始设为0。
    hit_at_1 = 0.0
    # 记录首次完成任一条证据的排名；尚未完成时为None。
    first_completion_rank = None
    # 依次检查前1条、前2条……直到前k条，才能确定最早完成位置。
    for rank in range(1, len(selected) + 1):
        # 每个前缀重新计算全部证据的覆盖比例。
        character_coverage = []
        # 分别处理每条标准证据，避免把不同证据的分母混在一起。
        for evidence in evidences:
            # 只收集同一文件、同一页面的片段区间。
            intervals = []
            # selected[:rank]表示当前检查的前rank个检索结果。
            for reference in selected[:rank]:
                # 文件不同或PDF页面不同的片段不能拼接为同一条证据。
                if reference["source"] == evidence["source"] and reference.get("page") == evidence.get("page"):
                    # 保存该片段在原文中的左闭右开区间。
                    intervals.append((reference["start"], reference["end"]))
            # 合并当前片段的覆盖范围，重叠字符只计算一次。
            covered_chars = interval_coverage(evidence["start"], evidence["end"], intervals)
            # 用覆盖字符数除以证据长度，得到0到1之间的覆盖比例。
            character_coverage.append(covered_chars / (evidence["end"] - evidence["start"]))
        # 只有比例等于1，即整条标注证据都覆盖，才算该证据完成。
        covered_evidence = [coverage == 1.0 for coverage in character_coverage]
        # 单独保存只看第一条检索结果时的命中情况。
        if rank == 1:
            # any表示至少一个为True，float把布尔值转成0.0或1.0。
            hit_at_1 = float(any(covered_evidence))
        # 只记录第一次完成证据的位置，后续前缀不覆盖这个结果。
        if first_completion_rank is None and any(covered_evidence):
            # rank从1开始，可以直接用于计算倒数排名。
            first_completion_rank = rank
    # 将各项结果合并成字典，供报告生成和后续保存使用。
    return {
        # 前1条是否完整覆盖至少一条标注证据。
        "evidence_hit_at_1": hit_at_1,
        # 前k条是否完整覆盖至少一条标注证据。
        "evidence_hit_at_k": float(any(covered_evidence)),
        # 布尔值求和得到完成条数，再除以全部标注证据条数。
        "evidence_recall_at_k": sum(covered_evidence) / len(evidences),
        # 首次完成排名的倒数；完全未完成任何证据时记为0。
        "evidence_rr_at_k": 1.0 / first_completion_rank if first_completion_rank is not None else 0.0,
        # all表示全部标注证据都完整覆盖，转为0.0或1.0。
        "all_evidence_at_k": float(all(covered_evidence)),
        # 保留每条证据是否完成，帮助解释多证据题漏掉了哪一条。
        "covered_evidence": covered_evidence,
        # 保留实际排名，方便读者核对倒数排名的计算。
        "first_completion_rank": first_completion_rank,
        # 保留部分覆盖程度，但它不等于语义事实覆盖率。
        "character_coverage": character_coverage,
    }
