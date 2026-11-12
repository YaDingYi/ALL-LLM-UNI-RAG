'''第二阶段：复用当前检索器，离线检查5道题的标注原文覆盖情况。

不调用DeepSeek，不使用参考答案改写问题，不创建或更新Chroma集合。
代码行上方解释其作用；完整指标定义和运行方法见STEP2.md。
'''

# 解析终端参数，例如指定要评估的集合。
import argparse
# 计算题库和原始资料的内容摘要。
import hashlib
# 读取题库、写入可复查的结果。
import json
# 为直接运行本文件配置项目模块的查找路径。
import sys
# 生成报告时间。
from datetime import datetime
# 记录实际安装的依赖版本。
from importlib.metadata import version
# 使用不依赖当前工作目录的文件路径。
from pathlib import Path
# 计算已完成请求的耗时中位数。
from statistics import median
# 使用单调时钟测量耗时，不受系统时间校准影响。
from time import perf_counter
# 给每次报告添加随机后缀，避免覆盖旧结果。
from uuid import uuid4

# __file__指向当前脚本，父目录是evaluation，再上一级是项目根目录。
PROJECT_DIR = Path(__file__).resolve().parent.parent
# 将项目根目录放入模块查找路径，支持直接点击PyCharm运行。
sys.path.insert(0, str(PROJECT_DIR))

# 复用项目已经采用的数据库检查和检索逻辑，不导入会执行网页的app.py。
from chroma_index import CHROMA_DIR, _get_ready_collection, list_indexes, search_index
# 读取本地向量配置，并在正式计时前预热本地模型。
from embedding_demo import MODEL_NAME, QUERY_PREFIX, embed_texts
# 导入独立的区间覆盖计算和评分函数。
from evaluation.retrieval_metrics import interval_coverage, score_retrieval


def load_dataset(path):
    '''读取题库、检查原文件版本，并将引用定位为字符区间。'''
    # 读取UTF-8 JSON；非法JSON会产生异常，不继续评分。
    dataset = json.loads(path.read_text(encoding="utf-8"))
    # 从题库位置出发解析../资料，而不是从终端工作目录出发。
    source_root = (path.parent / dataset["source_root"]).resolve()
    # 保存已核对的原文件全文，随后用于校验数据库片段。
    texts = {}
    # 逐个处理题库记录的原文版本。
    for item in dataset["source_versions"]:
        # 按相对来源路径找到原文件。
        source_path = source_root / item["source"]
        # 摘要按文件的原始字节计算。
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        # 原文变化时停止，避免拿旧答案评估新资料。
        if digest != item["sha256"]:
            # 错误信息指出需要重新核对的资料。
            raise ValueError(f"原文版本变化，请先复核题库：{item['source']}")
        # 与项目文档读取函数使用相同的编码规则。
        texts[item["source"]] = source_path.read_text(encoding="utf-8-sig")
    # 保存每一道题的证据位置；题库原文件保持不变。
    mapped = {}
    # 逐题定位证据。
    for case in dataset["cases"]:
        # 重复题号会导致结果覆盖，因此必须拒绝。
        if case["id"] in mapped:
            # 说明冲突的题号。
            raise ValueError(f"题号重复：{case['id']}")
        # 有答案题必须有依据，无答案题的依据应为空。
        if case["answerable"] != bool(case["gold_evidence"]):
            # 这类标注矛盾不能自动修正。
            raise ValueError(f"可回答性与证据不一致：{case['id']}")
        # 创建当前题的证据列表。
        mapped[case["id"]] = []
        # 一道题可以包含多条原文证据。
        for evidence in case["gold_evidence"]:
            # 本版题库使用Markdown行号，PDF需要另行实现页内定位。
            if evidence["page"] is not None:
                # 明确拒绝尚未支持的证据格式。
                raise ValueError("本版仅支持Markdown证据，PDF证据需要单独定位。")
            # 保留换行符，才能计算与原文件一致的字符偏移。
            lines = texts[evidence["source"]].splitlines(keepends=True)
            # Python切片会容忍越界，所以要先检查标注行号是否合法。
            if not 1 <= evidence["line_start"] <= evidence["line_end"] <= len(lines):
                # 不允许错位的行号碰巧仍包含引文而蒙混过关。
                raise ValueError(f"证据行号越界：{case['id']}")
            # 行号从1开始，切片下标从0开始；结束行也包含在内。
            block = "".join(lines[evidence["line_start"] - 1:evidence["line_end"]])
            # 要求在标注行范围内能唯一定位原文，而不是模糊匹配。
            if not evidence["quote"] or block.count(evidence["quote"]) != 1:
                # 引用或行号不准确时停止。
                raise ValueError(f"引用无法唯一定位：{case['id']}")
            # 起点等于前面所有行长度之和，加上引用在当前行块中的位置。
            start = sum(len(line) for line in lines[:evidence["line_start"] - 1]) + block.index(evidence["quote"])
            # 复制原标注并添加左闭右开的[start,end)，用于覆盖计算。
            mapped[case["id"]].append({**evidence, "start": start, "end": start + len(evidence["quote"])})
    # 同时返回题库、核对过的全文，以及证据位置。
    return dataset, texts, mapped


def audit_collection(collection, texts):
    '''核对两份教材的全部入库片段及全文覆盖，避免评估旧版或缺失资料。'''
    # 汇总核对过的数据库记录，便于给证据附上重叠片段ID。
    records = []
    # 只读取题库引用的原文件；正式检索仍然使用整个集合。
    for source, text in texts.items():
        # 明确读取已存文本和元数据，不重新生成资料向量。
        stored = collection.get(where={"source": source}, include=["documents", "metadatas"])
        # 收集本文件所有片段的字符范围。
        intervals = []
        # Chroma返回的三个数组按相同位置对应。
        for identifier, document, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"]):
            # 取出建库时保存的位置。
            start, end = metadata["start"], metadata["end"]
            # Markdown片段应无页码，范围合法，文本必须和原文完全一致。
            if metadata.get("page") is not None or not 0 <= start < end <= len(text) or document != text[start:end]:
                # 发现版本或位置问题就停止，不能悄悄将其当成检索没命中。
                raise ValueError(f"库中片段与题库原文不一致：{source} / {identifier}")
            # 记录区间，用并集检查是否漏掉文件中的任何字符。
            intervals.append((start, end))
            # 保存位置和来源，不复制整个资料库。
            records.append({"id": identifier, **metadata, "page": None})
        # 区间并集必须覆盖完整原文件，包括末尾新增内容。
        if interval_coverage(0, len(text), intervals) != len(text):
            # 缺失文件或不完整入库都不能作为有效基线。
            raise ValueError(f"集合未完整包含题库对应原文：{source}")
    # 返回核对结果，供报告和证据定位使用。
    return records


def attach_positions(collection, references):
    '''给实际检索结果补上start/end，保持原有检索排名不变。'''
    # 空结果无需再访问数据库。
    if not references:
        # 返回空列表，由评分函数计算未命中。
        return []
    # 按检索到的ID读取元数据；get的返回顺序不保证等于检索顺序。
    stored = collection.get(ids=[item["id"] for item in references], include=["documents", "metadatas"])
    # 使用ID建立映射，防止把位置附到错误的结果上。
    lookup = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))
    # 保存补齐后的有序结果。
    enriched = []
    # 按原始query返回顺序处理。
    for reference in references:
        # 从映射中取出对应的原文和元数据。
        document, metadata = lookup[reference["id"]]
        # 检查两次读取是否一致，避免数据库变化后混用结果。
        if document != reference["text"] or metadata["source"] != reference["source"] or metadata.get("page") != reference["page"]:
            # 不一致时终止这一题的评分。
            raise ValueError("检索与位置读取不一致，请固定集合后重试。")
        # 添加位置；原来的文本、排名、来源和相似度都保留。
        enriched.append({**reference, "start": metadata["start"], "end": metadata["end"]})
    # 返回顺序不变的结果。
    return enriched


def summarize(rows):
    '''只用有答案的题求指标均值；检索失败计0，并另外报告失败数量。'''
    # 无答案题没有正向证据，不加入覆盖指标分母。
    answerable = [row for row in rows if row["answerable"]]
    # 记录有答案题和实际完成的题数。
    summary = {
        # 正向指标只用这些题作为分母。
        "answerable_count": len(answerable),
        # 实际完成检索和评分的题数，包括资料不足题。
        "completed_count": sum(row["status"] == "ok" for row in rows),
        # 总运行题数，用于报告运行完成率。
        "case_count": len(rows),
    }
    # 这些都是针对标注原文的指标，不是语义相关片段的传统Recall/MRR。
    keys = ("evidence_hit_at_1", "evidence_hit_at_k", "evidence_recall_at_k", "evidence_rr_at_k", "all_evidence_at_k")
    # 分别汇总五个维度，不把它们加成一个总分。
    for key in keys:
        # 失败题缺少评分时按0计入，避免删掉失败题而抬高平均分。
        summary[key] = sum((row["metrics"] or {}).get(key, 0) or 0 for row in answerable) / len(answerable) if answerable else None
    # 只对已完成检索的耗时计算中位数；失败耗时仍保存在逐题结果中。
    completed_times = [row["retrieval_seconds"] for row in rows if row["status"] == "ok"]
    # 无成功请求时不虚构一个耗时。
    summary["median_retrieval_seconds"] = median(completed_times) if completed_times else None
    # 明确异常和分母的处理约定。
    summary["policy"] = "无答案题不计覆盖指标；检索失败的有答案题计0；预检失败不产生评分。"
    # 返回汇总结果。
    return summary


def markdown_report(report):
    '''将JSON结果整理为便于学习和逐题核查的Markdown报告。'''
    # 报告首先声明范围，防止把检索覆盖理解成回答准确率。
    lines = ["# 离线检索评估结果", "", "本报告只测标注原文覆盖，不调用DeepSeek；所有相似度都不是正确概率。", "", f"- 集合：`{report.get('collection', '未选择')}`", f"- 状态：{report['status']}", f"- Top-k：{report['top_k']}", ""]
    # 预检失败时保留错误，但不展示伪造的零分。
    if "error" in report:
        # 将失败原因写入报告。
        lines.extend([f"预检或运行失败：{report['error']}", ""])
    # 已完成评测时加入汇总。
    if "summary" in report:
        # 用JSON代码块完整保留数值和分母约定。
        lines.extend(["## 汇总", "", "```json", json.dumps(report["summary"], ensure_ascii=False, indent=2), "```", ""])
    # 每题都显示实际结果，包括资料不足题。
    for row in report["cases"]:
        # 显示问题、状态、耗时和本题评分。
        lines.extend([f"## {row['id']} · {row['category']}", "", row["question"], "", f"状态：{row['status']}；检索耗时：{row['retrieval_seconds']:.4f}秒。", "", "```json", json.dumps(row["metrics"], ensure_ascii=False, indent=2), "```", ""])
        # 单题异常也保留，不略过失败问题。
        if row.get("error"):
            # 添加可诊断的本地异常信息。
            lines.extend([f"错误：{row['error']}", ""])
        # 给每条标准证据编号，位置采用左闭右开区间。
        for number, evidence in enumerate(row["mapped_evidence"], start=1):
            # 显示原文位置，帮助和检索片段逐项对照。
            lines.extend([f"标准证据{number}：`{evidence['source']}`，字符[{evidence['start']}, {evidence['end']})。", ""])
        # 检索排名从1开始，保持query原有顺序。
        for rank, item in enumerate(row["references"], start=1):
            # 展示返回的片段ID、来源及分数。
            lines.extend([f"### 排名{rank} · {item['id']} · 相似度{item['score']:.4f}", "", f"来源：`{item['source']}`；页码：{item['page']}；字符[{item['start']}, {item['end']})。", ""])
            # 使用引用块显示原文，每行加前缀，不把资料当作Markdown指令执行。
            lines.extend("> " + line for line in item["text"].splitlines())
            # 空行将不同片段分开。
            lines.append("")
    # 给出解释边界，完整方法另见STEP2。
    lines.extend(["## 解释边界", "", "EvidenceRecall按完整原文摘录计数，不是所有相关片段的召回率；等价证据和部分足够回答的摘录可能被保守漏计。EvidenceRR允许多片段共同完成证据，不是传统MRR。无答案题的指标为null，不能据此判断DeepSeek是否会拒答。多轮题沿用当前网页的question-only检索。", ""])
    # 拼接为一个可写入文件的字符串。
    return "\n".join(lines)


def main():
    '''选择集合、预检、运行5题，并保存独立报告。'''
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(description=__doc__)
    # 不指定时采用项目的示范题库。
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("demo_questions.json"))
    # 可以显式固定集合；省略时使用list_indexes返回的最新可用集合。
    parser.add_argument("--collection", help="固定Chroma集合名称；默认选择最新可用集合")
    # 默认与当前网页一致，只检索3条。
    parser.add_argument("--top-k", type=int, default=3)
    # 报告默认保存在evaluation/runs，不进入资料或Histor。
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("runs"))
    # 读取终端传入的实际参数。
    args = parser.parse_args()
    # 返回数量必须是正整数。
    if args.top_k < 1:
        # 用命令行错误结束程序。
        parser.error("--top-k必须大于0")
    # 每次运行使用新名字，保存时间和随机后缀。
    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    # 提前建立报告对象，使预检失败也能留下记录。
    report = {
        # 保存本次运行的唯一标识。
        "run_id": run_id,
        # 时间包含本机时区偏移，避免跨时区混淆。
        "created_at": datetime.now().astimezone().isoformat(),
        # 初始阶段尚未执行任何检索。
        "status": "preflight",
        # 明确报告中的k究竟是多少。
        "top_k": args.top_k,
        # 稍后逐题追加结果。
        "cases": [],
        # 先记录用户指定的集合；默认选择将在预检时补齐。
        "collection": args.collection,
    }
    # 预检及运行统一捕获异常，保留失败报告。
    try:
        # 数据库不存在时停止，避免连接函数自动创建空库。
        if not (CHROMA_DIR / "chroma.sqlite3").is_file():
            # 提示先在网页中建立知识库。
            raise FileNotFoundError("项目中没有现成Chroma数据库，请先建立知识库。")
        # 获取写入完成的集合，顺序由新到旧。
        indexes = list_indexes()
        # 空列表不能用于默认选择。
        if not indexes:
            # 保留清楚的失败原因。
            raise ValueError("没有可评估的完整知识库集合。")
        # 优先使用用户指定值，否则明确采用最新集合。
        name = args.collection or indexes[0]
        # 无论后续是否通过预检，都记录实际选择。
        report["collection"] = name
        # 复用项目的模型、维度、距离类型和记录数检查。
        collection, count = _get_ready_collection(name)
        # 核对题库、原文件摘要和逐字引用。
        dataset, texts, mapped = load_dataset(args.dataset.resolve())
        # 再核对数据库里存的两份教材是否与原文一致。
        records = audit_collection(collection, texts)
        # 将每条标准摘录映射到有字符交集的片段，供复查而不是当作穷尽相关性标签。
        for evidences in mapped.values():
            # 分别处理每一条摘录。
            for evidence in evidences:
                # 严格使用同来源、同页、非空字符交集；相邻但不重叠不算。
                evidence["overlapping_chunk_ids"] = [
                    # 保存片段ID，而不是把这些片段直接提供给检索器。
                    record["id"]
                    # 检查已核对的全部教材片段。
                    for record in records
                    # 相同文件中的坐标才可比较。
                    if record["source"] == evidence["source"]
                    # 页码也必须一致；本版都是None。
                    and record["page"] == evidence["page"]
                    # 交集起点必须严格小于交集终点，才有实际字符交集。
                    and max(record["start"], evidence["start"]) < min(record["end"], evidence["end"])
                ]
        # 记录复现所需的集合、题库、模型和依赖信息。
        report.update({
            # 记录题库实际位置。
            "dataset_path": str(args.dataset.resolve()),
            # 记录题库文件摘要，不只依赖人为填写的版本号。
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            # 保留便于人阅读的题库版本。
            "dataset_version": dataset["version"],
            # 保存本次已核对的原文版本。
            "source_versions": dataset["source_versions"],
            # 保存集合自己的模型、维度、创建时间和文件数量等信息。
            "collection_metadata": collection.metadata,
            # 保存距离类型及HNSW配置。
            "collection_configuration": collection.configuration,
            # 正式检索在整个集合中进行，而不是只查两份教材。
            "collection_count": count,
            # 记录经过全文一致性核对的教材片段数量。
            "audited_chunk_count": len(records),
            # 保存实际使用的本地嵌入模型名称。
            "embedding_model": MODEL_NAME,
            # 保存检索函数会加到问题前面的提示文字。
            "query_prefix": QUERY_PREFIX,
            # 声明没有改变现有网页的检索输入策略。
            "query_policy": "question_only; no source filter; no reference answer",
            # 保存Python版本。
            "python": sys.version.split()[0],
            # 保存关键依赖的安装版本，方便复查环境差异。
            "packages": {package: version(package) for package in ("chromadb", "fastembed", "onnxruntime")},
        })
        # 输出明确的集合与规模，便于用户发现选择错误。
        print(f"集合：{name}；全部{count}个片段；题库来源已核对{len(records)}个片段。", flush=True)
        # 单独计时预热，避免第一道题独自承担模型加载耗时。
        warmup_started = perf_counter()
        # 仅在本机生成一个预热向量，不调用任何远程模型。
        embed_texts(["离线检索评估预热"])
        # 记录加载和首次推理成本，与后面的热检索分开。
        report["warmup_seconds"] = perf_counter() - warmup_started
        # 使用题库原有顺序运行，题目之间没有共享对话。
        for case in dataset["cases"]:
            # 复制题目以保存完整标注，但只有question会传给检索器。
            row = {**case, "mapped_evidence": mapped[case["id"]], "status": "error", "references": [], "metrics": None}
            # 记录单题检索开始时间。
            started = perf_counter()
            # 单题失败不阻止后续题运行。
            try:
                # 和网页一致：检索当前问题；不拼接history，也不传source_scope或标准答案。
                references, _ = search_index(case["question"], name, args.top_k)
                # 只统计现有检索函数耗时，不把位置核对和评分计入检索耗时。
                row["retrieval_seconds"] = perf_counter() - started
                # 补充位置而不改变结果顺序。
                row["references"] = attach_positions(collection, references)
                # 使用严格原文覆盖规则计算本题分数。
                row["metrics"] = score_retrieval(row["mapped_evidence"], row["references"], args.top_k)
                # 完成查询、核对和评分后标记成功。
                row["status"] = "ok"
            # 异常会单独记录，不把失败题从均值分母移除。
            except Exception as error:
                # 计时若已完成则保留，否则记到异常发生时。
                row.setdefault("retrieval_seconds", perf_counter() - started)
                # 这里只涉及本地检索，不读取或输出API Key。
                row["error"] = f"{type(error).__name__}: {error}"
            # 不论成功与否，都把本题保存到报告。
            report["cases"].append(row)
            # 即时显示进度，避免运行时没有反馈。
            print(f"{case['id']}：{row['status']}，{row['retrieval_seconds']:.4f}秒，{row['metrics']}", flush=True)
        # 对全部题汇总，明确正向题分母和失败情况。
        report["summary"] = summarize(report["cases"])
        # 某题失败时整体状态反映部分失败。
        report["status"] = "completed" if all(row["status"] == "ok" for row in report["cases"]) else "partial_failure"
    # 资料版本或模型加载等问题会导致整体评估无效。
    except Exception as error:
        # 不给未完成预检的运行生成指标。
        report["status"] = "failed"
        # 保存具体的本地失败原因。
        report["error"] = f"{type(error).__name__}: {error}"
    # 创建独立的结果目录，不改写知识库。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 构造同一轮JSON和Markdown报告的文件名前缀。
    destination = args.output_dir / run_id
    # x模式只允许创建新文件，防止覆盖已有报告。
    with destination.with_suffix(".json").open("x", encoding="utf-8") as handle:
        # 保留中文和缩进，禁止写入非标准JSON中的NaN值。
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    # 同样以只新建模式保存可阅读报告。
    with destination.with_suffix(".md").open("x", encoding="utf-8") as handle:
        # 从同一个report生成，避免两种报告出现不同结果。
        handle.write(markdown_report(report))
    # 告诉用户结果的实际绝对路径。
    print(f"报告：{destination.with_suffix('.json').resolve()}\n说明：{destination.with_suffix('.md').resolve()}")
    # 有失败时显示原因或提示查看逐题错误。
    if report["status"] != "completed":
        # 控制台同样保留异常提示。
        print(report.get("error", "存在失败题，请检查报告。"))
    # 成功退出码为0，其余为1，便于后续自动运行。
    return 0 if report["status"] == "completed" else 1


# 只有直接运行文件时才执行评估，导入函数不会启动检索。
if __name__ == "__main__":
    # 将main返回的结果传给操作系统。
    raise SystemExit(main())
