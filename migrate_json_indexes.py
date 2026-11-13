'''仅在 Chroma 空库时导入最新的旧版 JSON 索引，保留原始文件。'''

import sys

from chroma_index import import_index, list_indexes as list_chroma_indexes
from index_demo import list_indexes as list_json_indexes, load_index
from knowledge_upload import knowledge_update_lock


def migrate_json_indexes():
    '''仅迁移最新的一份旧索引，避免覆盖现有知识库或积累历史集合。'''
    # 只读取旧索引目录中符合命名规则的文件。
    filenames = list_json_indexes()
    if not filenames:
        print("没有找到旧版 JSON 索引，无需迁移。")
        return {"migrated": 0, "reused": 0, "failed": 0}

    counts = {"migrated": 0, "reused": 0, "failed": 0}
    # 检查和导入共用建库锁，避免其他进程在检查后并发创建集合。
    with knowledge_update_lock():
        if list_chroma_indexes():
            # 旧JSON可能不包含现代知识库的新资料，因此不覆盖已有集合。
            print("已有可用的 Chroma 知识库；为保留当前资料，停止迁移。")
            counts["failed"] = 1
            return counts
        # 旧索引按文件名从新到旧排列；只导入最新一份，不回退到更旧的版本。
        filename = filenames[0]
        try:
            # 原文件只用于读取；import_index 会验证兼容性并保留原JSON。
            index_data = load_index(filename)
            result = import_index(index_data, filename)
            if result["already_exists"]:
                counts["reused"] += 1
                print(f"已复用：{filename} → {result['name']}")
            else:
                counts["migrated"] += 1
                print(f"已迁移：{filename} → {result['name']}（{result['chunk_count']} 个片段）")
        except Exception as error:
            # 最新索引失败时停止，不静默回退并导入不完整的更旧版本。
            counts["failed"] += 1
            print(f"迁移失败：{filename}（{type(error).__name__}）")

    print(f"迁移结束：新迁移 {counts['migrated']} 份，复用 {counts['reused']} 份，失败 {counts['failed']} 份。")
    return counts


if __name__ == "__main__":
    try:
        result = migrate_json_indexes()
    except Exception as error:
        # 无法列出索引目录等整体错误应返回非零状态。
        print(f"无法完成迁移（{type(error).__name__}），请检查旧索引目录和运行环境。")
        sys.exit(1)
    sys.exit(1 if result["failed"] else 0)
