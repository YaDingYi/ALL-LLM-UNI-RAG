'''把旧版 JSON 向量索引复制到本地 Chroma，不删除或修改旧文件。'''

import sys

from chroma_index import import_index
from index_demo import list_indexes, load_index


def migrate_json_indexes():
    '''逐份迁移旧索引，返回成功、复用和失败的数量。'''
    # 只读取旧索引目录中符合命名规则的文件。
    filenames = list_indexes()
    if not filenames:
        print("没有找到旧版 JSON 索引，无需迁移。")
        return {"migrated": 0, "reused": 0, "failed": 0}

    counts = {"migrated": 0, "reused": 0, "failed": 0}
    # 旧列表按新到旧排列；先导入旧快照，让最新快照最后完成并排在最前。
    for filename in reversed(filenames):
        try:
            # 原文件只用于读取；import_index 验证兼容性，并复用已有相同快照。
            index_data = load_index(filename)
            result = import_index(index_data, filename)
            if result["already_exists"]:
                counts["reused"] += 1
                print(f"已复用：{filename} → {result['name']}")
            else:
                counts["migrated"] += 1
                print(f"已迁移：{filename} → {result['name']}（{result['chunk_count']} 个片段）")
        except Exception as error:
            # 只报告错误类型，不输出资料原文；其他索引继续迁移。
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
