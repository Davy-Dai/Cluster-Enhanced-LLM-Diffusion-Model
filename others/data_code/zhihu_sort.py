import os
import csv
from collections import defaultdict

# -------------------------- 核心配置（请修改这里的路径） --------------------------
INPUT_FOLDER = "zhihu_small"   # 你的原始数据文件夹（8个csv都在这里）
OUTPUT_FOLDER = "zhihu_sort" # 重排后的输出文件夹（自动创建）
# -----------------------------------------------------------------------------------

def build_id_mapping(input_dir, filename, id_col=0, debug_print=True):
    """
    读取info主键文件，建立 旧ID -> 新ID 的映射
    新ID = 文件行号（从0开始连续递增），严格按原始文件顺序分配
    """
    mapping = {}
    file_path = os.path.join(input_dir, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}，请检查输入路径")
    
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for new_id, row in enumerate(reader):
            if not row:
                continue
            # 统一转为字符串，避免数字/字符串类型不匹配导致映射失效
            old_id = str(row[id_col]).strip()
            mapping[old_id] = str(new_id)
    
    # 调试：打印前5个映射，确认映射建立成功
    if debug_print:
        print(f"【{filename}】ID映射建立完成，总数：{len(mapping)}")
        print(f"  前5个映射示例：{dict(list(mapping.items())[:5])}")
    
    return mapping

def process_primary_key_file(input_path, output_path, id_col=0, extra_processors=None, stats_dict=None):
    """
    处理主键info文件：强制重写主键列为新的连续ID，同时处理其他关联字段
    """
    extra_processors = extra_processors or []
    count = 0
    filename = os.path.basename(input_path)
    
    with open(input_path, 'r', encoding='utf-8', newline='') as in_f, \
         open(output_path, 'w', encoding='utf-8', newline='') as out_f:
        
        reader = csv.reader(in_f)
        writer = csv.writer(out_f)
        
        for new_id, row in enumerate(reader):
            count += 1
            # 核心：主键列强制替换为新ID（行号）
            if id_col < len(row):
                row[id_col] = str(new_id)
            # 处理其他关联字段
            for col_idx, func in extra_processors:
                row = func(row, col_idx)
            writer.writerow(row)
    
    if stats_dict is not None:
        stats_dict[filename] = count
    print(f"  ✓ {filename} 主键重排完成，总行数：{count:,}")
    return count

def process_single_id_field(row, col_idx, mapping):
    """处理单个ID字段（比如user_id、answer_id）"""
    if col_idx >= len(row):
        return row
    old_id = str(row[col_idx]).strip()
    # 只有在映射里存在的ID才替换，不存在的保留原值（兼容空值/异常值）
    if old_id in mapping:
        row[col_idx] = mapping[old_id]
    return row

def process_list_id_field(row, col_idx, mapping):
    """处理空格分隔的ID列表字段（比如token_ids、topic_ids）"""
    if col_idx >= len(row):
        return row
    id_str = str(row[col_idx]).strip()
    if not id_str:
        return row
    old_ids = id_str.split(' ')
    # 逐个替换，不存在的ID保留原值
    new_ids = [mapping.get(str(oid).strip(), oid) for oid in old_ids]
    row[col_idx] = ' '.join(new_ids)
    return row

def process_associated_file(input_path, output_path, processors, stats_dict):
    """处理关联文件（交互表、非主键字段）"""
    count = 0
    filename = os.path.basename(input_path)
    
    with open(input_path, 'r', encoding='utf-8', newline='') as in_f, \
         open(output_path, 'w', encoding='utf-8', newline='') as out_f:
        
        reader = csv.reader(in_f)
        writer = csv.writer(out_f)
        
        for row in reader:
            count += 1
            for col_idx, func in processors:
                row = func(row, col_idx)
            writer.writerow(row)
    
    stats_dict[filename] = count
    print(f"  ✓ {filename} 关联ID替换完成，总行数：{count:,}")
    return count

def main():
    # 1. 初始化
    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(f"输入文件夹不存在：{INPUT_FOLDER}")
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    stats = {}
    print("="*60)
    print(f"开始处理ZhihuRec数据集ID重排")
    print(f"原始数据路径：{INPUT_FOLDER}")
    print(f"输出数据路径：{OUTPUT_FOLDER}")
    print("="*60 + "\n")

    # -------------------------------------------------------------------------
    # 第一步：优先建立所有主键ID的映射（严格按info文件的行顺序）
    # -------------------------------------------------------------------------
    print("【1/2 建立ID映射关系】")
    user_map = build_id_mapping(INPUT_FOLDER, 'info_user.csv', id_col=0)
    answer_map = build_id_mapping(INPUT_FOLDER, 'info_answer.csv', id_col=0)
    question_map = build_id_mapping(INPUT_FOLDER, 'info_question.csv', id_col=0)
    author_map = build_id_mapping(INPUT_FOLDER, 'info_author.csv', id_col=0)
    topic_map = build_id_mapping(INPUT_FOLDER, 'info_topic.csv', id_col=0)
    token_map = build_id_mapping(INPUT_FOLDER, 'info_token.csv', id_col=0)
    print("-"*60 + "\n")

    # -------------------------------------------------------------------------
    # 第二步：处理所有文件（先处理主键文件，再处理关联文件）
    # -------------------------------------------------------------------------
    print("【2/2 处理文件并重写ID】")

    # -------------------------- 1. 处理主键info文件（核心重排） --------------------------
    # info_user.csv：主键user_id(0)，关联字段topic_ids(26)
    user_extra = [
        (26, lambda r, c: process_list_id_field(r, c, topic_map))
    ]
    process_primary_key_file(
        os.path.join(INPUT_FOLDER, 'info_user.csv'),
        os.path.join(OUTPUT_FOLDER, 'info_user.csv'),
        id_col=0, extra_processors=user_extra, stats_dict=stats
    )

    # info_answer.csv：主键answer_id(0)，关联字段question_id(1)、author_id(3)、token_ids(16)、topic_ids(17)
    answer_extra = [
        (1, lambda r, c: process_single_id_field(r, c, question_map)),
        (3, lambda r, c: process_single_id_field(r, c, author_map)),
        (16, lambda r, c: process_list_id_field(r, c, token_map)),
        (17, lambda r, c: process_list_id_field(r, c, topic_map)),
    ]
    process_primary_key_file(
        os.path.join(INPUT_FOLDER, 'info_answer.csv'),
        os.path.join(OUTPUT_FOLDER, 'info_answer.csv'),
        id_col=0, extra_processors=answer_extra, stats_dict=stats
    )

    # info_question.csv：主键question_id(0)，关联字段token_ids(6)、topic_ids(7)
    question_extra = [
        (6, lambda r, c: process_list_id_field(r, c, token_map)),
        (7, lambda r, c: process_list_id_field(r, c, topic_map)),
    ]
    process_primary_key_file(
        os.path.join(INPUT_FOLDER, 'info_question.csv'),
        os.path.join(OUTPUT_FOLDER, 'info_question.csv'),
        id_col=0, extra_processors=question_extra, stats_dict=stats
    )

    # info_author.csv：主键author_id(0)，无额外关联字段
    process_primary_key_file(
        os.path.join(INPUT_FOLDER, 'info_author.csv'),
        os.path.join(OUTPUT_FOLDER, 'info_author.csv'),
        id_col=0, extra_processors=[], stats_dict=stats
    )

    # info_topic.csv：主键topic_id(0)，无额外关联字段
    process_primary_key_file(
        os.path.join(INPUT_FOLDER, 'info_topic.csv'),
        os.path.join(OUTPUT_FOLDER, 'info_topic.csv'),
        id_col=0, extra_processors=[], stats_dict=stats
    )

    # info_token.csv：主键token_id(0)，无额外关联字段
    process_primary_key_file(
        os.path.join(INPUT_FOLDER, 'info_token.csv'),
        os.path.join(OUTPUT_FOLDER, 'info_token.csv'),
        id_col=0, extra_processors=[], stats_dict=stats
    )

    # -------------------------- 2. 处理关联交互文件 --------------------------
    # inter_impression.csv：user_id(0)、answer_id(1)
    imp_processors = [
        (0, lambda r, c: process_single_id_field(r, c, user_map)),
        (1, lambda r, c: process_single_id_field(r, c, answer_map)),
    ]
    process_associated_file(
        os.path.join(INPUT_FOLDER, 'inter_impression.csv'),
        os.path.join(OUTPUT_FOLDER, 'inter_impression.csv'),
        imp_processors, stats
    )

    # inter_query.csv：user_id(0)、token_ids(1)
    query_processors = [
        (0, lambda r, c: process_single_id_field(r, c, user_map)),
        (1, lambda r, c: process_list_id_field(r, c, token_map)),
    ]
    process_associated_file(
        os.path.join(INPUT_FOLDER, 'inter_query.csv'),
        os.path.join(OUTPUT_FOLDER, 'inter_query.csv'),
        query_processors, stats
    )

    # -------------------------------------------------------------------------
    # 第三步：输出最终统计
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("✅ 所有文件处理完成！最终文件行数统计：")
    print("="*60)
    for filename, count in sorted(stats.items()):
        print(f"{filename:<22} | 行数：{count:,}")
    print("="*60)
    print(f"重排后的文件已全部保存至：{os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()