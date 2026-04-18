import os

# ================= 配置区域 =================
INPUT_DIR = "./zhihu"  # 原始数据文件夹
OUTPUT_DIR = "./zhihu_1m"    # 输出文件夹

# 根据你提供的 ZhihuRec-1M 统计量定义的目标行数
# 注意：这里假设 CSV 第一行是表头，数据行数 = 统计量 + 1 (表头)
TARGET_LINES = {
    "inter_impression.csv": 999970 ,  # impressions
    "inter_query.csv": 38422,         # queries
    "info_user.csv": 7974 ,             # users
    "info_answer.csv": 81563,           # answers
    "info_question.csv": 29340,        # questions
    "info_author.csv": 47888 ,          # authors
    "info_topic.csv": 22897,           # topics     这里topics应该只有14158，具体可以用zhihu_small.py验证
    "info_token.csv": 249586           # tokens
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 处理逻辑 =================

def extract_first_n_lines(in_path, out_path, n_lines):
    """
    高效提取文件前 n 行，不使用 pandas，内存占用极低
    """
    print(f"Processing {os.path.basename(in_path)}...")
    try:
        with open(in_path, 'r', encoding='utf-8') as f_in, \
             open(out_path, 'w', encoding='utf-8') as f_out:
            
            for i, line in enumerate(f_in):
                if i >= n_lines:
                    break
                f_out.write(line)
    except FileNotFoundError:
        print(f"Warning: {in_path} not found, skipping.")

# 遍历执行
for filename, max_rows in TARGET_LINES.items():
    src = os.path.join(INPUT_DIR, filename)
    dst = os.path.join(OUTPUT_DIR, filename)
    extract_first_n_lines(src, dst, max_rows)

print("Done!")