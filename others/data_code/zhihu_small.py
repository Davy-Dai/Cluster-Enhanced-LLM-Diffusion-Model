import os
import pandas as pd
from tqdm import tqdm

# ================= 超参数设置 =================
# 请根据需要修改以下参数
A = 20                # 保留 impression 数量前 A 个的 answer
TARGET_RATIO =1/2.71         # 目标 clicks : nonclicks 比例
MAX_USERS_TO_REMOVE = 5000  # 最多淘汰的用户数量
INPUT_DIR = 'zhihu_1m'     # 原始数据文件夹
OUTPUT_DIR = 'zhihu_small'  # 输出数据文件夹
MIN_USER_PER_ANSWER = 20  # answer涉及的最小用户数
MAX_USER_PER_ANSWER = 200  # answer涉及的最大用户数
# ===========================================

def main():
    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    
    # ================= 步骤 1 & 2: 读取交互数据并筛选 Top A Answers =================
    print("\n[1/8] Reading interaction data...")
    inter_imp_path = os.path.join(INPUT_DIR, 'inter_impression.csv')
    # 读取时指定 dtype 以节省内存并避免警告
    dtype_imp = {0: 'int32', 1: 'int32', 2: 'int64', 3: 'int64'}
    df_imp = pd.read_csv(inter_imp_path, header=None, dtype=dtype_imp)
    df_imp.columns = ['user_id', 'answer_id', 'imp_ts', 'click_ts']
    
    # 选出 impression 数前 A 个的 answer
    print(f"Selecting top {A} answers by impressions...")
    top_answers = df_imp['answer_id'].value_counts().nlargest(A).index
    df_imp_filtered = df_imp[df_imp['answer_id'].isin(top_answers)].copy()
    
    # 释放内存
    del df_imp

    # ================= 步骤 3: 统计初始 clicks/nonclicks 并淘汰用户 =================
    print("\n[2/8] Calculating initial statistics and filtering users...")
    
    # 统计每个用户的 clicks, nonclicks 和 impressions
    df_imp_filtered['is_click'] = df_imp_filtered['click_ts'] != 0
    user_stats = df_imp_filtered.groupby('user_id').agg(
        clicks=('is_click', 'sum'),
        impressions=('is_click', 'count') # 总曝光数 = click + nonclick
    ).reset_index()
    
    user_stats['nonclicks'] = user_stats['impressions'] - user_stats['clicks']
    
    # 计算全局初始值
    total_clicks = user_stats['clicks'].sum()
    total_nonclicks = user_stats['nonclicks'].sum()
    total_impressions = total_clicks + total_nonclicks
    
    # 计算比例 (C/NC)
    initial_nc_over_c = total_clicks / total_nonclicks if total_clicks != 0 else float('inf')

    # 【新增】计算目标 CTR 阈值：1 / (1 + 1/TARGET_RATIO)
    target_ctr = 1 / (1+1/TARGET_RATIO)
    
    print(f"Initial Total Impressions: {total_impressions}")
    print(f"Initial Clicks: {total_clicks}, Non-clicks: {total_nonclicks}")
    print(f"Initial Ratio (C:NC): 1 : {initial_nc_over_c:.2f}")

    # 开始淘汰用户逻辑
    current_nc_over_c = initial_nc_over_c
    users_removed = 0
    
    # 复制一份用于操作，计算 CTR (Click-Through Rate) 用于排序
    # CTR = clicks / impressions，避免了 nonclick 为 0 的问题
    temp_stats = user_stats.copy()
    temp_stats['ctr'] = temp_stats['clicks'] / temp_stats['impressions']
    
    print(f"Target Ratio (C:NC): 1 : {TARGET_RATIO}. Starting user pruning...")

    #判断最开始的比例是比他大还是小
    initial_is_greater = (initial_nc_over_c > TARGET_RATIO)
    
    for i in tqdm(range(MAX_USERS_TO_REMOVE), desc="Pruning users"):
        # 如果初始大于目标，现在小于等于目标了，说明已经越过/达到目标，停止
        if initial_is_greater and (current_nc_over_c <= TARGET_RATIO):
            break
        # 如果初始小于目标，现在大于等于目标了，说明已经越过/达到目标，停止
        if (not initial_is_greater ) and (current_nc_over_c >= TARGET_RATIO):
            break
            
        # 排序逻辑：
        # CTR越高表示有更多的clicks
        # 1. 如果 current_nc_over_c > TARGET_RATIO (比如 1/2 > 1/2.7)，说明 click 太多，整体点击率太低
        #    -> 需要删掉 CTR 最高的用户 (sort ascending=True, 删第一个)
        # 2. 如果 current_nc_over_c < TARGET_RATIO (比如 1/3 < 1/2.7)，说明 non-click 太多，整体点击率太高
        #    -> 需要删掉 CTR 最高的用户 (sort ascending=False, 删第一个)
        
        #if current_nc_over_c > TARGET_RATIO:
            # Click 过多，删 CTR 最大的
        #    temp_stats_sorted = temp_stats.sort_values(by='ctr', ascending=False)
        # 【修改】排序逻辑：
        if current_nc_over_c > TARGET_RATIO:
            # 【新逻辑】Click 过多 (C/NC > TARGET)，从 CTR > target_ctr 的用户中删除
            # 先筛选出 CTR 超过阈值的用户
            candidates_to_remove = temp_stats[temp_stats['ctr'] > target_ctr]
            # 对超过阈值的用户按clicks (从超过阈值但是click数量较少的开始删，这样可以提高click/per user的数值)
            temp_stats_sorted = candidates_to_remove.sort_values(by='clicks', ascending=True)
        else :
            # Non-click 过多，删 CTR 最小的
            temp_stats_sorted = temp_stats.sort_values(by='ctr', ascending=True)
            
        if len(temp_stats_sorted) == 0:
            break
            
        # 取出要删除的那一行
        row_to_remove = temp_stats_sorted.iloc[0]
        user_id_to_remove = row_to_remove.name # 注意：groupby 后的 index 是 user_id，如果 reset_index 了需要用 iloc[0]['user_id']
        
        # 更新总数 (减去该用户的贡献)
        total_clicks -= row_to_remove['clicks']
        total_nonclicks -= row_to_remove['nonclicks']
        
        # 从 DataFrame 中移除该用户
        # 这里我们通过保留 "不等于该 user_id" 的行来实现
        # 注意：因为之前 reset_index 了，user_id 在列里
        temp_stats = temp_stats[temp_stats['user_id'] != row_to_remove['user_id']]
        
        # 更新当前比例
        if total_clicks == 0:
            break # 避免除以0
        
        current_nc_over_c = total_clicks / total_nonclicks
        users_removed += 1

    # 获取最终保留的用户集合
    final_users = set(temp_stats['user_id'])
    
    print(f"Removed {users_removed} users.")
    print(f"Final Clicks: {total_clicks}, Non-clicks: {total_nonclicks}")
    final_ratio = total_nonclicks / total_clicks if total_clicks != 0 else 0
    print(f"Final Ratio (C:NC): 1 : {final_ratio:.2f}")

    # 筛选最终的交互数据
    df_imp_final = df_imp_filtered[df_imp_filtered['user_id'].isin(final_users)]
    
    # 保存 inter_impression.csv
    df_imp_final[['user_id', 'answer_id', 'imp_ts', 'click_ts']].to_csv(
        os.path.join(OUTPUT_DIR, 'inter_impression.csv'), 
        index=False, header=False
    )
    del df_imp_filtered

    # ================= 步骤 4: 处理 Answer, Question, Topic =================
    print("\n[3/8] Processing Answer metadata...")
    info_answer_path = os.path.join(INPUT_DIR, 'info_answer.csv')
    dtype_ans = {0: 'int32', 1: 'Int32', 2: 'Int8', 3: 'Int32'} # 使用Nullable Integer
    df_answer = pd.read_csv(info_answer_path, header=None, dtype=dtype_ans, low_memory=False)
    
    # 筛选 answer
    df_answer_final = df_answer[df_answer[0].isin(top_answers)]
    df_answer_final.to_csv(os.path.join(OUTPUT_DIR, 'info_answer.csv'), index=False, header=False)
    
    # 获取相关 questions
    related_questions = set(df_answer_final[1].dropna().astype(int))
    
    del df_answer

    print("\n[4/8] Processing Question metadata...")
    info_question_path = os.path.join(INPUT_DIR, 'info_question.csv')
    df_question = pd.read_csv(info_question_path, header=None, dtype={0: 'int32'})
    df_question_final = df_question[df_question[0].isin(related_questions)]
    df_question_final.to_csv(os.path.join(OUTPUT_DIR, 'info_question.csv'), index=False, header=False)
    
    del df_question

    # 获取相关 topics (从 answer 和 question 中提取)
    print("\n[5/8] Processing Topic metadata...")
    related_topics = set()
    
    # 从 answer (col 17) 提取
    for topics_str in df_answer_final[17].dropna():
        related_topics.update(map(int, str(topics_str).split()))
        
    # 从 question (col 7) 提取
    for topics_str in df_question_final[7].dropna():
        related_topics.update(map(int, str(topics_str).split()))

    info_topic_path = os.path.join(INPUT_DIR, 'info_topic.csv')
    df_topic = pd.read_csv(info_topic_path, header=None, dtype={0: 'int32'})
    df_topic_final = df_topic[df_topic[0].isin(related_topics)]
    df_topic_final.to_csv(os.path.join(OUTPUT_DIR, 'info_topic.csv'), index=False, header=False)
    
    del df_topic, df_answer_final, df_question_final

    # ================= 步骤 5: 处理 User 和 Query =================
    print("\n[6/8] Processing User and Query data...")
    
    # Query
    inter_query_path = os.path.join(INPUT_DIR, 'inter_query.csv')
    df_query = pd.read_csv(inter_query_path, header=None, dtype={0: 'int32'})
    df_query_final = df_query[df_query[0].isin(final_users)]
    df_query_final.to_csv(os.path.join(OUTPUT_DIR, 'inter_query.csv'), index=False, header=False)
    
    # User
    info_user_path = os.path.join(INPUT_DIR, 'info_user.csv')
    df_user = pd.read_csv(info_user_path, header=None, dtype={0: 'int32'}, low_memory=False)
    df_user_final = df_user[df_user[0].isin(final_users)]
    df_user_final.to_csv(os.path.join(OUTPUT_DIR, 'info_user.csv'), index=False, header=False)
    
    # Author (从 answer col 3 获取)
    # 重新读取 answer 以获取 author 信息（或者刚才保留）
    # 这里为了内存，刚才删了，现在重新读入筛选后的 answer 来获取 author
    # 其实刚才应该保留 df_answer_final 的，这里修正逻辑：
    # 重新读取 answer final 来获取 author
    df_answer_final = pd.read_csv(os.path.join(OUTPUT_DIR, 'info_answer.csv'), header=None, dtype={3: 'Int32'})
    related_authors = set(df_answer_final[3].dropna().astype(int))
    
    info_author_path = os.path.join(INPUT_DIR, 'info_author.csv')
    df_author = pd.read_csv(info_author_path, header=None, dtype={0: 'int32'})
    df_author_final = df_author[df_author[0].isin(related_authors)]
    df_author_final.to_csv(os.path.join(OUTPUT_DIR, 'info_author.csv'), index=False, header=False)
    
    del df_user, df_query, df_author

    # ================= 步骤 6: 处理 Tokens =================
    print("\n[7/8] Processing Token metadata...")
    related_tokens = set()
    
    # 从 Query (col 1) 提取
    for tokens_str in df_query_final[1].dropna():
        related_tokens.update(map(int, str(tokens_str).split()))
    
    # 从 Answer (col 16) 提取
    for tokens_str in df_answer_final[16].dropna():
        related_tokens.update(map(int, str(tokens_str).split()))
    
    # 从 Question (col 6) 提取 (需要重新读入 question final)
    df_question_final = pd.read_csv(os.path.join(OUTPUT_DIR, 'info_question.csv'), header=None)
    for tokens_str in df_question_final[6].dropna():
        related_tokens.update(map(int, str(tokens_str).split()))
        
    info_token_path = os.path.join(INPUT_DIR, 'info_token.csv')
    # 读取 token，这个文件比较大，注意内存
    df_token = pd.read_csv(info_token_path, header=None, dtype={0: 'int32'})
    df_token_final = df_token[df_token[0].isin(related_tokens)]
    df_token_final.to_csv(os.path.join(OUTPUT_DIR, 'info_token.csv'), index=False, header=False)
    
    del df_token, df_answer_final, df_question_final, df_query_final, df_user_final

    # ================= 步骤 7: 生成统计信息 =================
    print("\n[8/8] Generating statistics...")
    
    # 重新读取最终的 inter 来统计最准确
    df_imp_final = pd.read_csv(os.path.join(OUTPUT_DIR, 'inter_impression.csv'), header=None, names=['u', 'a', 'ts', 'c'])
    df_q_final = pd.read_csv(os.path.join(OUTPUT_DIR, 'inter_query.csv'), header=None)
    
    num_impressions = len(df_imp_final)
    num_clicks = (df_imp_final['c'] != 0).sum()
    num_nonclicks = num_impressions - num_clicks
    ratio_str = f"1 : {num_nonclicks/num_clicks:.2f}" if num_clicks != 0 else "N/A"
    
    num_queries = len(df_q_final)
    num_users = df_imp_final['u'].nunique()
    avg_imp_per_user = num_impressions / num_users if num_users != 0 else 0
    avg_click_per_user = num_clicks / num_users if num_users != 0 else 0
    
    num_users_with_queries = df_q_final[0].nunique()
    avg_q_per_user = num_queries / num_users_with_queries if num_users_with_queries != 0 else 0
    
    # 读取其他文件计数
    num_answers = len(pd.read_csv(os.path.join(OUTPUT_DIR, 'info_answer.csv'), header=None, usecols=[0]))
    num_questions = len(pd.read_csv(os.path.join(OUTPUT_DIR, 'info_question.csv'), header=None, usecols=[0]))
    num_authors = len(pd.read_csv(os.path.join(OUTPUT_DIR, 'info_author.csv'), header=None, usecols=[0]))
    num_topics = len(pd.read_csv(os.path.join(OUTPUT_DIR, 'info_topic.csv'), header=None, usecols=[0]))
    num_tokens = len(pd.read_csv(os.path.join(OUTPUT_DIR, 'info_token.csv'), header=None, usecols=[0]))

    # 构建 JSON 字典 (注意：用 int() 和 float() 强制转换为 Python 原生类型)
    stats_dict = {
        "dataset_name": "ZhihuRec-Custom",
        "statistics": {
            "impressions": int(num_impressions),
            "clicks": int(num_clicks),
            "non_clicks": int(num_nonclicks),
            "ratio_clicks_to_nonclicks": f"{ratio_str}",
            "queries": int(num_queries),
            "users": int(num_users),
            "avg_impressions_per_user": float(avg_imp_per_user),
            "avg_clicks_per_user": float(avg_click_per_user),
            "users_with_queries": int(num_users_with_queries),
            "avg_queries_per_user": float(avg_q_per_user),
            "answers": int(num_answers),
            "questions": int(num_questions),
            "authors": int(num_authors),
            "topics": int(num_topics),
            "tokens": int(num_tokens)
        }
    }

    # 打印预览
    import json
    print(json.dumps(stats_dict, indent=4, ensure_ascii=False))
    
    # 保存为 JSON 文件
    json_output_path = os.path.join(OUTPUT_DIR, 'statistics.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=4, ensure_ascii=False)

    print(f"\nStatistics saved to {json_output_path}")
    print("\nAll Done!")

if __name__ == "__main__":
    main()