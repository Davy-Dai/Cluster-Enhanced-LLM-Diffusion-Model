import pandas as pd
import json
import os
import random

# ================= 配置区域 =================
# 请将此处修改为您存放原始 ZhihuRec 数据集的文件夹路径
INPUT_FOLDER = "./zhihu_sort"  

# 请将此处修改为您希望存放输出结果的文件夹路径
OUTPUT_FOLDER = "./zhihu"
# ===========================================

def load_token_vector_map(input_folder):
    """
    加载info_token.csv，构建token ID到64维向量的映射字典
    返回：key为token ID（整数），value为64维浮点数列表的映射字典
    """
    token_map = {}
    try:
        token_df = pd.read_csv(
            os.path.join(input_folder, "info_token.csv"),
            header=None,
            names=['token_id', 'vector'],
            usecols=[0, 1]  # 第0列：token ID，第1列：64维向量（空格分隔）
        )
        for _, row in token_df.iterrows():
            token_id = int(row['token_id'])  # 统一转为整数key
            vector_str = row['vector']
            
            # 解析64维向量并做有效性校验
            if pd.notna(vector_str):
                vec = [float(x.strip()) for x in vector_str.strip().split()]
                if len(vec) == 64:
                    token_map[token_id] = vec
        
        print(f"✅ 成功加载 {len(token_map)} 个token的64维向量映射")
    except FileNotFoundError:
        print("❌ 未找到info_token.csv文件，请检查数据集路径")
    except Exception as e:
        print(f"❌ 加载token向量失败: {e}")
    return token_map

def calculate_avg_token_vector(token_ids_str, token_vector_map):
    """
    根据token ID计算对应的64维词向量平均值
    输入：
        token_ids_str: 空格分隔的token ID字符串（如 "123 456 789"）
        token_vector_map: token ID到64维向量的映射字典
    返回：64维平均向量（无有效向量时返回全0）
    """
    # 空值/无效值处理
    if pd.isna(token_ids_str) or token_ids_str.strip() == "":
        return [0.0] * 64
    
    # 分割token ID并过滤空值
    token_ids = [tid.strip() for tid in token_ids_str.strip().split() if tid.strip()]
    valid_vectors = []
    
    for tid in token_ids:
        try:
            token_id = int(tid)
            # 从映射字典获取向量
            vec = token_vector_map.get(token_id, None)
            if vec is not None:
                valid_vectors.append(vec)
        except (ValueError, IndexError):
            # 非数字ID/解析失败则跳过
            continue
    
    # 无有效向量返回全0
    if not valid_vectors:
        return [0.0] * 64
    
    # 按维度计算平均值（64维分别求平均）
    avg_vector = [sum(col) / len(valid_vectors) for col in zip(*valid_vectors)]
    return avg_vector

def save_json(data, filename):
    """辅助函数：保存JSON文件（保证UTF-8编码和格式化）"""
    path = os.path.join(OUTPUT_FOLDER, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # 1. 创建输出目录
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 创建输出目录: {OUTPUT_FOLDER}")

    # 2. 加载token ID到64维向量的映射
    token_vector_map = load_token_vector_map(INPUT_FOLDER)

    # ===========================================
    # 第一步：处理交互数据 (train/dev/test.json) - 按answer_id划分
    # ===========================================
    print("\n🔄 正在处理交互日志 (inter_impression.csv)...")
    try:
        # 读取交互数据
        inter_df = pd.read_csv(
            os.path.join(INPUT_FOLDER, "inter_impression.csv"),
            header=None,
            names=['user_id', 'answer_id', 'time', 'click_time']
        )
        # 按时间升序排列（保证时序性）
        inter_df = inter_df.sort_values("time", ascending=True).reset_index(drop=True)
        # 构造点击标签（0/1）
        inter_df['is_click'] = (inter_df['click_time'] != 0).astype(int)
        
        # 格式化输出字段
        output_inter = inter_df[['user_id', 'answer_id', 'time', 'is_click']].copy()
        output_inter['time'] = output_inter['time'].astype(str)  # 时间转为字符串格式

        # ---------------- 核心修改：按answer_id划分数据集 ----------------
        # 1. 获取所有唯一的answer_id并随机打乱（保证划分随机性）
        unique_answers = output_inter['answer_id'].unique().tolist()
        random.seed(42)  # 固定随机种子，保证划分可复现
        random.shuffle(unique_answers)
        
        # 2. 按8:1:1划分answer_id
        total_answers = len(unique_answers)
        train_answer_num = int(total_answers * 0.8)
        dev_answer_num = int(total_answers * 0.1)
        # 处理整除误差（保证所有answer都被划分）
        test_answer_num = total_answers - train_answer_num - dev_answer_num
        
        train_answers = unique_answers[:train_answer_num]
        dev_answers = unique_answers[train_answer_num:train_answer_num+dev_answer_num]
        test_answers = unique_answers[train_answer_num+dev_answer_num:]
        
        # 3. 根据answer_id筛选各数据集的impression
        train_data_df = output_inter[output_inter['answer_id'].isin(train_answers)]
        dev_data_df = output_inter[output_inter['answer_id'].isin(dev_answers)]
        test_data_df = output_inter[output_inter['answer_id'].isin(test_answers)]
        
        # 4. 转换为字典格式
        train_data = train_data_df.to_dict(orient='records')
        dev_data = dev_data_df.to_dict(orient='records')
        test_data = test_data_df.to_dict(orient='records')

        # 5. 保存文件
        save_json(train_data, "train.json")
        save_json(dev_data, "dev.json")
        save_json(test_data, "test.json")
        
        # 6. 计算并输出各数据集的impression数量及比例
        total_impressions = len(output_inter)
        train_imp_num = len(train_data)
        dev_imp_num = len(dev_data)
        test_imp_num = len(test_data)
        
        train_imp_ratio = train_imp_num / total_impressions
        dev_imp_ratio = dev_imp_num / total_impressions
        test_imp_ratio = test_imp_num / total_impressions
        
        print("\n📊 数据集划分统计（按answer_id）:")
        print(f"总answer数量: {total_answers}")
        print(f"Train answer数量: {len(train_answers)} ({len(train_answers)/total_answers:.2%})")
        print(f"Dev answer数量: {len(dev_answers)} ({len(dev_answers)/total_answers:.2%})")
        print(f"Test answer数量: {len(test_answers)} ({len(test_answers)/total_answers:.2%})")
        print("-" * 50)
        print(f"总impression数量: {total_impressions}")
        print(f"Train impression数量: {train_imp_num} ({train_imp_ratio:.2%})")
        print(f"Dev impression数量: {dev_imp_num} ({dev_imp_ratio:.2%})")
        print(f"Test impression数量: {test_imp_num} ({test_imp_ratio:.2%})")
        print("✅ 交互数据处理完成")

    except Exception as e:
        print(f"❌ 处理交互数据出错: {e}")

    # ===========================================
    # 第二步：处理回答信息 (answer_info.json)
    # ===========================================
    print("\n🔄 正在处理回答信息 (info_answer.csv)...")
    try:
        # 读取回答数据（仅取需要的列）
        answer_df = pd.read_csv(
            os.path.join(INPUT_FOLDER, "info_answer.csv"),
            header=None,
            usecols=[0, 16, 17],  # answer_id, token_ids, topic_ids
            names=['answer_id', 'token_ids', 'topic_ids']
        )

        answer_list = []
        for _, row in answer_df.iterrows():
            answer_id = row['answer_id']
            
            # 计算token的64维平均词向量
            avg_token_vector = calculate_avg_token_vector(row['token_ids'], token_vector_map)
            
            # 处理话题ID（修正为数字数组格式，而非字典）
            topics_list = []
            if pd.notna(row['topic_ids']):
                topic_ids = [tid.strip() for tid in str(row['topic_ids']).split() if tid.strip()]
                topics_list = [int(tid) for tid in topic_ids if tid.isdigit()]

            answer_list.append({
                "answer_id": int(answer_id) if pd.notna(answer_id) else 0,
                "answer_info": avg_token_vector,  # 64维平均词向量
                "topics": topics_list  # 话题ID数组
            })

        save_json(answer_list, "answer_info.json")
        print("✅ 回答信息处理完成")

    except Exception as e:
        print(f"❌ 处理回答信息出错: {e}")

    # ===========================================
    # 第三步：处理用户信息 (user_info.json)
    # ===========================================
    print("\n🔄 正在处理用户信息 (info_user.csv & inter_query.csv)...")
    try:
        # 1. 读取用户基础属性
        user_df = pd.read_csv(
            os.path.join(INPUT_FOLDER, "info_user.csv"),
            header=None,
            usecols=[0, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 26], 
            names=['user_id', 'gender', 'num_followers', 'num_topics_followed', 
                   'num_answers', 'num_questions', 'num_comments', 
                   'num_thanks_received', 'num_comments_received', 
                   'num_likes_received', 'num_dislikes_received', 'topic_ids_followed']
        )

        # 2. 读取用户查询记录并分组
        query_dict = {}
        try:
            query_df = pd.read_csv(
                os.path.join(INPUT_FOLDER, "inter_query.csv"),
                header=None,
                names=['user_id', 'query_tokens', 'time']
            )
            
            for qid, (_, row) in enumerate(query_df.iterrows()):
                uid = row['user_id']
                if uid not in query_dict:
                    query_dict[uid] = []
                
                # 计算查询token的64维平均词向量
                avg_query_vector = calculate_avg_token_vector(row['query_tokens'], token_vector_map)
                
                query_dict[uid].append({
                    "query_id": qid,
                    "content": avg_query_vector,  # 64维平均词向量
                    "time": str(row['time']) if pd.notna(row['time']) else ""
                })
        except FileNotFoundError:
            print("⚠️  未找到 inter_query.csv，查询字段将为空")
        except Exception as e:
            print(f"⚠️  处理查询数据出错: {e}")

        # 3. 整合用户数据
        user_list = []
        for _, row in user_df.iterrows():
            uid = row['user_id']
            
            # 处理性别
            gender_code = row['gender']
            if pd.isna(gender_code):
                gender_str = "unknown"
            elif gender_code == 1:
                gender_str = "male"
            elif gender_code == 2:
                gender_str = "female"
            else:
                gender_str = "unknown"

            # 处理用户行为统计（空值转0）
            stats = {
                "num_topics_followed": int(row['num_topics_followed']) if pd.notna(row['num_topics_followed']) else 0,       #关注话题数
                "num_answers": int(row['num_answers']) if pd.notna(row['num_answers']) else 0,                           #回答数量
                "num_likes_received": int(row['num_likes_received']) if pd.notna(row['num_likes_received']) else 0,             #获赞数量
                "num_followers": int(row['num_followers']) if pd.notna(row['num_followers']) else 0,                       #粉丝数量
            }

            # 处理用户关注话题（修正为数字数组）
            user_topics = []
            if pd.notna(row['topic_ids_followed']):
                t_ids = [tid.strip() for tid in str(row['topic_ids_followed']).split() if tid.strip()]
                user_topics = [int(tid) for tid in t_ids if tid.isdigit()]

            user_list.append({
                "user_id": int(uid) if pd.notna(uid) else 0,
                "gender": gender_str,
                "user_info": stats,
                "topics": user_topics,  # 关注话题ID数组
                "query": query_dict.get(uid, [])
            })

        save_json(user_list, "user_info.json")
        print("✅ 用户信息处理完成")

    except Exception as e:
        print(f"❌ 处理用户信息出错: {e}")

    print("\n🎉 所有任务处理完毕！输出文件已保存至:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()