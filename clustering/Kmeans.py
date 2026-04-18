# -*- coding: UTF-8 -*-
import json
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import argparse
import utils

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_user_answer_click_mapping(train_data):
    """
    从train数据中构建用户的点击/未点击answer映射
    :param train_data: train.json加载后的列表
    :return: dict，key=user_id，value={'click': [answer_id列表], 'nonclick': [answer_id列表], 'time': {answer_id: 时间戳}}
    """
    user_answer_map = defaultdict(lambda: {'click': [], 'nonclick': [], 'time': {}})
    for item in train_data:
        user_id = item['user_id']
        answer_id = item['answer_id']
        is_click = item['is_click']
        time_stamp = int(item['time'])
        
        if is_click == 1:
            user_answer_map[user_id]['click'].append(answer_id)
        else:
            user_answer_map[user_id]['nonclick'].append(answer_id)
        user_answer_map[user_id]['time'][answer_id] = time_stamp
    return user_answer_map

def time_decay_weighted_avg(vectors_with_time, gamma=0.8):
    """
    【修改版】计算基于时间排序的衰减加权平均的64维向量
    逻辑：按时间戳从小到大排序（越早越靠前），排序越靠后（越新）权重越大
    :param vectors_with_time: 列表，每个元素是(64维向量, 时间戳)
    :param gamma: 时间衰减系数
    :return: 64维加权平均向量（无数据时返回全0）
    """
    if not vectors_with_time:
        return np.zeros(64, dtype=np.float32)
    
    # 1. 按时间戳从小到大排序（index 0 是最早的记录）
    sorted_items = sorted(vectors_with_time, key=lambda x: x[1])
    n_items = len(sorted_items)
    
    vectors = []
    weights = []
    
    # 2. 遍历排序后的列表，根据“位置索引”计算权重
    for i, (vec, _) in enumerate(sorted_items):
        exponent = (n_items - 1) - i
        weight = np.power(gamma, exponent)
        
        vectors.append(vec)
        weights.append(weight)
    
    # 转为numpy数组
    vectors = np.array(vectors, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    
    # 3. 归一化权重并计算加权平均
    weights = weights / weights.sum()
    weighted_avg = np.sum(vectors * weights.reshape(-1, 1), axis=0)
    
    return weighted_avg

def build_user_feature_matrix(user_info_data, answer_info_data, user_answer_map, gamma=0.8):
    """
    构建用户特征矩阵：每个用户3个64维特征（query/click_answer/nonclick_answer）
    :param user_info_data: user_info.json加载后的列表
    :param answer_info_data: answer_info.json加载后的列表
    :param user_answer_map: 用户点击/未点击answer映射
    :param gamma: 时间衰减系数
    :return: (user_ids, features)，features是shape=(n_users, 192)的数组（3*64）
    """
    # 构建answer_id到answer_info的映射
    answer2vec = {item['answer_id']: item['answer_info'] for item in answer_info_data}
    
    user_ids = []
    user_features = []
    
    for user_item in user_info_data:
        user_id = user_item['user_id']
        user_ids.append(user_id)
        
        # 1. 计算query的时间衰减加权平均特征
        query_records = user_item.get('query', [])
        query_vectors_with_time = []
        for q in query_records:
            q_time = q.get('time', '')
            if q_time and q_time.isdigit():
                q_vec = q.get('content', np.zeros(64))
                query_vectors_with_time.append((q_vec, int(q_time)))
        query_feature = time_decay_weighted_avg(query_vectors_with_time, gamma)
        
        # 2. 计算click_answer的时间衰减加权平均特征
        click_answer_ids = user_answer_map.get(user_id, {}).get('click', [])
        click_vectors_with_time = []
        for ans_id in click_answer_ids:
            ans_vec = answer2vec.get(ans_id, np.zeros(64))
            ans_time = user_answer_map[user_id]['time'].get(ans_id, 0)
            click_vectors_with_time.append((ans_vec, ans_time))
        click_answer_feature = time_decay_weighted_avg(click_vectors_with_time, gamma)
        
        # 3. 计算nonclick_answer的时间衰减加权平均特征
        nonclick_answer_ids = user_answer_map.get(user_id, {}).get('nonclick', [])
        nonclick_vectors_with_time = []
        for ans_id in nonclick_answer_ids:
            ans_vec = answer2vec.get(ans_id, np.zeros(64))
            ans_time = user_answer_map[user_id]['time'].get(ans_id, 0)
            nonclick_vectors_with_time.append((ans_vec, ans_time))
        nonclick_answer_feature = time_decay_weighted_avg(nonclick_vectors_with_time, gamma)
        
        # 拼接三个特征（后续可通过权重调整各维度重要性）
        combined_feature = np.concatenate([query_feature, click_answer_feature, nonclick_answer_feature])
        user_features.append(combined_feature)
    
    return np.array(user_ids), np.array(user_features, dtype=np.float32)

def apply_linear_kernel_weight(features, w_query=1.0, w_click=1.0, w_nonclick=1.0):
    """
    对特征应用加权线性核（调整三个维度的权重）
    :param features: shape=(n_users, 192)的特征矩阵
    :param w_query: query维度权重
    :param w_click: click_answer维度权重
    :param w_nonclick: nonclick_answer维度权重
    :return: 加权后的特征矩阵
    """
    # 拆分三个64维特征
    query_feat = features[:, :64] * w_query
    click_feat = features[:, 64:128] * w_click
    nonclick_feat = features[:, 128:] * w_nonclick
    
    # 重新拼接
    weighted_feat = np.concatenate([query_feat, click_feat, nonclick_feat], axis=1)
    # 归一化（提升聚类效果）
    weighted_feat = normalize(weighted_feat, norm='l2', axis=1)
    return weighted_feat

def run_kmeans(features, n_clusters, random_seed=1000):
    """
    执行K-means聚类
    :param features: 加权后的特征矩阵
    :param n_clusters: 聚类数量
    :param random_seed: 随机种子
    :return: (cluster_labels, cluster_centers)
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=10,  # 多次初始化取最优
        max_iter=300
    )
    cluster_labels = kmeans.fit_predict(features)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers

def convert_numpy_to_python(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_to_python(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(x) for x in obj]
    else:
        return obj

def generate_cluster_statistics(user_ids, cluster_labels, user_info_data, cluster_centers):
    """
    生成聚类统计信息（符合指定JSON格式）
    :param user_ids: 用户ID列表
    :param cluster_labels: 每个用户的聚类标签
    :param user_info_data: user_info.json数据
    :param cluster_centers: 聚类中心（shape=(n_clusters, 192)）
    :return: 聚类结果列表（每个元素是一个簇的统计信息）
    """
    # 构建user_id到user_info的映射
    user2info = {item['user_id']: item for item in user_info_data}
    
    # 按簇分组用户
    cluster_user_map = defaultdict(list)
    for idx, user_id in enumerate(user_ids):
        cluster_id = cluster_labels[idx]
        cluster_user_map[cluster_id].append(user_id)
    
    cluster_results = []
    for cluster_id in sorted(cluster_user_map.keys()):
        user_list = cluster_user_map[cluster_id]
        cluster_size = len(user_list)
        
        # 拆分聚类中心的三个维度，并转换为Python原生类型
        center_query = convert_numpy_to_python(cluster_centers[cluster_id][:64])
        center_click_answer = convert_numpy_to_python(cluster_centers[cluster_id][64:128])
        center_nonclick_answer = convert_numpy_to_python(cluster_centers[cluster_id][128:])
        
        # 统计性别分布（dominant_gender）
        gender_counts = Counter()
        # 统计平均user_info
        user_info_sum = {
            'num_topics_followed': 0.0,
            'num_answers': 0.0,
            'num_likes_received': 0.0,
            'num_followers': 0.0
        }
        # 统计话题分布
        topic_counts = defaultdict(int)
        
        for user_id in user_list:
            user_info = user2info.get(user_id, {})
            
            # 性别统计
            gender = user_info.get('gender', 'unknown')
            gender_counts[gender] += 1
            
            # user_info累加（确保转换为Python原生类型）
            ui = user_info.get('user_info', {})
            user_info_sum['num_topics_followed'] += convert_numpy_to_python(ui.get('num_topics_followed', 0))
            user_info_sum['num_answers'] += convert_numpy_to_python(ui.get('num_answers', 0))
            user_info_sum['num_likes_received'] += convert_numpy_to_python(ui.get('num_likes_received', 0))
            user_info_sum['num_followers'] += convert_numpy_to_python(ui.get('num_followers', 0))
            
            # 话题统计
            topics = user_info.get('topics', [])
            for topic in topics:
                topic_counts[topic] += 1
        
        # 计算平均user_info（转换为Python原生float）
        avg_user_info = {
            k: convert_numpy_to_python(v / cluster_size) if cluster_size > 0 else 0.0
            for k, v in user_info_sum.items()
        }
        
        # 确定主导性别
        dominant_gender = gender_counts.most_common(1)[0][0] if gender_counts else 'unknown'
        
        # 构建簇统计信息（确保所有值都是Python原生类型）
        cluster_info = {
            'cluster_id': convert_numpy_to_python(cluster_id),
            'user_ids': convert_numpy_to_python(user_list),
            'cluster_size': convert_numpy_to_python(cluster_size),
            'center_query': center_query,
            'center_click_answer': center_click_answer,
            'center_nonclick_answer': center_nonclick_answer,
            'dominant_gender': dominant_gender,
            'avg_user_info': avg_user_info,
            'topics': convert_numpy_to_python(dict(topic_counts))  # 转为普通dict并转换类型
        }
        cluster_results.append(cluster_info)
    
    return cluster_results

def save_cluster_results(cluster_results, save_path):
    """保存聚类结果到JSON文件（处理numpy类型序列化）"""
    # 先转换所有numpy类型为Python原生类型
    cluster_results_py = convert_numpy_to_python(cluster_results)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_results_py, f, ensure_ascii=False, indent=2)

def run_kmeans_cluster(args,user_scope,top_k):
    """
    主函数：执行K-means聚类（供main.py调用）
    :param args: 命令行参数对象
    """
    # 1. 加载数据
    print("Loading data...")
    train_data = load_json_file(f"{args.data_path}/train.json")
    answer_info_data = load_json_file(f"{args.data_path}/answer_info.json")
    user_info_data = load_json_file(f"{args.data_path}/user_info.json")

    # ===================== 新增：用户范围过滤逻辑 =====================
    if user_scope == 'top_k':
        print(f"Filtering Top-{top_k} active users...")
        # 获取 Top-K 活跃用户 ID
        top_k_user_ids = utils.get_top_k_active_users(train_data, top_k, user_info_data)
        # 过滤 user_info_data（仅保留 Top-K 用户）
        user_info_data = [item for item in user_info_data if item['user_id'] in top_k_user_ids]
        # 过滤 train_data（仅保留 Top-K 用户的交互记录）
        train_data = [item for item in train_data if item['user_id'] in top_k_user_ids]
    
    # 2. 构建用户点击/未点击answer映射
    print("Building user-answer click mapping...")
    user_answer_map = get_user_answer_click_mapping(train_data)
    
    # 3. 构建用户特征矩阵
    print("Building user feature matrix (time-decay weighted)...")
    user_ids, raw_features = build_user_feature_matrix(
        user_info_data=user_info_data,
        answer_info_data=answer_info_data,
        user_answer_map=user_answer_map,
        gamma=args.kmeans_gamma
    )
    
    # 4. 应用线性核权重
    print("Applying linear kernel weights...")
    weighted_features = apply_linear_kernel_weight(
        features=raw_features,
        w_query=args.kmeans_w_query,
        w_click=args.kmeans_w_click,
        w_nonclick=args.kmeans_w_nonclick
    )
    
    # 5. 执行K-means聚类
    print(f"Running K-means clustering (n_clusters={args.kmeans_n_clusters})...")
    cluster_labels, cluster_centers = run_kmeans(
        features=weighted_features,
        n_clusters=args.kmeans_n_clusters,
        random_seed=args.seed
    )
    
    # 6. 生成聚类统计信息
    print("Generating cluster statistics...")
    cluster_results = generate_cluster_statistics(
        user_ids=user_ids,
        cluster_labels=cluster_labels,
        user_info_data=user_info_data,
        cluster_centers=cluster_centers
    )
    
    # 7. 保存结果
    print(f"Saving cluster results to {args.kmeans_save_path}...")
    save_cluster_results(cluster_results, args.kmeans_save_path)
    
    print("K-means clustering completed successfully!")