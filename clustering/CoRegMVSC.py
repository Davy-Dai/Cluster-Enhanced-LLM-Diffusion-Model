import json
import numpy as np
from scipy.spatial.distance import cosine
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from collections import defaultdict
import utils  # 新增：导入工具函数

def clean_vector(vec):
    """清洗向量：NaN/Inf → 0，强制转为float64"""
    vec = np.array(vec, dtype=np.float64)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec

def calculate_time_decayed_average(vectors_with_timestamps, gamma=0.8):
    """
    计算【排序次方衰减】的加权平均向量（完全按你的要求重写）
    逻辑：
    1. 按时间戳从小到大排序（旧→新）
    2. 最新行为权重=γ⁰=1，次新=γ¹，最旧=γ^(n-1)
    3. 时间越小（越旧）的行为，乘以γ的次数越多，权重越低
    :param vectors_with_timestamps: 列表，每个元素是 (vector, timestamp_str)
    :param gamma: 时间衰减系数，默认0.8
    :return: 64维加权平均向量
    """
    if not vectors_with_timestamps:
        return np.zeros(64, dtype=np.float64)
    
    # 步骤1：过滤无效数据（空时间戳、全零向量、非数字时间戳）
    valid_pairs = []
    for vec, ts_str in vectors_with_timestamps:
        if not ts_str:
            continue
        clean_vec = clean_vector(vec)
        if np.linalg.norm(clean_vec) < 1e-8:
            continue
        try:
            ts = int(ts_str)
            valid_pairs.append((clean_vec, ts))
        except ValueError:
            continue
    
    if not valid_pairs:
        return np.zeros(64, dtype=np.float64)
    
    # 步骤2：按时间戳升序排序（旧→新）
    valid_pairs.sort(key=lambda x: x[1])
    
    # 步骤3：计算排序次方衰减权重（最新行为权重=1，越旧权重越低）
    n = len(valid_pairs)
    total_weight = 0.0
    weighted_sum = np.zeros(64, dtype=np.float64)
    
    for idx in range(n):
        vec, _ = valid_pairs[idx]
        # 核心：时间越小（索引越小），次方越高 → 权重=gamma^(n-1-idx)
        weight = gamma ** (n - 1 - idx)
        weight = max(weight, 1e-8)  # 防止权重过小导致数值不稳定
        
        weighted_sum += vec * weight
        total_weight += weight
    
    # 步骤4：加权平均（防止除零）
    avg_vec = weighted_sum / total_weight if total_weight > 1e-8 else np.zeros(64, dtype=np.float64)
    return clean_vector(avg_vec)

def build_user_similarity_matrices(train_data, user_info_dict, answer_info_dict):
    # 核心修正1：从user_info_dict获取全量用户ID，而非仅train_data中的用户
    user_ids = list(user_info_dict.keys())  # 保证包含所有用户
    user_list = sorted(user_ids)
    n = len(user_list)
    user2idx = {uid: idx for idx, uid in enumerate(user_list)}
    
    user_vectors = {}
    for uid in user_list:
        # 2.1 排序次方衰减的Query向量
        query_records = user_info_dict[uid].get('query', [])
        q_vectors_with_ts = [
            (q['content'], q.get('time', '')) 
            for q in query_records
        ]
        query_avg = calculate_time_decayed_average(q_vectors_with_ts, gamma=0.8)
        
        # 2.2 排序次方衰减的Click-answer向量
        click_records = [item for item in train_data if item['user_id'] == uid and item['is_click'] == 1]
        ca_vectors_with_ts = [
            (answer_info_dict[rec['answer_id']]['answer_info'], rec.get('time', ''))
            for rec in click_records
            if rec['answer_id'] in answer_info_dict  # 过滤无answer_info的记录
        ]
        click_answer_avg = calculate_time_decayed_average(ca_vectors_with_ts, gamma=0.8)
        
        # 2.3 排序次方衰减的Nonclick-answer向量
        nonclick_records = [item for item in train_data if item['user_id'] == uid and item['is_click'] == 0]
        nca_vectors_with_ts = [
            (answer_info_dict[rec['answer_id']]['answer_info'], rec.get('time', ''))
            for rec in nonclick_records
            if rec['answer_id'] in answer_info_dict  # 过滤无answer_info的记录
        ]
        nonclick_answer_avg = calculate_time_decayed_average(nca_vectors_with_ts, gamma=0.8)
        
        user_vectors[uid] = {
            'query': query_avg,
            'click_answer': click_answer_avg,
            'nonclick_answer': nonclick_answer_avg
        }
    
    K1 = np.zeros((n, n), dtype=np.float64)
    K2 = np.zeros((n, n), dtype=np.float64)
    K3_raw = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        uid_i = user_list[i]
        vec_i_q = user_vectors[uid_i]['query']
        vec_i_ca = user_vectors[uid_i]['click_answer']
        vec_i_nca = user_vectors[uid_i]['nonclick_answer']
        
        for j in range(i, n):
            uid_j = user_list[j]
            vec_j_q = user_vectors[uid_j]['query']
            vec_j_ca = user_vectors[uid_j]['click_answer']
            vec_j_nca = user_vectors[uid_j]['nonclick_answer']
            
            # 修复：余弦相似度双重防NaN + 强制clip到[0,1]
            def calc_sim(a, b):
                if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
                    return 0.0
                sim = 1 - cosine(a, b)
                return np.clip(sim, 0.0, 1.0)  # 杜绝浮点误差负数
            
            sim_q = calc_sim(vec_i_q, vec_j_q)
            sim_ca = calc_sim(vec_i_ca, vec_j_ca)
            sim_nca = calc_sim(vec_i_nca, vec_j_nca)
            
            K1[i][j] = K1[j][i] = sim_q
            K2[i][j] = K2[j][i] = sim_ca
            K3_raw[i][j] = K3_raw[j][i] = sim_nca
    
    # 最终清洗：矩阵全局清除NaN
    K1 = clean_vector(K1)
    K2 = clean_vector(K2)
    K3_raw = clean_vector(K3_raw)
    return user_list, K1, K2, K3_raw

def collaborative_regularized_multiview_spectral_clustering(K1, K2, K3_raw, n_clusters, lambda_=1.0, epsilon=1e-4, max_iter=10):
    n = K1.shape[0]
    
    # 修复：K3清洗 + 防负数
    K3 = clean_vector(1 - K3_raw)
    K3 = np.clip(K3, 0.0, 1.0)
    K_list = [K1, K2, K3]
    U_list = []
    
    # 通用：拉普拉斯矩阵计算（封装防零+防NaN）
    def compute_laplacian(K):
        K = clean_vector(K)
        D = np.diag(np.sum(K, axis=1))
        D_diag = np.diag(D)
        # 终极防零：小于1e-6 → 强制设为1e-6，杜绝sqrt(0)
        D_diag = np.where(D_diag < 1e-6, 1e-6, D_diag)
        D_sqrt_inv = np.diag(1.0 / np.sqrt(D_diag))
        L = D_sqrt_inv @ K @ D_sqrt_inv
        return clean_vector(L)  # 最终清洗
    
    # 步骤1：初始化
    for K in K_list:
        L = compute_laplacian(K)
        eigvals, eigvecs = eigh(L)
        top_k_idx = np.argsort(eigvals)[-n_clusters:]
        U = clean_vector(eigvecs[:, top_k_idx])
        U, _ = np.linalg.qr(U)
        U_list.append(U)
    
    # 步骤2：迭代优化
    prev_obj = None
    for iter_idx in range(max_iter):
        for v in range(3):
            K_v = K_list[v]
            L_v = compute_laplacian(K_v)
            
            # 正则项清洗
            reg_term = np.zeros((n, n), dtype=np.float64)
            for w in range(3):
                if w != v:
                    reg_term += clean_vector(U_list[w] @ U_list[w].T)
            reg_term *= lambda_
            
            # 修正矩阵+特征分解
            L_new = clean_vector(L_v + reg_term)
            eigvals, eigvecs = eigh(L_new)
            top_k_idx = np.argsort(eigvals)[-n_clusters:]
            U_v_new = clean_vector(eigvecs[:, top_k_idx])
            U_v_new, _ = np.linalg.qr(U_v_new)
            U_list[v] = U_v_new
        
        # 目标函数（清洗防NaN）
        obj = 0.0
        for v in range(3):
            L_v = compute_laplacian(K_list[v])
            obj += np.trace(clean_vector(U_list[v].T @ L_v @ U_list[v]))
        
        cross_term = 0.0
        for v in range(3):
            for w in range(3):
                if v != w:
                    cross_term += np.trace(clean_vector((U_list[v] @ U_list[v].T) @ (U_list[w] @ U_list[w].T)))
        obj += lambda_ * cross_term
        
        if prev_obj is not None and abs(obj - prev_obj) < epsilon:
            break
        prev_obj = obj
    
    # 步骤3：聚类
    U_final = clean_vector(np.hstack(U_list))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(U_final)
    return cluster_labels

def get_cluster_info(cluster_labels, user_list, user_info_dict, train_data, answer_info_dict):
    """
    统计每个簇的核心信息
    返回：
        cluster_info: {簇id: {
            'users': [uid1, uid2,...],          # 簇内用户ID
            'center_query': np.array,           # 簇中心query向量
            'center_click_answer': np.array,    # 簇中心click-answer向量
            'center_nonclick_answer': np.array, # 簇中心nonclick-answer向量
            'dominant_gender': str,             # 簇主导性别
            'avg_user_info': dict,              # 簇用户info平均值
            'topics': dict                      # 簇topics统计（id: 个数）
        }}
    """
    n_clusters = len(np.unique(cluster_labels))
    cluster_info = {}
    
    for c in range(n_clusters):
        # 1. 簇内用户列表
        cluster_idxs = np.where(cluster_labels == c)[0]
        cluster_users = [user_list[idx] for idx in cluster_idxs]
        
        # 2. 收集簇内用户的核心特征（复用排序次方衰减逻辑）
        query_vecs, ca_vecs, nca_vecs = [], [], []
        genders = []
        user_info_list = []
        all_topics = []
        
        for uid in cluster_users:
            # Query向量（排序次方衰减）
            query_records = user_info_dict[uid].get('query', [])
            q_vectors_with_ts = [
                (q['content'], q.get('time', '')) 
                for q in query_records
            ]
            q_avg = calculate_time_decayed_average(q_vectors_with_ts, gamma=0.8)
            query_vecs.append(q_avg)
            
            # Click-answer向量（排序次方衰减）
            click_records = [item for item in train_data if item['user_id'] == uid and item['is_click'] == 1]
            ca_vectors_with_ts = [
                (answer_info_dict[rec['answer_id']]['answer_info'], rec.get('time', ''))
                for rec in click_records
                if rec['answer_id'] in answer_info_dict
            ]
            ca_avg = calculate_time_decayed_average(ca_vectors_with_ts, gamma=0.8)
            ca_vecs.append(ca_avg)
            
            # Nonclick-answer向量（排序次方衰减）
            nonclick_records = [item for item in train_data if item['user_id'] == uid and item['is_click'] == 0]
            nca_vectors_with_ts = [
                (answer_info_dict[rec['answer_id']]['answer_info'], rec.get('time', ''))
                for rec in nonclick_records
                if rec['answer_id'] in answer_info_dict
            ]
            nca_avg = calculate_time_decayed_average(nca_vectors_with_ts, gamma=0.8)
            nca_vecs.append(nca_avg)
            
            # 性别（兼容无性别信息的情况）
            genders.append(user_info_dict[uid].get('gender', 'unknown'))
            
            # 用户info（兼容无info的情况）
            ui = user_info_dict[uid].get('user_info', {
                'num_topics_followed': 0,
                'num_answers': 0,
                'num_likes_received': 0,
                'num_followers': 0
            })
            user_info_list.append([
                ui['num_topics_followed'], ui['num_answers'],
                ui['num_likes_received'], ui['num_followers']
            ])
            
            # 关注的topics（兼容无topics的情况）
            all_topics.extend(user_info_dict[uid].get('topics', []))
        
        # 3. 计算簇中心向量（兼容空列表）
        center_query = clean_vector(np.mean(query_vecs, axis=0)) if query_vecs else np.zeros(64, dtype=np.float64)
        center_ca = clean_vector(np.mean(ca_vecs, axis=0)) if ca_vecs else np.zeros(64, dtype=np.float64)
        center_nca = clean_vector(np.mean(nca_vecs, axis=0)) if nca_vecs else np.zeros(64, dtype=np.float64)
        
        # 4. 主导性别（统计最多，兼容空列表）
        gender_count = defaultdict(int)
        for g in genders:
            gender_count[g] += 1
        dominant_gender = max(gender_count, key=gender_count.get) if gender_count else 'unknown'
        
        # 5. 用户info平均值（兼容空列表）
        avg_ui = np.mean(user_info_list, axis=0) if user_info_list else np.zeros(4, dtype=np.float64)
        avg_user_info = {
            'num_topics_followed': float(avg_ui[0]),
            'num_answers': float(avg_ui[1]),
            'num_likes_received': float(avg_ui[2]),
            'num_followers': float(avg_ui[3])
        }
        
        # 6. Topics统计（dict形式，兼容空列表）
        topic_count = defaultdict(int)
        for t in all_topics:
            topic_count[t] += 1
        
        cluster_info[c] = {
            'users': cluster_users,
            'center_query': center_query,
            'center_click_answer': center_ca,
            'center_nonclick_answer': center_nca,
            'dominant_gender': dominant_gender,
            'avg_user_info': avg_user_info,
            'topics': dict(topic_count)
        }
    
    return cluster_info

def run_mvsc_and_save(data_path, n_clusters, mvsc_lambda, save_path, user_scope, top_k):
    """
    执行完整的多视图谱聚类流程并保存结果到JSON
    JSON包含：cluster_id, user_ids, cluster_size,
    """
    # 1. 加载数据
    train_data = utils.load_json(f"{data_path}/train.json")
    user_info = utils.load_json(f"{data_path}/user_info.json")
    answer_info = utils.load_json(f"{data_path}/answer_info.json")
    
    # ===================== 新增：用户范围过滤逻辑 =====================
    if user_scope == 'top_k':
        print(f"Filtering Top-{top_k} active users...")
        # 获取 Top-K 活跃用户 ID
        top_k_user_ids = utils.get_top_k_active_users(train_data, top_k, user_info)
        # 过滤 user_info（仅保留 Top-K 用户）
        user_info = [item for item in user_info if item['user_id'] in top_k_user_ids]
        # 过滤 train_data（仅保留 Top-K 用户的交互记录）
        train_data = [item for item in train_data if item['user_id'] in top_k_user_ids]


    user_profile_dict = {item['user_id']: item for item in user_info}
    answer_info_dict = {item['answer_id']: item for item in answer_info}
    
    # 2. 构建相似度矩阵并聚类
    print(f"总用户数：{len(user_profile_dict)}")  # 新增：打印全量用户数，验证是否为3333
    print("Building user similarity matrices...")
    user_list, K1, K2, K3_raw = build_user_similarity_matrices(
        train_data, user_profile_dict, answer_info_dict
    )
    print(f"相似度矩阵维度：{K1.shape}")  # 新增：验证矩阵维度是否为3333x3333
    
    print("Running collaborative regularized multi-view spectral clustering...")
    cluster_labels = collaborative_regularized_multiview_spectral_clustering(
        K1, K2, K3_raw, n_clusters=n_clusters, lambda_=mvsc_lambda
    )
    print(f"聚类标签数量：{len(cluster_labels)}")  # 新增：验证聚类标签数是否为3333
    
    print("Getting cluster info...")
    cluster_info = get_cluster_info(
        cluster_labels, user_list, user_profile_dict, train_data, answer_info_dict
    )
    
    # 3. 转换为JSON格式并保存
    json_results = []
    total_users_in_result = 0
    for cluster_id, info in cluster_info.items():
        total_users_in_result += len(info['users'])
        json_results.append({
            "cluster_id": cluster_id,
            "user_ids": info['users'],
            "cluster_size": len(info['users']),
            "center_query": info['center_query'].tolist(),  # numpy数组转list
            "center_click_answer": info['center_click_answer'].tolist(),
            "center_nonclick_answer": info['center_nonclick_answer'].tolist(),
            "dominant_gender": info['dominant_gender'],
            "avg_user_info": info['avg_user_info'],
            "topics": info['topics']
        })
    print(f"结果中总用户数：{total_users_in_result}")  # 新增：验证结果用户总数
    
    utils.save_json(json_results, save_path)
    print(f"MVSC clustering complete! Results saved to: {save_path}")
    return json_results