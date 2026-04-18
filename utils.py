import os
import json
import math
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from scipy.sparse import coo_matrix, csr_matrix
from scipy import sparse as ss
from collections import defaultdict
from typing import Union

def init_seed(seed):
    """初始化随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def format_metric(result_dict: dict) -> str:
    """格式化指标输出"""
    format_str = []
    for k, v in sorted(result_dict.items()):
        if isinstance(v, float):
            format_str.append(f'{k}:{v:.4f}')
        else:
            format_str.append(f'{k}:{v}')
    return ','.join(format_str)


def check_dir(file_name: str):
    """检查并创建目录"""
    dir_path = os.path.dirname(file_name)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_json(file_path: str) -> list:
    """读取JSON文件（兼容数组/对象格式）"""
    check_dir(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Union[list, dict], file_path: str):
    """保存JSON文件"""
    check_dir(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_torch_file(file_path: str, device: torch.device) -> object:
    """读取PyTorch序列化文件"""
    return torch.load(file_path, map_location=device)


def save_torch_file(data: object, file_path: str):
    """保存PyTorch序列化文件"""
    check_dir(file_path)
    torch.save(data, file_path)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个向量的余弦相似度（处理零向量）"""
    vec1 = np.array(vec1).astype(np.float32)
    vec2 = np.array(vec2).astype(np.float32)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def get_top_k_active_users(data: list, k: int, user_info_data: list = None) -> list:
    """
    筛选Top-K活跃用户
    :param data: 交互数据列表（包含user_id）
    :param k: 种子用户数量
    :param user_info_data: 用户信息数据列表（包含num_topics_followed等字段）
    :return: Top-K活跃用户ID列表
    """
    # 综合多字段评分：num_topics_followed + num_answers + num_likes_received + num_followers
    user_info_map = {item['user_id']: item for item in user_info_data} if user_info_data else {}
    user_score = {}
        
    # 步骤1：统计点击数（基础分）
    user_click_count = {}
    for item in data:
        user_id = item['user_id']
        user_click_count[user_id] = user_click_count.get(user_id, 0) + item['is_click']
        
    # 步骤2：计算综合得分（归一化+加权）
    for user_id in user_click_count.keys():
        info = user_info_map.get(user_id, {})
        # 提取用户行为字段（默认0）
        num_topics = info.get('user_info', {}).get('num_topics_followed', 0)
        num_answers = info.get('user_info', {}).get('num_answers', 0)
        num_likes = info.get('user_info', {}).get('num_likes_received', 0)
        num_followers = info.get('user_info', {}).get('num_followers', 0)
        click_count = user_click_count[user_id]
            
        # 归一化（避免量级差异）
        all_num_topics = [u.get('user_info', {}).get('num_topics_followed', 0) for u in user_info_map.values()] if user_info_map else [0]
        all_num_answers = [u.get('user_info', {}).get('num_answers', 0) for u in user_info_map.values()] if user_info_map else [0]
        all_num_likes = [u.get('user_info', {}).get('num_likes_received', 0) for u in user_info_map.values()] if user_info_map else [0]
        all_num_followers = [u.get('user_info', {}).get('num_followers', 0) for u in user_info_map.values()] if user_info_map else [0]
        all_click_counts = list(user_click_count.values())
            
        max_topics = max(all_num_topics) if all_num_topics else 1
        max_answers = max(all_num_answers) if all_num_answers else 1
        max_likes = max(all_num_likes) if all_num_likes else 1
        max_followers = max(all_num_followers) if all_num_followers else 1
        max_clicks = max(all_click_counts) if all_click_counts else 1
            
        norm_topics = num_topics / max_topics
        norm_answers = num_answers / max_answers
        norm_likes = num_likes / max_likes
        norm_followers = num_followers / max_followers
        norm_clicks = click_count / max_clicks
            
        # 加权得分（可按需调整权重）
        score = (
            norm_clicks * 0.6 +    # 点击数（60%）
            norm_topics * 0.1 +    # 关注话题数（10%）
            norm_answers * 0.15 +  # 发布回答数（15%）
            norm_likes * 0.05 +     # 获赞数（5%）
            norm_followers * 0.2   # 粉丝数（20%）
        )
        user_score[user_id] = score
        
    # 按综合得分降序取Top-K
    sorted_users = sorted(user_score.items(), key=lambda x: x[1], reverse=True)
    top_k_users = [user[0] for user in sorted_users[:k]] if sorted_users else []
    return top_k_users


def build_id2idx_mapping(id_list: list) -> dict:
    """
    构建原始ID到连续索引的映射
    :param id_list: 原始ID列表
    :return: id2idx字典（padding_idx=0）
    """
    unique_ids = sorted(list(set(id_list)))
    id2idx = {pid: idx + 1 for idx, pid in enumerate(unique_ids)}  # 0作为padding
    id2idx['padding'] = 0
    return id2idx


def _convert_sp_mat_to_sp_tensor(sp_mat):
    """将scipy稀疏矩阵转换为PyTorch稀疏张量"""
    coo = sp_mat.tocoo().astype(np.float32)
    # 先合并为单个numpy数组，再转tensor
    indices_np = np.vstack((coo.row, coo.col))
    indices = torch.tensor(indices_np, dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

def build_hypergraph(
    user_item_data: list,  # 用于构建用户-物品超图（需含is_click）
    user_user_data: list,  # 用于构建用户-用户超图（需含time）
    user2idx: dict, 
    answer2idx: dict, 
    window: int = 5
) -> list:
    """
    重构版超图构建：分离用户-物品/用户-用户超图的数据源
    :param user_item_data: 构建用户-物品超图的数据源（含is_click，如train/seed数据）
    :param user_user_data: 构建用户-用户超图的数据源（含time，如train/test数据）
    :param user2idx: 用户ID到索引的映射
    :param answer2idx: 回答ID到索引的映射
    :param window: 滑动窗口大小
    :return: [user_item_adj_sparse, user_user_adj_sparse] 稀疏矩阵
    """
    n_users = len(user2idx)
    n_answers = len(answer2idx)

    # ===================== 1. 构建用户-物品（Answer）超图（依赖is_click） =====================
    row_user = []
    col_answer = []
    data_interact = []
    for item in user_item_data:
        user_id = item['user_id']
        answer_id = item['answer_id']
        if user_id not in user2idx or answer_id not in answer2idx:
            continue
        # 只保留点击行为的交互（is_click=1）
        click_label = item.get('is_click', item.get('seed_click', 0))  # 兼容seed_data的seed_click字段
        if click_label == 1:
            u_idx = user2idx[user_id]
            a_idx = answer2idx[answer_id]
            row_user.append(u_idx)
            col_answer.append(a_idx)
            data_interact.append(1.0)
    
    # 转为稀疏矩阵 (n_users, n_answers)
    user_item_adj = coo_matrix(
        (data_interact, (row_user, col_answer)),
        shape=(n_users, n_answers),
        dtype=np.float32
    )
    
    # 转换为PyTorch稀疏张量
    indices_np = np.vstack((user_item_adj.row, user_item_adj.col))
    indices_tensor = torch.tensor(indices_np, dtype=torch.long)
    user_item_adj_sparse = torch.sparse_coo_tensor(
        indices_tensor,
        torch.tensor(user_item_adj.data, dtype=torch.float32),
        size=user_item_adj.shape,
        dtype=torch.float32
    )

    # ===================== 2. 构建用户-用户超图（依赖time） =====================
    # 步骤1：构建Cascade列表（按answer分组，按时间戳排序）
    answer2users = defaultdict(list)
    for item in user_user_data:
        user_id = item['user_id']
        answer_id = item['answer_id']
        if user_id not in user2idx or answer_id not in answer2idx:
            continue
        # 仅保留有time字段的记录（test/train数据）
        if 'time' not in item:
            continue
        # 只处理点击/曝光行为（兼容test无is_click的情况）
        u_idx = user2idx[user_id]
        timestamp = int(item['time'])  # 转换为整数时间戳
        answer2users[answer_id].append((u_idx, timestamp))
    
    # 生成最终cascade列表（过滤空列表 + 按时间戳排序）
    all_cascade = []
    for ans_id, user_with_ts in answer2users.items():
        # 按时间戳排序
        user_with_ts_sorted = sorted(user_with_ts, key=lambda x: x[1])
        cascade = [u[0] for u in user_with_ts_sorted]  # 提取排序后的用户索引
        if len(cascade) > 0:
            all_cascade.append(cascade)

    # 步骤2：滑动窗口构建用户上下文
    user_cont = {u_idx: [] for u_idx in range(n_users)}  # 初始化每个用户的上下文
    win = window
    for cas in all_cascade:
        if len(cas) < win:
            # 若cascade长度小于窗口，整个序列作为上下文
            for idx in cas:
                user_cont[idx] = list(set(user_cont[idx] + cas))
            continue
        # 滑动窗口遍历cascade（步长=1）
        for j in range(len(cas) - win + 1):
            cas_win = cas[j:j+win]  # 窗口内的用户序列
            for idx in cas_win:
                # 上下文去重（避免重复邻居）
                user_cont[idx] = list(set(user_cont[idx] + cas_win))

    # 步骤3：构建用户-用户超图稀疏矩阵
    indptr, indices, data = [], [], []
    indptr.append(0)
    # 处理空上下文用户：保留用户索引位，避免indptr错位
    for j in range(n_users):
        ctx = user_cont[j]
        if len(ctx) == 0:
            indptr.append(indptr[-1])  # 空上下文：indptr延续，保证长度=n_users+1
            continue
        # 去重并排序邻居（提升稀疏矩阵效率）
        neighbors = np.unique(ctx)
        length = len(neighbors)
        # 填充indices和data
        indices.extend(neighbors)
        data.extend([1.0] * length)
        # 更新indptr
        indptr.append(indptr[-1] + length)

    # 构建H_U稀疏矩阵（shape=(n_users, n_users)）
    H_U = ss.csr_matrix(
        (data, indices, indptr),
        shape=(n_users, n_users),
        dtype=np.float32
    )

    # 归一化计算
    # 计算BH_T: H_U的行归一化
    H_U_sum = H_U.sum(axis=1).reshape(-1, 1)  # 行求和
    H_U_sum[H_U_sum == float("inf")] = 0
    H_U_sum[H_U_sum == 0] = 1  # 避免除零（空上下文用户）
    H_U_sum_inv = 1.0 / H_U_sum
    BH_T = H_U.T.multiply(H_U_sum_inv).T

    # 计算DH: H_U.T的行归一化
    H = H_U.T
    H_sum = H.sum(axis=1).reshape(-1, 1)
    H_sum[H_sum == float("inf")] = 0
    H_sum[H_sum == 0] = 1
    H_sum_inv = 1.0 / H_sum
    DH = H.T.multiply(H_sum_inv).T

    # 计算最终用户-用户超图邻接矩阵
    user_user_adj = (DH @ BH_T).tocoo()

    # 转换为torch稀疏张量
    user_user_adj_sparse = _convert_sp_mat_to_sp_tensor(user_user_adj)

    return [user_item_adj_sparse, user_user_adj_sparse]


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """
    计算评估指标：ACC、Recall、Precision、F1
    :param y_true: 真实标签
    :param y_pred: 预测概率
    :param threshold: 概率转标签的阈值
    :return: 指标字典
    """
    y_pred_label = (y_pred >= threshold).astype(int)
    return {
        "ACC": accuracy_score(y_true, y_pred_label),
        "RECALL": recall_score(y_true, y_pred_label, zero_division=0),
        "PRECISION": precision_score(y_true, y_pred_label, zero_division=0),
        "F1": f1_score(y_true, y_pred_label, zero_division=0)
    }