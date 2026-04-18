import json
import os
import logging
from typing import Dict, List, Tuple, Any, Set
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ================= 配置与日志 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 路径配置 - 新增output文件夹
ZHIHU_DATA_DIR = "./data/zhihu"
INTERIM_DATA_DIR = "./data"
OUTPUT_DIR = "./output"  # 所有评估结果存放到此文件夹

# 确保output文件夹存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 所有文件名统一配置（核心修改）=====================
# 在这里增删改文件名，无需改动下方逻辑
FILE_CONFIG = {
    # 知乎原始数据文件
    "train_data": "train.json",
    "dev_data": "dev.json",
    "test_data": "test.json",
    "answer_info": "answer_info.json",
    
    # 聚类结果文件
    "kmeans_cluster": "kmeans_results_100.json",
    "mvc_cluster": "mvc_results_100.json",
    
    # LLM种子结果文件名模板（{method}_{num} 会自动替换）
    "llm_seed_template": "llm_seed_results_{method}_{num}.json"
}

# ================= 辅助函数：直接加载JSON文件 =================
def load_json_file(file_path: str) -> List[Dict]:
    """直接加载标准JSON文件（文件内容为一个数组）"""
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 如果文件是单个对象而非数组，尝试寻找可能的列表字段（如 'data' 或 'results'）
            for k, v in data.items():
                if isinstance(v, list):
                    logger.info(f"文件 {file_path} 为对象包装格式，提取字段 '{k}'")
                    return v
            logger.warning(f"文件 {file_path} 是单个对象，非预期的数组格式")
            return [data]
        logger.info(f"成功加载 {file_path}: {len(data)} 条记录")
        return data
    except Exception as e:
        logger.error(f"加载文件失败 {file_path}: {e}")
        return []

# ================= 辅助函数：构建映射关系 =================
def build_user_cluster_weight(cluster_file: str) -> Dict[int, int]:
    """从聚类结果构建 user_id -> cluster_size 的权重映射"""
    user_weight = {}
    cluster_data = load_json_file(cluster_file)
    for cluster in cluster_data:
        user_ids = cluster.get("user_ids", [])
        cluster_size = cluster.get("cluster_size", 1)
        for uid in user_ids:
            user_weight[uid] = cluster_size
    return user_weight

def build_answer_topic_map(answer_info_file: str) -> Dict[int, List[int]]:
    """构建 answer_id -> topics 映射"""
    ans_topics = {}
    answer_data = load_json_file(answer_info_file)
    for ans in answer_data:
        aid = ans.get("answer_id")
        topics = ans.get("topics", [])
        if aid is not None:
            ans_topics[aid] = topics
    return ans_topics

def build_all_label_map() -> Dict[Tuple[int, int], int]:
    """加载train/dev/test全量数据集，构建 (user_id, answer_id) -> is_click 真实标签映射"""
    label_map = {}
    # 从配置读取数据集文件名
    data_files = [
        os.path.join(ZHIHU_DATA_DIR, FILE_CONFIG["train_data"]),
        os.path.join(ZHIHU_DATA_DIR, FILE_CONFIG["dev_data"]),
        os.path.join(ZHIHU_DATA_DIR, FILE_CONFIG["test_data"])
    ]
    
    for file_path in data_files:
        data = load_json_file(file_path)
        for item in data:
            uid = item.get("user_id")
            aid = item.get("answer_id")
            click = item.get("is_click")
            if uid is not None and aid is not None and click is not None:
                label_map[(uid, aid)] = click
    logger.info(f"加载全量标签完成，共 {len(label_map)} 条有效标签")
    return label_map

# ================= 核心指标计算 =================
def calculate_metrics(y_true: List[int], y_pred: List[int], y_score: List[float]) -> Dict[str, float]:
    """计算分类指标: ACC, PRECISION, RECALL, F1, AUC"""
    metrics = {}
    metrics["ACC"] = round(accuracy_score(y_true, y_pred), 6)
    metrics["PRECISION"] = round(precision_score(y_true, y_pred, zero_division=0), 6)
    metrics["RECALL"] = round(recall_score(y_true, y_pred, zero_division=0), 6)
    metrics["F1"] = round(f1_score(y_true, y_pred, zero_division=0), 6)
    try:
        metrics["AUC"] = round(roc_auc_score(y_true, y_score), 6)
    except ValueError:
        metrics["AUC"] = 0.0  # 处理只有一类样本的情况
    return metrics

def calculate_weighted_confidence(confidences: List[int], weights: List[int]) -> float:
    """计算加权平均置信度"""
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return round(sum(c * w for c, w in zip(confidences, weights)) / total_w, 6)

def get_unique_user_count(users: List[int]) -> int:
    """获取独立用户个数"""
    return len(set(users))

# ================= 单个种子结果处理 =================
def process_single_seed(
    method: str,
    num: int,
    llm_file: str,
    label_map: Dict[Tuple[int, int], int],
    ans_topic_map: Dict[int, List[int]],
    user_weight_map: Dict[int, int]
) -> Dict[str, Any]:
    """处理单个LLM种子结果，返回该种子的全量统计和按topic的统计"""
    logger.info(f"正在处理种子: {method}_{num}")
    
    # 1. 加载LLM结果并对齐真实标签
    llm_data = load_json_file(llm_file)
    valid_samples = []
    for item in llm_data:
        uid = item.get("user_id")
        aid = item.get("answer_id")
        y_pred = item.get("seed_click")
        conf = item.get("confidence", 3)  # 置信度默认值3
        
        key = (uid, aid)
        y_true = label_map.get(key)
        if y_true is not None and y_pred is not None:
            valid_samples.append({
                "uid": uid, "aid": aid,
                "y_true": y_true, "y_pred": y_pred, "conf": conf
            })
    
    if not valid_samples:
        logger.warning(f"{method}_{num} 无有效样本，跳过")
        return {}

    # 2. 计算全量指标
    y_true_all = [s["y_true"] for s in valid_samples]
    y_pred_all = [s["y_pred"] for s in valid_samples]
    y_score_all = [(s["conf"] - 1) / 4.0 for s in valid_samples]  # 1-5归一化到0-1
    conf_all = [s["conf"] for s in valid_samples]
    weights_all = [user_weight_map.get(s["uid"], 1) for s in valid_samples]
    users_all = [s["uid"] for s in valid_samples]
    
    overall_metrics = calculate_metrics(y_true_all, y_pred_all, y_score_all)
    overall_metrics["weighted_confidence"] = calculate_weighted_confidence(conf_all, weights_all)
    overall_metrics["user_count"] = get_unique_user_count(users_all)
    overall_metrics["total_samples"] = len(valid_samples)

    # 3. 按Topic统计（核心需求：每个topic关联的answer都要算）
    topic_metrics = {}  # topic_id -> 该种子在该topic下的7个指标
    topic_user_collector = {}  # topic_id -> 该topic下的用户列表（用于统计独立用户数）
    
    for s in valid_samples:
        topics = ans_topic_map.get(s["aid"], [])
        if not topics:
            continue
        for t in topics:
            # 初始化topic数据
            if t not in topic_metrics:
                topic_metrics[t] = {
                    "y_true": [], "y_pred": [], "y_score": [], 
                    "conf": [], "weights": [], "users": []
                }
            # 填充topic级数据
            topic_metrics[t]["y_true"].append(s["y_true"])
            topic_metrics[t]["y_pred"].append(s["y_pred"])
            topic_metrics[t]["y_score"].append((s["conf"] - 1) / 4.0)
            topic_metrics[t]["conf"].append(s["conf"])
            topic_metrics[t]["weights"].append(user_weight_map.get(s["uid"], 1))
            topic_metrics[t]["users"].append(s["uid"])

    # 计算每个topic的7个指标
    for t, data in topic_metrics.items():
        if len(data["y_true"]) < 2:
            # 样本量过少时，仅保留基础统计，指标置0
            topic_metrics[t] = {
                "ACC": 0.0, "PRECISION": 0.0, "RECALL": 0.0,
                "F1": 0.0, "AUC": 0.0, "weighted_confidence": 0.0,
                "user_count": get_unique_user_count(data["users"]),
                "total_samples": len(data["y_true"])
            }
            continue
        
        # 计算核心指标
        t_metrics = calculate_metrics(data["y_true"], data["y_pred"], data["y_score"])
        t_metrics["weighted_confidence"] = calculate_weighted_confidence(data["conf"], data["weights"])
        t_metrics["user_count"] = get_unique_user_count(data["users"])
        t_metrics["total_samples"] = len(data["y_true"])
        topic_metrics[t] = t_metrics

    # 4. 组装该种子的结果
    seed_result = {
        "seed_id": f"{method}_{num}",
        "method": method,
        "num_clusters": num,
        "overall": overall_metrics,
        "topic_detail": topic_metrics
    }
    return seed_result

# ================= Topic维度汇总 =================
def summarize_by_topic(all_seed_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """按topic_id汇总所有9个种子的指标"""
    topic_summary = {}
    
    # 遍历所有种子结果
    for seed_res in all_seed_results:
        if not seed_res:
            continue
        seed_id = seed_res["seed_id"]
        topic_detail = seed_res["topic_detail"]
        
        # 遍历该种子的所有topic
        for topic_id, metrics in topic_detail.items():
            if topic_id not in topic_summary:
                topic_summary[topic_id] = {
                    "total_samples": 0,  # 该topic下所有种子的总样本数
                    "seeds": {}  # seed_id -> 7个指标
                }
            
            # 累加总样本数
            topic_summary[topic_id]["total_samples"] += metrics["total_samples"]
            # 保存该种子在该topic下的7个核心指标
            topic_summary[topic_id]["seeds"][seed_id] = {
                "ACC": metrics["ACC"],
                "PRECISION": metrics["PRECISION"],
                "RECALL": metrics["RECALL"],
                "F1": metrics["F1"],
                "AUC": metrics["AUC"],
                "weighted_confidence": metrics["weighted_confidence"],
                "user_count": metrics["user_count"]
            }
    
    # 排序并格式化
    sorted_topic_summary = {
        int(tid): data for tid, data in sorted(topic_summary.items(), key=lambda x: int(x[0]))
    }
    return sorted_topic_summary

# ================= 主流程 =================
def main():
    # 1. 从配置加载基础数据文件路径
    answer_info_file = os.path.join(ZHIHU_DATA_DIR, FILE_CONFIG["answer_info"])
    kmeans_cluster_file = os.path.join(INTERIM_DATA_DIR, FILE_CONFIG["kmeans_cluster"])
    mvc_cluster_file = os.path.join(INTERIM_DATA_DIR, FILE_CONFIG["mvc_cluster"])

    # 加载全量标签（train+dev+test）
    label_map = build_all_label_map()
    # 构建answer-topic映射
    ans_topic_map = build_answer_topic_map(answer_info_file)
    # 构建用户权重映射
    user_weight_maps = {
        "kmeans": build_user_cluster_weight(kmeans_cluster_file),
        "mvc": build_user_cluster_weight(mvc_cluster_file),
        "topk": {}  # TopK权重默认为1
    }

    # 2. 定义9个种子（3方法×3规模），从模板生成文件名
    #methods = ["kmeans", "mvc", "topk"]
    methods=["kmeans1000","kmeans1500"]
    nums = [10, 20, 50]
    tasks = []
    for m in methods:
        for n in nums:
            # 使用配置模板生成实际文件名
            fname = FILE_CONFIG["llm_seed_template"].format(method=m, num=n)
            fpath = os.path.join(INTERIM_DATA_DIR, fname)
            if os.path.exists(fpath):
                tasks.append((m, n, fpath))
            else:
                logger.warning(f"种子文件未找到，跳过: {fpath}")

    # 3. 处理所有种子
    all_seed_results = []
    for m, n, fpath in tasks:
        seed_res = process_single_seed(
            method=m,
            num=n,
            llm_file=fpath,
            label_map=label_map,
            ans_topic_map=ans_topic_map,
            user_weight_map=user_weight_maps.get(m, {})
        )
        if seed_res:
            all_seed_results.append(seed_res)
            # 保存单个种子结果到output文件夹
            out_file = os.path.join(OUTPUT_DIR, f"seed_eval_{m}_{n}.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(seed_res, f, ensure_ascii=False, indent=2)
            logger.info(f"单个种子结果已保存至: {out_file}")

    if not all_seed_results:
        logger.error("无有效种子结果，流程终止")
        return

    # 4. 按Topic汇总所有种子
    topic_summary = summarize_by_topic(all_seed_results)
    topic_summary_file = os.path.join(OUTPUT_DIR, "topic_wise_summary.json")
    with open(topic_summary_file, 'w', encoding='utf-8') as f:
        json.dump(topic_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Topic维度汇总结果已保存至: {topic_summary_file}")

    # 5. 保存全量汇总结果
    summary_file = os.path.join(OUTPUT_DIR, "all_seeds_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_seed_results, f, ensure_ascii=False, indent=2)
    logger.info(f"所有种子汇总结果已保存至: {summary_file}")

    logger.info("全量评估流程完成，所有结果已保存至output文件夹")

if __name__ == "__main__":
    main()