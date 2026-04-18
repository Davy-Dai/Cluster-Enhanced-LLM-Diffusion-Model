import json
from collections import Counter

# ====================== 在这里设置你的相对路径 ======================
JSON_FILE_PATH = "kmeans_results_1000.json"  # 改成你实际的文件名/相对路径
# ==================================================================

def load_clusters(json_path):
    """读取聚类JSON文件"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return []

def analyze_user_ids(clusters):
    """统计所有 user_id 信息"""
    all_user_ids = []
    cluster_info = []

    for cluster in clusters:
        cid = cluster.get("cluster_id", -1)
        users = cluster.get("user_ids", [])
        size = cluster.get("cluster_size", len(users))

        all_user_ids.extend(users)
        cluster_info.append({
            "cluster_id": cid,
            "user_count": size,
            "user_list_length": len(users)
        })

    # 统计
    total_users = len(all_user_ids)
    unique_users = len(set(all_user_ids))
    duplicate_users = [uid for uid, cnt in Counter(all_user_ids).items() if cnt > 1]

    # 输出结果
    print("=" * 60)
    print("📊 用户ID 统计结果")
    print("=" * 60)
    print(f"✅ 所有簇的用户总数（含重复）: {total_users}")
    print(f"✅ 唯一用户总数: {unique_users}")
    print(f"⚠️  出现在多个簇中的重复用户数量: {len(duplicate_users)}")

    if len(duplicate_users) > 0:
        print(f"⚠️  重复用户ID: {duplicate_users}")
    else:
        print("✅ 无重复用户，所有用户只属于一个簇")

    print("\n" + "=" * 60)
    print("📦 每个簇的大小统计")
    print("=" * 60)
    for info in cluster_info:
        print(f"簇 {info['cluster_id']:2d} | 标注大小: {info['user_count']:3d} | 实际列表长度: {info['user_list_length']:3d}")

    return all_user_ids, unique_users, duplicate_users

# ====================== 主程序 ======================
if __name__ == "__main__":
    clusters_data = load_clusters(JSON_FILE_PATH)
    if clusters_data:
        analyze_user_ids(clusters_data)