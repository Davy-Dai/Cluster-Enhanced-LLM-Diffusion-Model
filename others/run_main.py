import os
import subprocess
from pathlib import Path

# ====================== 超参数配置（可根据需求修改） ======================
METHODS = ["kmeans", "mvsc"]  # 聚类方法
TOPKS = list(range(100, 1100, 100))  # 100,200,...,1000
CLUSTERS = [10, 20, 50]  # 聚类簇数
DATA_ROOT = "./data"  # 数据根目录
DATA_PATH = "./data/zhihu"  # 原始数据路径
LLM_API_KEY = "sk-b430ecd7209a492284c98140edc869e0"  # LLM API密钥
CKPT_PATH = "./diffusion_ckpt/best_model.pth"  # 扩散模型权重路径

# ====================== 工具函数 ======================
def run_command(cmd: list, desc: str):
    """执行系统命令，打印执行信息和错误"""
    print(f"\n{'='*50}")
    print(f"执行命令: {desc}")
    print(f"命令详情: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"✅ {desc} 执行成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ {desc} 执行失败")
        print(f"错误输出: {e.stderr}")
        raise  # 执行失败时终止脚本，可根据需求改为continue

def check_file_exists(file_path: str) -> bool:
    """检查文件是否存在，存在则打印提示并返回True"""
    if Path(file_path).exists():
        print(f"📌 文件已存在，跳过: {file_path}")
        return True
    return False

# ====================== 主流程 ======================
def main():
    # 1. 创建method对应的文件夹
    for method in METHODS:
        method_dir = Path(DATA_ROOT) / method
        method_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 确保文件夹存在: {method_dir}")

    # 2. 遍历所有超参数组合执行流程
    for method in METHODS:
        for topk in TOPKS:
            for cluster in CLUSTERS:
                print(f"\n\n{'='*60}")
                print(f"开始处理: method={method}, topk={topk}, cluster={cluster}")
                print(f"{'='*60}")

                # -------------------------- 步骤1: 运行聚类（kmeans/mvsc） --------------------------
                method_dir = Path(DATA_ROOT) / method
                if method == "mvsc":
                    cluster_save_path = str(method_dir / f"mvsc{topk}_results_{cluster}.json")
                    cluster_cmd = [
                        "python", "main.py",
                        "--mode", "mvsc",
                        "--data_path", DATA_PATH,
                        "--user_scope", "top_k",
                        "--llm_seed_k", str(topk),
                        "--mvsc_n_clusters", str(cluster),
                        "--mvsc_save_path", cluster_save_path
                    ]
                else:  # kmeans
                    cluster_save_path = str(method_dir / f"kmeans{topk}_results_{cluster}.json")
                    cluster_cmd = [
                        "python", "main.py",
                        "--mode", "kmeans",
                        "--data_path", DATA_PATH,
                        "--user_scope", "top_k",
                        "--llm_seed_k", str(topk),
                        "--kmeans_n_clusters", str(cluster),
                        "--kmeans_save_path", cluster_save_path
                    ]

                # 检查聚类文件是否存在，不存在则执行
                if not check_file_exists(cluster_save_path):
                    run_command(cluster_cmd, f"[{method}] 聚类 (topk={topk}, cluster={cluster})")

                # -------------------------- 步骤2: 运行LLM生成seed结果 --------------------------
                llm_seed_save_path = str(method_dir / f"llm_seed_results_{method}2_{topk}_{cluster}.json")
                llm_cmd = [
                    "python", "main.py",
                    "--mode", "llm",
                    "--data_path", DATA_PATH,
                    "--llm_cluster_path", cluster_save_path,
                    "--llm_seed_k", str(cluster),  # 聚类簇数作为llm_seed_k
                    "--llm_api_key", LLM_API_KEY,
                    "--llm_save_path", llm_seed_save_path
                ]

                # 检查LLM seed文件是否存在，不存在则执行
                if not check_file_exists(llm_seed_save_path):
                    run_command(llm_cmd, f"[{method}] LLM生成seed (topk={topk}, cluster={cluster})")

                # -------------------------- 步骤3: 运行Diffusion预测 --------------------------
                diffusion_pred_save_path = str(method_dir / f"diffusion_pred_results_{method}_{topk}_{cluster}.json")
                diffusion_cmd = [
                    "python", "main.py",
                    "--mode", "diffusion",
                    "--data_path", DATA_PATH,
                    "--diffusion_mode", "eval",
                    "--load_ckpt_path", CKPT_PATH,
                    "--diffusion_seed_path", llm_seed_save_path,
                    "--diffusion_pred_save_path", diffusion_pred_save_path
                ]

                # 检查预测文件是否存在，不存在则执行
                if not check_file_exists(diffusion_pred_save_path):
                    run_command(diffusion_cmd, f"[{method}] Diffusion预测 (topk={topk}, cluster={cluster})")

    print(f"\n\n🎉 所有超参数组合处理完成！")

if __name__ == "__main__":
    main()