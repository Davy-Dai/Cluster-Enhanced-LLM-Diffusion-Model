# -*- coding: UTF-8 -*-
import argparse
import torch
import utils
import LLMAgent
import DiffusionAgent
import clustering.CoRegMVSC as CoRegMVSC
import clustering.Kmeans as kmeans


def parse_args():
    parser = argparse.ArgumentParser(description='LLM+Diffusion Click Prediction Model')
    
    parser.add_argument('--mode', type=str, required=True, choices=['llm', 'diffusion', 'mvsc', 'kmeans'],
                        help='运行模式：llm（生成种子）/ diffusion（训练预测）/ mvsc（谱聚类）/ kmeans（加权K-means聚类）')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据集目录路径（包含train.json/dev.json/test.json等）')
    parser.add_argument('--seed', type=int, default=1000, help='随机种子')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='运行设备')
    
    parser.add_argument('--llm_seed_k', type=int, default=100,
                        help='LLM种子用户数量/簇选择数量（仅llm模式有效）')
    parser.add_argument('--llm_model_name', type=str, default='qwen-flash',
                        help='LLM模型名称')
    parser.add_argument('--llm_api_key', type=str, required=False,
                        help='LLM API Key（建议通过环境变量设置）')
    parser.add_argument('--llm_base_url', type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1',
                        help='LLM API Base URL')
    parser.add_argument('--llm_save_path', type=str, default='./data/llm_seed_results.json',
                        help='LLM种子结果保存路径')
    parser.add_argument('--llm_gamma', type=float, default=0.6,
                        help='时间衰减因子γ（仅llm模式有效）')
    parser.add_argument('--llm_cluster_path', type=str, default=None,
                        help='聚类结果JSON路径（仅llm模式使用聚类时有效）')
    
    parser.add_argument('--mvsc_n_clusters', type=int, default=10,
                        help='多视图聚类簇数（仅mvsc模式有效）')
    parser.add_argument('--mvsc_mvsc_lambda', type=float, default=0.05,
                        help='多视图聚类协同正则化系数λ（仅mvsc模式有效）')
    parser.add_argument('--mvsc_save_path', type=str, default='./data/mvsc_cluster_results.json',
                        help='MVSC聚类结果保存路径（仅mvsc模式有效）')
    
    parser.add_argument('--kmeans_n_clusters', type=int, default=10,
                        help='K-means聚类簇数（仅kmeans模式有效）')
    parser.add_argument('--kmeans_gamma', type=float, default=0.8,
                        help='K-means时间衰减系数γ（仅kmeans模式有效）')
    parser.add_argument('--kmeans_w_query', type=float, default=0.5,
                        help='K-means query特征权重（线性核，仅kmeans模式有效）')
    parser.add_argument('--kmeans_w_click', type=float, default=0.3,
                        help='K-means click_answer特征权重（线性核，仅kmeans模式有效）')
    parser.add_argument('--kmeans_w_nonclick', type=float, default=0.2,
                        help='K-means nonclick_answer特征权重（线性核，仅kmeans模式有效）')
    parser.add_argument('--kmeans_save_path', type=str, default='./data/kmeans_cluster_results.json',
                        help='K-means聚类结果保存路径（仅kmeans模式有效）')
    
    parser.add_argument('--diffusion_mode', type=str, default='train', choices=['train', 'eval'],
                        help='Diffusion模式：train（训练）/ eval（评估）（仅diffusion模式有效）')
    parser.add_argument('--patience', type=int, default=1000, help='早停耐心值（仅diffusion训练有效）')
    parser.add_argument('--emb_size', type=int, default=64, help='嵌入维度')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=4, help='多头注意力头数')
    parser.add_argument('--epoch', type=int, default=30, help='训练轮数（仅diffusion训练有效）')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    # 调整seed路径：仅eval模式需要
    parser.add_argument('--diffusion_seed_path', type=str,
                        help='LLM种子结果路径（仅diffusion eval模式有效）')
    # 新增推理结果保存路径
    parser.add_argument('--diffusion_pred_save_path', type=str, default='./data/diffusion_pred_metrics.json',
                        help='Diffusion推理结果保存路径（仅diffusion eval模式有效）')
    parser.add_argument('--save_every_n_epochs', type=int, default=5,
                        help='训练每隔N轮保存权重（仅diffusion训练有效）')
    parser.add_argument('--save_dir', type=str, default='./diffusion_ckpt',
                        help='权重保存目录（仅diffusion训练有效）')
    parser.add_argument('--load_ckpt_path', type=str,
                        help='评估时加载的权重文件路径（仅diffusion eval模式有效）')

    #新增实验部分实验方案选择
    parser.add_argument('--user_scope', type=str, default='all', choices=['all', 'top_k'],
                        help='聚类用户范围：all(全部用户)/top_k(Top-K活跃用户)')
    
    args = parser.parse_args()
    
    # 校验diffusion模式的参数
    if args.mode == 'diffusion':
        if args.diffusion_mode == 'eval':
            if not args.diffusion_seed_path:
                parser.error('--diffusion_seed_path is required when diffusion_mode=eval')
            if not args.load_ckpt_path:
                parser.error('--load_ckpt_path is required when diffusion_mode=eval')
    
    return args


def run_llm(args):
    llm_config = {
        "model_name": args.llm_model_name,
        "api_key": args.llm_api_key,
        "base_url": args.llm_base_url,
        "temperature": 0.7,
        "max_tokens": 512,
        "timeout": 120,
        "gamma": args.llm_gamma
    }
    
    llm_agent = LLMAgent.LLMAgent(llm_config)
    
    llm_agent.generate_seed_clicks(
        data_path=args.data_path,
        seed_k=args.llm_seed_k,
        save_path=args.llm_save_path,
        cluster_path=args.llm_cluster_path
    )


def run_kmeans(args):
    kmeans.run_kmeans_cluster(args,user_scope=args.user_scope, top_k=args.llm_seed_k)  # 复用llm_seed_k参数作为top_k值（仅当user_scope=top_k时有效）


def run_mvsc(args):
    CoRegMVSC.run_mvsc_and_save(
        data_path=args.data_path,
        n_clusters=args.mvsc_n_clusters,
        mvsc_lambda=args.mvsc_mvsc_lambda,
        save_path=args.mvsc_save_path,
        user_scope=args.user_scope,
        top_k=args.llm_seed_k  # 复用llm_seed_k参数作为top_k值（仅当user_scope=top_k时有效
    )


def run_diffusion(args):
    utils.init_seed(args.seed)
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，自动切换到CPU")
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    # 加载全量数据集（确保DiffusionAgent能获取完整的交互数据）
    train_data = utils.load_json(f"{args.data_path}/train.json")
    dev_data = utils.load_json(f"{args.data_path}/dev.json")
    test_data = utils.load_json(f"{args.data_path}/test.json")
    
    # 构建全局用户/物品映射表（保证训练/评估阶段ID映射一致）
    all_users = list(set([item['user_id'] for item in train_data + dev_data + test_data]))
    all_answers = list(set([item['answer_id'] for item in train_data + dev_data + test_data]))
    user2idx = utils.build_id2idx_mapping(all_users)
    answer2idx = utils.build_id2idx_mapping(all_answers)
    
    corpus = {
        "n_users": len(user2idx),
        "n_answers": len(answer2idx)
    }
    print(f"Corpus stats: {len(all_users)} users, {len(all_answers)} answers")
    
    # 初始化模型参数
    model_args = {
        "emb_size": args.emb_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "device": device,
        "dropout": 0.2
    }
    model = DiffusionAgent.DiffusionAgent(model_args, corpus).to(device)
    
    # 初始化运行器参数
    runner_args = {
        "epoch": args.epoch,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "device": device,
        "patience": args.patience,
        "save_every_n_epochs": args.save_every_n_epochs,
        "save_dir": args.save_dir,
        "emb_size": args.emb_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "max_his": 20
    }
    runner = DiffusionAgent.SimpleRunner(runner_args)
    
    if args.diffusion_mode == 'train':
        print("Training Enhanced Diffusion model (without seed data)...")
        # 训练阶段：传递原始数据，超图由DiffusionAgent内部构建
        model = runner.train(
            model=model,
            train_data=train_data,
            dev_data=dev_data,
            user2idx=user2idx,
            answer2idx=answer2idx
        )
    elif args.diffusion_mode == 'eval':
        print("Evaluating Enhanced Diffusion model (with seed data)...")
        # 推理阶段：加载seed数据，超图由DiffusionAgent内部构建
        seed_data = utils.load_json(args.diffusion_seed_path)
        metrics = runner.evaluate_from_checkpoint(
            model=model,
            test_data=test_data,
            seed_data=seed_data,
            user2idx=user2idx,
            answer2idx=answer2idx,
            load_ckpt_path=args.load_ckpt_path,
            save_path=args.diffusion_pred_save_path
        )
        print(f"Final Metrics: {utils.format_metric(metrics)}")


if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == 'llm':
        run_llm(args)
    elif args.mode == 'diffusion':
        run_diffusion(args)
    elif args.mode == 'mvsc':
        run_mvsc(args)
    elif args.mode == 'kmeans':
        run_kmeans(args)