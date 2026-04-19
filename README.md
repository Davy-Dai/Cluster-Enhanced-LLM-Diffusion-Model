# 聚类增强的大语言 - 扩散模型智能体社会仿真系统

## Cluster-Enhanced LLM-Diffusion Agent for Social Simulation

### 项目介绍

本项目为**中山大学本科毕业设计**代码实现，对应论文《聚类增强的基于大语言扩散模型的智能体社会仿真模拟》，作者：戴骏腾。

项目融合 **多视图谱聚类 (CoRegMVSC)** 、 **K-Means 聚类** 、**大语言模型 (LLM) 智能体**与 **扩散模型 (Diffusion)** ，实现大规模社交网络信息扩散仿真，基于知乎 ZhihuRec 数据集完成模型训练、推理与全流程实验。

### 参考项目和论文

本项目部分框架参考：[Social-Simulation-for-Information-Diffusion](https://github.com/lixinyi22/Social-Simulation-for-Information-Diffusion)

本项目的DiffusionAgent模块实现参考项目和论文：

·***FuxiCTR***：[reczoo/FuxiCTR: A configurable, tunable, and reproducible library for CTR prediction https://fuxictr.github.io](https://github.com/reczoo/FuxiCTR)

·双通道编码框架论文：***《Modeling Information Diffusion With Sequential Interactive Hypergraphs》***

### 数据集来源

实验数据集：**ZhihuRec 知乎推荐数据集**

下载地址：[清华云盘 ZhihuRec](https://cloud.tsinghua.edu.cn/d/d6c045c55aa14bb39ebc/)

### 项目文件架构

```
Cluster-Enhanced-LLM-Diffusion-Model/
├── main.py                          # 主函数
├── LLMAgent.py                      # LLMAgent
├── DiffusionAgent.py                # DiffusionAgent
├── utils.py   
├── clustering/                      # 聚类算法模块
│   ├── CoRegMVSC.py   
│   └── Kmeans.py  
├── data/                            # 数据集根目录
│   └── zhihu/                       # ZhihuRec预处理后数据集
├── diffusion_ckpt/                  # 扩散模型权重保存目录
│   └── best_model.pth               # 训练200轮的最优模型权重
├── others/                          # 其他辅助实验代码
│   ├── run_main.py                  # 实验批量运行代码
│   ├── test.py                      # 测试生成的llm_seed种子文件ACC等值，需要放在main.py同级文件
│   ├── test_cluster.py		     # 测试聚类是否成功分组，需要放在和聚类生成的中间文件同一文件夹
│   └── data_code/                   # 数据集处理代码
│       ├── data.py                  # 将CSV格式文件转成本实验需要的JSON格式文件
│       ├── zhihu_1M.py              # 按照数据集要求读取1M数据
│       ├── zhihu_small.py           # 将数据集从1M压缩到10K
│       └── zhihu_sort.py            # 将压缩后的数据集重新排序
├── requirements.txt   
└── README.md  
```

### 环境依赖

```
#CPU环境即可，若有GPU将环境设备调成CUDA
#Python 3.10
torch==1.13.1
numpy==1.26.4
pandas==2.2.3
tqdm==4.66.4
openai==2.30.0
scipy==1.11.4
scikit-learn==1.7.2
```

### 运行流程和命令

整体流程：**聚类生成用户分簇 → LLM 生成种子节点 → 扩散模型训练 → 扩散模型推理预测**

关于运行的超参数设定详情请见——`main.py`文件，根据需要进行更改。

#### 1. 生成 LLM 种子节点文件

支持两种用户范围： **全量用户聚类** 、 **Top-K 活跃用户聚类** ；支持两种聚类算法：`CoRegMVSC` / `KMeans`

##### 1.1 全量用户聚类

**Step1 执行聚类算法**

```
# CoRegMVSC 聚类
python main.py --mode mvsc --data_path ./data/zhihu --mvsc_n_clusters 1500 --mvsc_save_path ./data/mvsc_results_1500.json

# KMeans 聚类
python main.py --mode kmeans --data_path ./data/zhihu  --kmeans_n_clusters 10   --kmeans_save_path ./data/kmeans_results_10.json
```

**Step2 生成 LLM 种子节点**

种子文件命名说明：`llm_seed_results_mvsc1000_20`表示对全用户分成1000类，然后激活数量为20个簇。

LLM调用有关参数说明（参考main.py中的参数设定）

·`--llm_base_url`：调正成你所用的API接口 `base_url`

·`--llm_api_key #YOUR-API-KEY`：将 `#YOUR-API-KEY`换成你的API接口的KEY

```
# Top-K 无聚类（基线方法）
python main.py --mode llm --data_path ./data/zhihu --llm_seed_k 100 --llm_api_key #YOUR-API-KEY --llm_save_path  ./data/llm_seed_results_topk_100.json 

# MVSC 聚类驱动LLM
python main.py --mode llm --data_path ./data/zhihu --llm_cluster_path ./data/mvsc_results_1000.json --llm_seed_k 20 --llm_api_key #YOUR-API-KEY --llm_save_path  ./data/llm_seed_results_mvsc1000_20.json

# KMeans 聚类驱动LLM
python main.py --mode llm --data_path ./data/zhihu --llm_cluster_path ./data/kmeans_results_10.json --llm_seed_k 10 --llm_api_key #YOUR-API-KEY --llm_save_path  ./data/llm_seed_results_kmeans10_10.json
```

##### 1.2 Top-K 活跃用户聚类

种子文件命名说明：`llm_seed_results_mvsc2_10_5`表示激活用户数量topk=10，对这个10个用户划分成5个聚类簇。

```
# MVSC 活跃用户聚类 + LLM

#指定对于top_k=n的用户聚类
python main.py --mode mvsc --data_path ./data/zhihu --user_scope top_k --llm_seed_k 1000 --mvsc_n_clusters 10 --mvsc_save_path ./data/mvsc1000_results_10.json
#对于上一步聚类的n要作为这一步的llm_seed_k的值，同时更改--llm_cluster_path对应文件
python main.py --mode llm --data_path ./data/zhihu --llm_cluster_path ./data/mvsc1000_results_10.json --llm_seed_k 10 --llm_api_key #YOUR-API-KEY --llm_save_path  ./data/llm_seed_results_mvsc2_1000_10.json

# KMeans 活跃用户聚类 + LLM

##指定对于top_k=n的用户聚类
python main.py --mode kmeans --data_path ./data/zhihu  --user_scope top_k --llm_seed_k 1000 --kmeans_n_clusters 10 --kmeans_save_path ./data/kmeans1000_results_10.json
#对于上一步聚类的n要作为这一步的llm_seed_k的值
python main.py --mode llm --data_path ./data/zhihu --llm_cluster_path ./data/kmeans1000_results_10.json --llm_seed_k 10 --llm_api_key  #YOUR-API-KEY--llm_save_path  ./data/llm_seed_results_kmeans2_1000_10.json
```

#### 2.扩散模型训练

```
python main.py --mode diffusion --data_path ./data/zhihu --diffusion_mode train --epoch 200  --batch_size 128 --lr 1e-4 --save_every_n_epochs 10 --save_dir ./diffusion_ckpt
```

#### 3.扩散模型推理预测

```
# KMeans 种子驱动预测
python main.py --mode diffusion --data_path ./data/zhihu --diffusion_mode eval --load_ckpt_path ./diffusion_ckpt/best_model.pth --diffusion_seed_path ./data/llm_seed_results_kmeans2_1000_10.json --diffusion_pred_save_path ./data/diffusion_pred_results.json

# MVSC 种子驱动预测
python main.py --mode diffusion --data_path ./data/zhihu  --diffusion_mode eval --diffusion_seed_path ./data/llm_seed_results_mvsc2_1000_10.json --load_ckpt_path ./diffusion_ckpt/best_model.pth --diffusion_pred_save_path ./data/diffusion_pred_results.json

# Top-K 基线预测
python main.py --mode diffusion --data_path ./data/zhihu --device cuda --diffusion_mode eval --load_ckpt_path ./diffusion_ckpt/best_model.pth --diffusion_seed_path ./data/llm_seed_results_topk_200.json --diffusion_pred_save_path ./data/diffusion_pred_results.json
```

### 数据文件规范与字段说明

#### 1. 输入数据集格式

##### 输入文件结构

```
data/zhihu
├── train.json         
├── dev.json       
├── test.json   
├── answer_info.json
└── user_info.json           
```

##### 1.1 交互数据集：train.json/dev.json/test.json

用户 - 回答交互行为数据，核心训练 / 验证 / 测试集

```
{
  "user_id": 123456,
  "answer_id": 789012,
  "time": "1620000000",
  "is_click": 1
}
```

表格

| 字段名    | 类型 | 含义说明                       | 取值范围   |
| --------- | ---- | ------------------------------ | ---------- |
| user_id   | int  | 用户唯一 ID                    | 非负整数   |
| answer_id | int  | 回答唯一 ID                    | 非负整数   |
| time      | str  | 行为时间戳（字符串格式）       | 数字字符串 |
| is_click  | int  | 点击标签：1 = 点击，0 = 仅曝光 | {0, 1}     |

##### 1.2 回答特征集：answer_info.json

回答文本向量与话题特征

```
{
  "answer_id": 789012,
  "answer_info": [0.123, 0.456, 0.789],
  "topics": [101, 202, 303]
}
```

| 字段名      | 类型        | 含义说明                    | 取值范围      |
| ----------- | ----------- | --------------------------- | ------------- |
| answer_id   | int         | 回答唯一 ID（与交互集对齐） | 非负整数      |
| answer_info | list[float] | 64 维文本词向量均值         | 64 维浮点数组 |
| topics      | list[int]   | 回答所属话题 ID 列表        | 非负整数数组  |

##### 1.3 用户特征集：user_info.json

用户画像、行为统计与查询历史

```
{
  "user_id": 123456,
  "gender": "female",
  "user_info": {
    "num_topics_followed": 5,
    "num_answers": 10,
    "num_likes_received": 100,
    "num_followers": 200
  },
  "topics": [101, 404, 505],
  "query": [{"query_id": 0, "content": [0.234, 0.567], "time": "1620001000"}]
}
```

| 字段名    | 类型       | 含义说明             | 取值范围            |
| --------- | ---------- | -------------------- | ------------------- |
| user_id   | int        | 用户唯一 ID          | 非负整数            |
| gender    | str        | 用户性别             | male/female/unknown |
| user_info | dict       | 用户行为统计特征     | 数值型字典          |
| topics    | list[int]  | 用户关注话题 ID 列表 | 非负整数数组        |
| query     | list[dict] | 用户历史查询记录     | 结构化数组          |

#### 2. 聚类中间文件字段说明

聚类输出 JSON 文件，描述用户分簇特征

| 字段名              | 类型        | 含义说明                   |
| ------------------- | ----------- | -------------------------- |
| cluster_id          | int         | 聚类簇 ID（从 0 开始递增） |
| user_ids            | list[int]   | 簇内用户 ID 列表           |
| cluster_size        | int         | 簇内用户数量               |
| center_query        | list[float] | 查询特征空间聚类中心       |
| center_click_answer | list[float] | 点击行为特征聚类中心       |
| dominant_gender     | str         | 簇内主导性别               |
| avg_user_info       | dict        | 簇内用户平均行为统计       |
| topics              | dict        | 簇内话题分布统计           |

#### 3. LLM 中间文件字段说明

LLM 种子节点输出文件，扩散模型输入

| 字段名     | 类型 | 含义说明                     | 取值范围   |
| ---------- | ---- | ---------------------------- | ---------- |
| user_id    | int  | 用户唯一 ID                  | 非负整数   |
| answer_id  | int  | 回答唯一 ID                  | 非负整数   |
| seed_click | int  | LLM 预测点击标签（扩散种子） | {0, 1}     |
| reasoning  | str  | LLM 决策推理文本（可解释性） | 文本字符串 |

## 版权说明

本项目可用于**学术研究，若借鉴项目请引用。**

联系作者——daijt3@mail2.sysu.edu.cn或704048706@qq.com。
