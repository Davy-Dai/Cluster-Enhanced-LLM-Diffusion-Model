# -*- coding: UTF-8 -*-
import json
from tqdm import tqdm
from typing import Dict, List
from openai import OpenAI
import utils
from collections import defaultdict
import numpy as np

class LLMAgent:
    """LLM代理：基于聚类结果生成用户点击决策种子"""
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config
        self.gamma = llm_config.get('gamma', 0.05)  # 时间衰减因子
        self.prompts = self._build_prompt_template()
        self.client = OpenAI(
            api_key=llm_config.get('api_key'),
            base_url=llm_config.get('base_url'),
        )
    
    def _build_prompt_template(self) -> Dict:
        """构建适配新用户画像/簇画像的Prompt模板（英文）"""
        return {
            "prompt": """Act as a social media user agent. Based on the provided user profile and answer content, determine whether to click the answer using a balanced decision framework.
=== OUTPUT REQUIREMENTS ===
Return JSON format ONLY with exactly three fields:
{{
    "score": 0-100,
    "decision": "yes" or "no",
    "reasoning": "xxxx",
    "confidence": 1-5
}}
The JSON response should conclude your decision, reasoning, and confidence score. And Do not include any other text or comments.
The "score" field: the total weighted score (0-100) calculated from the Stage 2 evaluation.
The "reasoning" field must be **no more than 100 words**
The "confidence" field: 5=very certain, 4=certain, 3=neutral, 2=uncertain, 1=very uncertain
=== DECISION STRATEGY ===
- Consider edge cases: Even if a answer doesn't perfectly match, consider if it could be interesting
- Balance quality and quantity: Aim for meaningful recommendations while avoiding empty results
- Use relative scoring: Compare answers within the user's context rather than absolute standards
=== MULTI-STAGE DECISION FRAMEWORK ===
STAGE 1: INITIAL ASSESSMENT (Quick Filter)
- If the answer is clearly irrelevant or inappropriate: "no"
- If the answer has any potential relevance: proceed to Stage 2
STAGE 2: COMPREHENSIVE EVALUATION (Total 100 points)
Evaluate using these weighted factors:
1. TOPIC INTEREST (Weight: 10%):
- User followed topics vs answer related topics: 0-100 points
2. ANSWER INTEREST (Weight: 40%):
- Based on user query cosine similarity (time-decayed): 0-100 points
3. CLICK PROBABILITY (Weight: 35%):
- Compare clicked vs non-clicked historical answer similarities: 0-100 points
4. PERSONALITY & SOCIAL BEHAVIOR (Weight: 15%):
- Gender, activity level, followers, engagement: 0-100 points
STAGE 3: DECISION LOGIC
- Score 70+ points: Strong "yes"
- Score 50-69 points: Moderate "yes"
- Score 30-49 points: Weak "yes"
- Score 0-29 points: "no"
ADAPTIVE THRESHOLDS:
- Active users: Standard threshold
- Inactive users: Higher threshold, more selective
Start your analysis now:
=== USER PROFILE ===
- Gender: {gender}
- User behavior stats: {user_info}
- Followed topics: {followed_topics}
=== ANSWER CONTENT ===
- Related topics: {answer_topics}
=== TIME-DECAYED SIMILARITY METRICS ===
- Query similarity: {query_sim_str}
- Clicked answer similarity: {clicked_sim:.6f}
- Non-clicked answer similarity: {non_clicked_sim:.6f}
Begin analysis.
""",
        }
    
    def _calculate_cosine_similarity(self, vec1: list, vec2: list) -> float:
        """计算两个向量的余弦相似度"""
        return utils.cosine_similarity(vec1, vec2)
    
    def _time_decay_weight(self, time_diff: float) -> float:
        """
        时间衰减函数：越近的时间权重越大
        :param time_diff: 时间差（当前时间 - 历史时间），单位可自定义
        :return: 衰减权重
        """
        return np.exp(-self.gamma * time_diff)
    
    def _calculate_weighted_history_similarity(
        self, 
        user_history: List[Dict], 
        current_answer_vec: list, 
        answer_info_dict: Dict,
        click_filter: int
    ) -> float:
        """
        计算用户历史交互内容与当前answer的时间加权余弦相似度
        :param user_history: 用户历史交互列表（按时间排序）
        :param current_answer_vec: 当前answer的向量
        :param answer_info_dict: answer信息字典
        :param click_filter: 筛选条件（1=仅点击，0=仅未点击）
        :return: 加权平均余弦相似度
        """
        filtered_history = [h for h in user_history if h['is_click'] == click_filter]
        if not filtered_history:
            return 0.0
        
        # 使用最新时间作为当前基准
        current_time = max(h.get('timestamp', 0) for h in filtered_history)
        
        total_weight = 0.0
        weighted_sim_sum = 0.0
        
        for hist in filtered_history:
            answer_id = hist['answer_id']
            hist_answer = answer_info_dict.get(answer_id, {})
            hist_vec = hist_answer.get('answer_info', [])
            
            if not isinstance(hist_vec, list) or len(hist_vec) != 64:
                continue
            
            # 计算余弦相似度
            sim = self._calculate_cosine_similarity(hist_vec, current_answer_vec)
            
            # 计算时间衰减权重
            hist_time = hist.get('timestamp', 0)
            time_diff = current_time - hist_time
            weight = self._time_decay_weight(time_diff)
            
            weighted_sim_sum += sim * weight
            total_weight += weight
        
        return weighted_sim_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_weighted_query_similarity(
        self, 
        user_queries: List, 
        current_answer_vec: list
    ) -> float:
        """
        计算用户Query与当前answer的时间衰减余弦相似度
        :param user_queries: 用户Query列表
        :param current_answer_vec: 当前answer的向量
        :return: 加权平均余弦相似度
        """
        if not isinstance(user_queries, list) or len(user_queries) == 0:
            return 0.0
        
        # 获取Query时间（假设query中有timestamp字段，否则使用索引作为时间顺序）
        current_time = len(user_queries)  # 使用索引作为时间基准
        
        total_weight = 0.0
        weighted_sim_sum = 0.0
        
        for idx, query in enumerate(user_queries):
            query_vec = query.get('content', []) if isinstance(query, dict) else []
            if not isinstance(query_vec, list) or len(query_vec) != 64:
                continue
            
            # 计算余弦相似度
            sim = self._calculate_cosine_similarity(query_vec, current_answer_vec)
            
            # 计算时间衰减权重（越新的query权重越大）
            time_diff = current_time - idx
            weight = self._time_decay_weight(time_diff)
            
            weighted_sim_sum += sim * weight
            total_weight += weight
        
        return weighted_sim_sum / total_weight if total_weight > 0 else 0.0
    
    def _format_query_similarity(self, query_sim: float) -> str:
        """格式化Query相似度为字符串"""
        return f"Time-decayed cosine similarity between user queries and answer: {query_sim:.6f}"
    
    def _array_to_str(self, arr: list) -> str:
        """将数组转为逗号分隔的字符串"""
        if not isinstance(arr, list) or len(arr) == 0:
            return "None"
        return ', '.join([f"{x:.6f}" for x in arr])
    
    def _list_to_str(self, lst: list) -> str:
        """将列表转为逗号分隔的字符串"""
        if not isinstance(lst, list) or len(lst) == 0:
            return "None"
        return ', '.join([str(x) for x in lst])
    
    def _dict_to_str(self, dct: dict) -> str:
        """将字典转为可读性强的字符串（key:value）"""
        if not isinstance(dct, dict) or len(dct) == 0:
            return "None"
        return ', '.join([f"{k}:{v}" for k, v in dct.items()])
    
    def _build_user_profile_str(self, user_profile: Dict) -> Dict:
        """拼接用户画像字符串"""
        # 处理用户行为信息
        user_info = user_profile.get('user_info', {})
        num_topics = user_info.get('num_topics_followed', 0) or 0
        num_answers = user_info.get('num_answers', 0) or 0
        num_likes = user_info.get('num_likes_received', 0) or 0
        num_followers = user_info.get('num_followers', 0) or 0
        # 格式化为易读字符串
        user_info_str = (
            f"Total topics followed: {num_topics}, "
            f"Total answers posted: {num_answers}, "
            f"Total likes received: {num_likes}, "
            f"Total followers: {num_followers}"
        )
        # 处理关注的话题（数组转字符串）
        followed_topics = user_profile.get('topics', [])
        followed_topics_str = self._list_to_str(followed_topics)
        
        return {
            "gender": user_profile.get('gender', 'Unknown').lower(),
            "user_info": user_info_str,
            "followed_topics": followed_topics_str
        }
    
    def _build_cluster_profile_str(self, cluster_info: Dict) -> Dict:
        """拼接簇画像字符串（适配簇级别的Prompt）"""
        # 处理簇的用户行为信息（平均值字典转字符串）
        avg_user_info = cluster_info.get('avg_user_info', {})
        num_topics_avg = avg_user_info.get('num_topics_followed', 0.0) or 0.0
        num_answers_avg = avg_user_info.get('num_answers', 0.0) or 0.0
        num_likes_avg = avg_user_info.get('num_likes_received', 0.0) or 0.0
        num_followers_avg = avg_user_info.get('num_followers', 0.0) or 0.0
        # 处理簇的关注话题（字典转字符串，topic:数量）
        followed_topics = cluster_info.get('topics', {})
        followed_topics_str = self._dict_to_str(followed_topics)
        
        user_info_str = (
            f"Total topics followed: {num_topics_avg}, "
            f"Total answers posted: {num_answers_avg}, "
            f"Total likes received: {num_likes_avg}, "
            f"Total followers: {num_followers_avg}"
        )
        return {
            "gender": cluster_info.get('dominant_gender', 'Unknown').lower(),
            "user_info": user_info_str,
            "followed_topics": followed_topics_str
        }
    
    def _build_answer_info_str(self, answer_info: Dict) -> Dict:
        """拼接回答信息字符串"""
        answer_topics = answer_info.get('topics', [])
        answer_topics_str = self._list_to_str(answer_topics)
        
        return {
            "answer_topics": answer_topics_str
        }
    
    def _build_prompt(
        self, 
        user_profile: Dict, 
        answer_info: Dict, 
        query_sim: float,
        clicked_sim: float,
        non_clicked_sim: float
    ) -> str:
        """
        构建完整Prompt（用户级别）
        :param user_profile: 用户画像
        :param answer_info: 回答信息
        :param query_sim: Query相似度
        :param clicked_sim: 点击历史相似度
        :param non_clicked_sim: 未点击历史相似度
        :return: 完整Prompt
        """
        user_profile_str = self._build_user_profile_str(user_profile)
        answer_info_str = self._build_answer_info_str(answer_info)
        query_sim_str = self._format_query_similarity(query_sim)
        
        return self.prompts['prompt'].format(
            **user_profile_str,
            **answer_info_str,
            query_sim_str=query_sim_str,
            clicked_sim=clicked_sim,
            non_clicked_sim=non_clicked_sim
        )
    
    def _build_cluster_prompt(
        self,
        cluster_info: Dict,
        answer_info: Dict,
        query_sim: float,
        clicked_sim: float,
        non_clicked_sim: float
    ) -> str:
        """
        构建完整Prompt（簇级别）
        :param cluster_info: 簇画像信息
        :param answer_info: 回答信息
        :param query_sim: 簇Query中心与answer的相似度
        :param clicked_sim: 簇Click-Answer中心与answer的相似度
        :param non_clicked_sim: 簇NonClick-Answer中心与answer的相似度
        :return: 完整Prompt
        """
        cluster_profile_str = self._build_cluster_profile_str(cluster_info)
        answer_info_str = self._build_answer_info_str(answer_info)
        query_sim_str = self._format_query_similarity(query_sim)
        
        return self.prompts['prompt'].format(
            **cluster_profile_str,
            **answer_info_str,
            query_sim_str=query_sim_str,
            clicked_sim=clicked_sim,
            non_clicked_sim=non_clicked_sim
        )
    
    def _call_llm_api(self, prompt: str) -> Dict:
        """调用LLM API（增加JSON解析异常捕获）"""
        try:
            completion = self.client.chat.completions.create(
                model=self.llm_config.get('model_name'),
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant. Please output valid JSON only.'},
                    {'role': 'user', 'content': prompt}
                ],
                response_format={"type": "json_object"},
                temperature=self.llm_config.get('temperature', 0.7),
                max_tokens=self.llm_config.get('max_tokens', 512),
                timeout=self.llm_config.get('timeout', 120)
            )
            response_content = completion.choices[0].message.content.strip()
            try:
                return json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f"LLM返回JSON格式错误: {e}, 响应内容: {response_content}")
                return {"decision": "no", "reasoning": f"JSON parse error: {str(e)}", "confidence": 3}
        except Exception as e:
            print(f"LLM API调用错误: {e}")
            return {"decision": "no", "reasoning": f"API error: {str(e)}", "confidence": 3}
    
    def _parse_response(self, response: Dict) -> Dict:
        """解析LLM响应"""
        try:
            decision = response.get('decision', 'no').lower()
            return {
                "score": response.get('score'),
                "click": 1 if decision == 'yes' else 0,
                "reasoning": response.get('reasoning', 'No reasoning provided'),
                "confidence": response.get('confidence', 3)
            }
        except Exception as e:
            print(f"响应解析错误: {e}")
            return {"click": 0, "reasoning": f"Parse error: {str(e)}", "confidence": 3}
    
    def _build_user_history(self, train_data: List) -> Dict:
        """
        构建用户历史交互字典（按时间排序）
        :param train_data: 训练数据
        :return: user_id -> 历史交互列表（按时间排序）
        """
        user_history = defaultdict(list)
        for item in train_data:
            user_id = item['user_id']
            user_history[user_id].append(item)
        
        # 按时间排序每个用户的历史
        for user_id in user_history:
            user_history[user_id].sort(key=lambda x: x.get('timestamp', 0))
        
        return user_history
    
    def generate_seed_clicks(self, data_path: str, seed_k: int, save_path: str, cluster_path: str = None):
        """
        生成种子点击结果并保存（基于聚类结果或Top-K用户）
        :param data_path: 数据集目录
        :param seed_k: Top-K活跃用户数量 / 簇选择数量
        :param save_path: 种子结果保存路径
        :param cluster_path: 聚类结果JSON路径（可选，若提供则基于聚类生成）
        """
        # 加载数据
        train_data = utils.load_json(f"{data_path}/train.json")
        test_data = utils.load_json(f"{data_path}/test.json")
        user_info = utils.load_json(f"{data_path}/user_info.json")
        answer_info = utils.load_json(f"{data_path}/answer_info.json")
        
        user_profile_dict = {item['user_id']: item for item in user_info}
        answer_info_dict = {item['answer_id']: item for item in answer_info}
        user_history_dict = self._build_user_history(train_data)
        all_answer_ids = list(answer_info_dict.keys())
        
        seed_results = []
        
        # 分支1：基于聚类结果生成
        if cluster_path:
            print(f"Loading cluster results from: {cluster_path}")
            cluster_results = utils.load_json(cluster_path)
            
            for answer_id in tqdm(all_answer_ids, desc="Processing answers (cluster)"):
                answer_info = answer_info_dict.get(answer_id, {})
                current_answer_vec = answer_info.get('answer_info', [])
                if not isinstance(current_answer_vec, list) or len(current_answer_vec) != 64:
                    current_answer_vec = [0.0] * 64
                
                # 计算每个簇与当前answer的相似度
                cluster_sims = []
                for cluster in cluster_results:
                    c_id = cluster['cluster_id']
                    center_query = cluster['center_query']
                    center_click = cluster['center_click_answer']
                    center_nonclick = cluster['center_nonclick_answer']
                    
                    # 计算三个维度的相似度
                    query_sim = self._calculate_cosine_similarity(center_query, current_answer_vec)
                    clicked_sim = self._calculate_cosine_similarity(center_click, current_answer_vec)
                    non_clicked_sim = self._calculate_cosine_similarity(center_nonclick, current_answer_vec)
                    avg_sim = (query_sim + clicked_sim + non_clicked_sim) / 3
                    
                    cluster_sims.append((c_id, avg_sim, query_sim, clicked_sim, non_clicked_sim, cluster))
                
                # 选择Top-K相似簇
                cluster_sims.sort(key=lambda x: x[1], reverse=True)
                selected_clusters = cluster_sims[:seed_k]
                
                # 对每个选中的簇进行推理
                for c_id, _, query_sim, clicked_sim, non_clicked_sim, cluster in selected_clusters:
                    # 构建簇级别Prompt
                    c_info = {
                        'dominant_gender': cluster['dominant_gender'],
                        'avg_user_info': cluster['avg_user_info'],
                        'topics': cluster['topics']
                    }
                    prompt = self._build_cluster_prompt(c_info, answer_info, query_sim, clicked_sim, non_clicked_sim)
                    llm_response = self._call_llm_api(prompt)
                    parsed_result = self._parse_response(llm_response)
                    
                    # 结果复用给簇内所有用户
                    for user_id in cluster['user_ids']:
                        seed_results.append({
                            "user_id": user_id,
                            "answer_id": answer_id,
                            "score": parsed_result['score'],
                            "seed_click": parsed_result['click'],
                            "reasoning": parsed_result['reasoning'],
                            "confidence": parsed_result['confidence'],
                            "query_similarity": query_sim,
                            "clicked_similarity": clicked_sim,
                            "non_clicked_similarity": non_clicked_sim,
                            "cluster_id": c_id
                        })
        
        # 分支2：原有Top-K活跃用户种子选择
        else:
            print("Running Top-K active user seed selection...")
            top_k_users = utils.get_top_k_active_users(
                data=train_data,
                k=seed_k,
                user_info_data=user_info
            )
            print(f"Selected Top-{seed_k} active users: {len(top_k_users)} users")
            
            for answer_id in tqdm(all_answer_ids, desc="Processing answers (top-k)"):
                answer_info = answer_info_dict.get(answer_id, {})
                current_answer_vec = answer_info.get('answer_info', [])
                if not isinstance(current_answer_vec, list) or len(current_answer_vec) != 64:
                    current_answer_vec = [0.0] * 64
                
                for user_id in top_k_users:
                    user_profile = user_profile_dict.get(user_id, {})
                    user_history = user_history_dict.get(user_id, [])
                    user_queries = user_profile.get('query', [])
                    
                    clicked_sim = self._calculate_weighted_history_similarity(
                        user_history, current_answer_vec, answer_info_dict, click_filter=1
                    )
                    non_clicked_sim = self._calculate_weighted_history_similarity(
                        user_history, current_answer_vec, answer_info_dict, click_filter=0
                    )
                    query_sim = self._calculate_weighted_query_similarity(user_queries, current_answer_vec)
                    
                    prompt = self._build_prompt(
                        user_profile, answer_info, query_sim, clicked_sim, non_clicked_sim
                    )
                    llm_response = self._call_llm_api(prompt)
                    parsed_result = self._parse_response(llm_response)
                    
                    seed_results.append({
                        "user_id": user_id,
                        "answer_id": answer_id,
                        "score": parsed_result['score'],
                        "seed_click": parsed_result['click'],
                        "reasoning": parsed_result['reasoning'],
                        "confidence": parsed_result['confidence'],
                        "query_similarity": query_sim,
                        "clicked_similarity": clicked_sim,
                        "non_clicked_similarity": non_clicked_sim
                    })
        
        # 保存最终种子结果
        utils.save_json(seed_results, save_path)
        print(f"Seed clicks saved to {save_path} (total: {len(seed_results)} records)")
        return seed_results