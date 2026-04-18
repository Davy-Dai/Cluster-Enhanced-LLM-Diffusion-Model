# -*- coding: UTF-8 -*-
import math
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any
import utils


# ============================== 复用layers.py的核心层定义 ==============================
class Fusion(nn.Module):
    def __init__(self, emb_dim, dropout=0.2):
        super(Fusion, self).__init__()
        self.gate_layer = nn.Linear(2 * emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_static = nn.LayerNorm(emb_dim)
        self.norm_dynamic = nn.LayerNorm(emb_dim)
        self.init_weights()

    def init_weights(self):
        # 修复：避免重复初始化覆盖（先kaiming再xavier会覆盖，保留合理的初始化策略）
        nn.init.xavier_normal_(self.gate_layer.weight)
        if self.gate_layer.bias is not None:
            nn.init.constant_(self.gate_layer.bias, 0.0)

    def forward(self, u_static, u_dynamic):
        # 输入维度：[batch/num, emb_dim]
        u_static = self.norm_static(u_static)
        u_dynamic = self.norm_dynamic(u_dynamic)
        concat = torch.cat([u_static, u_dynamic], dim=-1) * 1.5 - 0.25  # 控制输出范围
        
        gate = torch.sigmoid(self.gate_layer(concat))
        u_fused = gate * u_static + (1 - gate) * u_dynamic
        return u_fused  # 输出维度：[batch/num, emb_dim]


class MultiFusion(nn.Module):
    def __init__(self, emb_dim, dropout=0.2):
        super(MultiFusion, self).__init__()
        self.gate_layer = nn.Linear(3 * emb_dim, 2)
        self.dropout = nn.Dropout(dropout)
        self.emb_dim = emb_dim
        self.norm_input1 = nn.LayerNorm(emb_dim)
        self.norm_input2 = nn.LayerNorm(emb_dim)
        self.norm_input3 = nn.LayerNorm(emb_dim)
        self.init_weights()
        
    def init_weights(self):
        # 修复：避免重复初始化覆盖
        nn.init.xavier_normal_(self.gate_layer.weight)
        if self.gate_layer.bias is not None:
            nn.init.constant_(self.gate_layer.bias, 0.0)
        
    def forward(self, input1, input2, input3):
        # 输入维度：[num, emb_dim]（num为用户/物品数量）
        input1 = self.norm_input1(input1)
        input2 = self.norm_input2(input2)
        input3 = self.norm_input3(input3)
        
        concat = torch.cat([input1, input2, input3], dim=-1) * 1.5 - 0.25  # [num, 3*emb_dim]
        
        gates = torch.sigmoid(self.gate_layer(concat))  # [num, 2]
        # 修复：expand维度对齐（兼容任意前置维度）
        gate1 = gates[..., 0:1].expand(*gates.shape[:-1], self.emb_dim)
        gate2 = gates[..., 1:2].expand(*gates.shape[:-1], self.emb_dim)
        gate3 = (1 - gates[..., 0:1] - gates[..., 1:2]).expand(*gates.shape[:-1], self.emb_dim)
        
        fused_output = gate1 * input1 + gate2 * input2 + gate3 * input3  # [num, emb_dim]
        return fused_output


class MultiHeadAttention(nn.Module):
    """改进版多头注意力：修复mask维度匹配问题"""
    def __init__(self, d_model, n_heads, kq_same=False, bias=True, attention_d=-1):
        super().__init__()
        self.d_model = d_model
        self.h = n_heads
        self.attention_d = attention_d if attention_d > 0 else self.d_model
        assert self.attention_d % self.h == 0, "attention_d must be divisible by n_heads"
        self.d_k = self.attention_d // self.h
        self.kq_same = kq_same

        if not kq_same:
            self.q_linear = nn.Linear(d_model, self.attention_d, bias=bias)
        self.k_linear = nn.Linear(d_model, self.attention_d, bias=bias)
        self.v_linear = nn.Linear(d_model, self.attention_d, bias=bias)

    def head_split(self, x):
        # 输入：[batch, seq_len, d_model] 输出：[batch, n_heads, seq_len, d_k]
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        return x.view(*new_x_shape).transpose(-2, -3)

    def forward(self, q, k, v, mask=None):
        origin_shape = q.size()  # [batch, seq_len, d_model]
        batch_size, seq_len = origin_shape[0], origin_shape[1]

        # 线性变换 + 分桶
        if not self.kq_same:
            q = self.head_split(self.q_linear(q))  # [batch, h, seq_len, d_k]
        else:
            q = self.head_split(self.k_linear(q))
        k = self.head_split(self.k_linear(k))
        v = self.head_split(self.v_linear(v))

        # 注意力计算
        output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)
        # 合并head + 恢复维度
        output = output.transpose(-2, -3).reshape(batch_size, seq_len, self.attention_d)
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, d_k, mask=None):
        # 输入：q/k/v [batch, h, seq_len, d_k] | mask [batch, 1, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [batch, h, seq_len, seq_len]
        if mask is not None:
            # 修复：mask维度广播兼容
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 扩展head维度
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 数值稳定性优化
        scores = scores - scores.max(dim=-1, keepdim=True)[0]
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)
        
        output = torch.matmul(attn, v)  # [batch, h, seq_len, d_k]
        return output


class AttLayer(nn.Module):
    """通用注意力层：修复维度计算逻辑"""
    def __init__(self, in_dim, att_dim):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)
        # 初始化
        nn.init.xavier_normal_(self.w.weight)
        nn.init.normal_(self.h, std=0.01)

    def forward(self, infeatures):
        # 输入：[num, layers, in_dim] (num=用户/物品数, layers=层数)
        att_signal = self.w(infeatures)  # [num, layers, att_dim]
        att_signal = F.relu(att_signal)
        att_signal = torch.mul(att_signal, self.h)  # [num, layers, att_dim]
        att_signal = torch.sum(att_signal, dim=-1)  # [num, layers]
        att_signal = F.softmax(att_signal, dim=-1)  # 沿layers维度归一化
        return att_signal  # [num, layers]


class TransformerLayer(nn.Module):
    """改进版Transformer层：修复layer_norm1命名错误 + 对齐残差逻辑"""
    def __init__(self, d_model, d_ff, n_heads, dropout=0.2, kq_same=False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)
        self.norm1 = nn.LayerNorm(d_model)  # 修复：原错误命名layer_norm1
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        # 初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, seq, mask=None):
        # 输入：seq [batch, seq_len, d_model] | mask [batch, seq_len, seq_len]
        # 自注意力 + 残差 + 归一化
        attn_out = self.attn(seq, seq, seq, mask)
        attn_out = self.dropout1(attn_out)
        context = self.norm1(attn_out + seq)  # 修复：原错误layer_norm1
        
        # 前馈网络 + 残差 + 归一化
        ff_out = self.linear1(context)
        ff_out = F.relu(ff_out)
        ff_out = self.linear2(ff_out)
        ff_out = self.dropout2(ff_out)
        output = self.norm2(ff_out + context)
        
        return output  # [batch, seq_len, d_model]


class Dice(nn.Module):
    """数据自适应激活函数：修复设备兼容问题"""
    def __init__(self, emb_size, dim=2, epsilon=1e-8, device=None):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        # 修复：device自动适配，避免硬编码
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,), device=self.device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1), device=self.device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)  # [batch, emb_size, seq_len]
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)  # 恢复原维度
        return out


class MLP_Block(nn.Module):
    """通用MLP块：兼容Dice激活 + 维度对齐"""
    def __init__(self, input_dim, hidden_units, output_dim=None, dropout=0.2, 
                 hidden_activations="ReLU", batch_norm=False, layer_norm=False, 
                 norm_before_activation=True, use_bias=True, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dense_layers = []
        
        # 处理dropout和激活函数的列表化
        if not isinstance(dropout, list):
            dropout = [dropout] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        
        # 激活函数映射（支持Dice）
        hidden_activations = [
            getattr(nn, act)() if act != "Dice" else Dice(emb_size=h_dim, device=self.device)
            for act, h_dim in zip(hidden_activations, hidden_units)
        ]
        
        prev_dim = input_dim
        for idx in range(len(hidden_units)):
            curr_dim = hidden_units[idx]
            # 线性层
            dense_layers.append(nn.Linear(prev_dim, curr_dim, bias=use_bias))
            # 归一化（前）
            if norm_before_activation:
                if batch_norm:
                    dense_layers.append(nn.BatchNorm1d(curr_dim))
                elif layer_norm:
                    dense_layers.append(nn.LayerNorm(curr_dim))
            # 激活函数
            dense_layers.append(hidden_activations[idx])
            # 归一化（后）
            if not norm_before_activation:
                if batch_norm:
                    dense_layers.append(nn.BatchNorm1d(curr_dim))
                elif layer_norm:
                    dense_layers.append(nn.LayerNorm(curr_dim))
            # Dropout
            if dropout[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = curr_dim
        
        # 输出层
        if output_dim is not None:
            dense_layers.append(nn.Linear(prev_dim, output_dim, bias=use_bias))
        
        self.mlp = nn.Sequential(*dense_layers)
        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # 输入：[batch, input_dim] 输出：[batch, output_dim/hidden_units[-1]]
        return self.mlp(x)


# ============================== 原有核心模块改进 ==============================
class EnhancedLSTMGNN(nn.Module):
    """增强版LSTMGNN：修复超图dropout + 维度对齐"""
    def __init__(self, emb_size, user_num, answer_num, dropout=0.2, device=None):
        super().__init__()
        self.emb_size = emb_size
        self.user_num = user_num
        self.answer_num = answer_num
        self.layers = 2
        self.drop_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 自门控参数
        self.user_gate_weights = nn.Parameter(torch.zeros(self.emb_size, self.emb_size, device=self.device))
        self.user_gate_bias = nn.Parameter(torch.zeros(1, self.emb_size, device=self.device))
        self.item_gate_weights = nn.Parameter(torch.zeros(self.emb_size, self.emb_size, device=self.device))
        self.item_gate_bias = nn.Parameter(torch.zeros(1, self.emb_size, device=self.device))
        
        # 嵌入层：用户+物品
        self.user_embedding = nn.Embedding(self.user_num, self.emb_size, padding_idx=0, device=self.device)
        self.answer_embedding = nn.Embedding(self.answer_num, self.emb_size, padding_idx=0, device=self.device)
        
        # 融合层
        self.user_fusion = MultiFusion(self.emb_size, dropout)
        self.item_fusion = Fusion(self.emb_size, dropout)
        
        # 注意力层
        self.user_att_layer = AttLayer(in_dim=emb_size, att_dim=emb_size//2)
        self.item_att_layer = AttLayer(in_dim=emb_size, att_dim=emb_size//2)
        
        self._init_weights()

    def self_gating(self, em, weights, bias):
        # 输入：[num, emb_size] 输出：[num, emb_size]
        gate = torch.sigmoid(torch.matmul(em, weights) + bias)
        return em * gate

    def _init_weights(self):
        # 统一初始化策略
        stdv = 1.0 / math.sqrt(self.emb_size)
        # 嵌入层初始化
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=stdv)
        nn.init.normal_(self.answer_embedding.weight, mean=0.0, std=stdv)
        # 自门控参数初始化
        nn.init.uniform_(self.user_gate_weights, -stdv, stdv)
        nn.init.uniform_(self.user_gate_bias, -stdv, stdv)
        nn.init.uniform_(self.item_gate_weights, -stdv, stdv)
        nn.init.uniform_(self.item_gate_bias, -stdv, stdv)
        # padding位置置零
        if self.user_embedding.padding_idx is not None:
            self.user_embedding.weight.data[self.user_embedding.padding_idx].zero_()
        if self.answer_embedding.padding_idx is not None:
            self.answer_embedding.weight.data[self.answer_embedding.padding_idx].zero_()

    def _dropout_graph(self, graph):
        """修复超图dropout逻辑：兼容CPU/GPU + 数值稳定性"""
        if not self.training or self.drop_rate <= 0:
            return graph
        
        graph = graph.to(self.device)
        coo = graph.coalesce()
        indices = coo.indices()
        values = coo.values()
        
        # 随机保留节点
        mask = torch.rand(len(values), device=self.device) > self.drop_rate
        indices = indices[:, mask]
        values = values[mask] / (1 - self.drop_rate)  # 缩放保留的权重
        
        # 重建稀疏矩阵
        dropout_graph = torch.sparse.FloatTensor(indices, values, coo.size(), device=self.device)
        return dropout_graph

    def forward(self, hypergraphs, phase):
        # 输入：hypergraphs = [user_item_adj, user_user_adj]（稀疏矩阵）
        # phase: train/test
        self.train(mode=(phase == 'train'))
        user_item_adj, user_user_adj = hypergraphs
        
        # 超图dropout
        if phase == 'train':
            user_item_adj = self._dropout_graph(user_item_adj)
            user_user_adj = self._dropout_graph(user_user_adj)
        
        # 初始嵌入
        user_emb = self.user_embedding.weight  # [user_num, emb_size]
        answer_emb = self.answer_embedding.weight  # [answer_num, emb_size]

        # 自门控特征选择
        user_emb = self.self_gating(user_emb, self.user_gate_weights, self.user_gate_bias)
        answer_emb = self.self_gating(answer_emb, self.item_gate_weights, self.item_gate_bias)
        
        user_emb_list = [user_emb]
        answer_emb_list = [answer_emb]
        
        for _ in range(self.layers):
            # 1. 用户-物品超图传播
            answer_emb_from_user = torch.sparse.mm(user_item_adj.T, user_emb)
            answer_emb_from_user = F.normalize(answer_emb_from_user, p=2, dim=1)
            
            user_emb_from_item = torch.sparse.mm(user_item_adj, answer_emb)
            user_emb_from_item = F.normalize(user_emb_from_item, p=2, dim=1)
            
            # 2. 用户-用户超图传播
            user_emb_from_neighbor = torch.sparse.mm(user_user_adj, user_emb)
            user_emb_from_neighbor = F.normalize(user_emb_from_neighbor, p=2, dim=1)
            
            # 3. 门控融合
            user_emb = self.user_fusion(user_emb, user_emb_from_item, user_emb_from_neighbor)
            answer_emb = self.item_fusion(answer_emb, answer_emb_from_user)
            
            # Dropout
            user_emb = self.dropout(user_emb)
            answer_emb = self.dropout(answer_emb)
            
            user_emb_list.append(user_emb)
            answer_emb_list.append(answer_emb)
        
        # 多层嵌入聚合（注意力加权）
        user_emb_cat = torch.stack(user_emb_list, dim=1)  # [user_num, layers+1, emb_size]
        user_att_weights = self.user_att_layer(user_emb_cat).unsqueeze(-1)  # [user_num, layers+1, 1]
        final_user_emb = (user_emb_cat * user_att_weights).sum(dim=1)  # [user_num, emb_size]
        
        answer_emb_cat = torch.stack(answer_emb_list, dim=1)  # [answer_num, layers+1, emb_size]
        item_att_weights = self.item_att_layer(answer_emb_cat).unsqueeze(-1)  # [answer_num, layers+1, 1]
        final_answer_emb = (answer_emb_cat * item_att_weights).sum(dim=1)  # [answer_num, emb_size]
        
        return final_user_emb, final_answer_emb


class DiffusionDataset(Dataset):
    """适配你的数据格式：动态生成历史序列"""
    def __init__(self, data: list, user2idx: dict = None, answer2idx: dict = None, 
                 phase: str = 'train', max_his: int = 50):  # 移除 diffusion_graph 参数
        self.phase = phase
        self.user2idx = user2idx if user2idx is not None else {}
        self.answer2idx = answer2idx if answer2idx is not None else {}
        self.max_his = max_his
        
        # 关键步骤1：按用户分组 + 按时间排序，为每个样本生成历史序列
        self.processed_data = self._generate_history(data)
        
        # 训练阶段过滤历史长度为0的样本（避免无历史可学）
        if self.phase != 'test':
            self.processed_data = [item for item in self.processed_data if len(item['history_items']) > 0]

    def _generate_history(self, raw_data: list) -> list:
        """
        动态生成历史序列：
        1. 按user_id分组
        2. 每组内按time（时间戳）升序排序
        3. 每个样本的history_items = 该用户当前样本之前的answer_id序列
        """
        # 按user_id分组
        user_groups = {}
        for item in raw_data:
            uid = item['user_id']
            if uid not in user_groups:
                user_groups[uid] = []
            user_groups[uid].append(item)
        
        processed_data = []
        for uid, items in user_groups.items():
            # 按时间戳升序排序（确保历史是"过去"的交互）
            items_sorted = sorted(items, key=lambda x: int(x['time']))  # time转int排序
            
            # 为每个样本生成历史序列
            for i, item in enumerate(items_sorted):
                # 历史序列 = 当前样本之前的answer_id（不包含当前item）
                history_raw = [it['answer_id'] for it in items_sorted[:i]]
                processed_item = {
                    'user_id': item['user_id'],
                    'answer_id': item['answer_id'],
                    'is_click': item['is_click'],
                    'history_items': history_raw  # 动态生成的历史
                }
                processed_data.append(processed_item)
        
        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def _get_idx(self, raw_id, id2idx):
        """获取原始ID对应的连续索引（padding=0）"""
        return id2idx.get(raw_id, 0)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        user_idx = self._get_idx(item['user_id'], self.user2idx)
        answer_idx = self._get_idx(item['answer_id'], self.answer2idx)
        
        # 处理历史序列：截断或padding到max_his
        history_raw = item['history_items']
        history_items = [self._get_idx(aid, self.answer2idx) for aid in history_raw]
        if len(history_items) > self.max_his:
            history_items = history_items[-self.max_his:]  # 保留最近的max_his个
        lengths = len(history_items)
        history_items = history_items + [0] * (self.max_his - lengths)  # padding到max_his
        
        feed_dict = {
            "user_idx": user_idx,
            "answer_idx": answer_idx,
            "history_items": torch.tensor(history_items, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "is_click": torch.tensor(item['is_click'], dtype=torch.float32),
            "phase": self.phase
        }
        return feed_dict

    @staticmethod
    def collate_batch(feed_dicts: List[Dict]) -> Dict:
        """批处理：保证tensor类型和设备兼容"""
        batch = {
            "user_idx": torch.tensor([d['user_idx'] for d in feed_dicts], dtype=torch.long),
            "answer_idx": torch.tensor([d['answer_idx'] for d in feed_dicts], dtype=torch.long),
            "history_items": torch.stack([d['history_items'] for d in feed_dicts], dim=0),
            "lengths": torch.tensor([d['lengths'] for d in feed_dicts], dtype=torch.long),
            "is_click": torch.tensor([d['is_click'] for d in feed_dicts], dtype=torch.float32),
            "phase": feed_dicts[0]['phase']
        }
        return batch

class DiffusionAgent(nn.Module):
    """优化版Diffusion模型：融合序列历史+GNN结构特征"""
    def __init__(self, args: Dict, corpus: Dict):
        super().__init__()
        self.emb_size = args.get('emb_size', 64)
        self.num_layers = args.get('num_layers', 2)
        self.num_heads = args.get('num_heads', 4)
        self.user_num = corpus['n_users']
        self.answer_num = corpus['n_answers']
        self.max_his = args.get('max_his', 50)  # 需与Dataset一致
        self.device = args.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.d_ff = args.get('d_ff', self.emb_size * 4)
        self.dropout = args.get('dropout', 0.2)
        
        # 1. GNN编码器：获取用户/物品的结构特征
        self.gnn = EnhancedLSTMGNN(
            emb_size=self.emb_size,
            user_num=self.user_num,
            answer_num=self.answer_num,
            dropout=self.dropout,
            device=self.device
        )
        
        # 2. 序列特征编码层
        self.i_embeddings = nn.Embedding(self.answer_num, self.emb_size, padding_idx=0)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.register_buffer('len_range', torch.arange(self.max_his))  # 自动设备迁移
        
        # 3. Transformer层：建模历史序列依赖
        self.transformer_blocks = nn.ModuleList([
            TransformerLayer(
                d_model=self.emb_size,
                d_ff=self.d_ff,
                n_heads=self.num_heads,
                dropout=self.dropout,
                kq_same=False
            ) for _ in range(self.num_layers)
        ])
        
        # 4. 融合层：历史特征 + GNN结构特征
        self.user_fusion = MLP_Block(
            input_dim=self.emb_size * 2,
            hidden_units=[self.emb_size],
            output_dim=self.emb_size,
            dropout=self.dropout,
            hidden_activations=["ReLU"],
            layer_norm=True,
            norm_before_activation=True,
            device=self.device
        )
        
        # 5. 预测层
        self.predictor = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            Dice(self.emb_size, device=self.device),
            nn.Dropout(self.dropout),
            nn.Linear(self.emb_size, 1)
        )
        
        # 统一初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()

    def forward(self, feed_dict: Dict, diffusion_graph) -> Dict:
        # 1. 输入数据预处理
        user_idx = feed_dict['user_idx'].to(self.device)
        answer_idx = feed_dict['answer_idx'].to(self.device)
        history_items = feed_dict['history_items'].to(self.device)
        lengths = feed_dict['lengths'].to(self.device)
        batch_size = user_idx.size(0)
        
        # 2. GNN编码
        user_graph_emb, answer_graph_emb = self.gnn(diffusion_graph, phase=feed_dict.get('phase', 'train'))
        batch_user_graph_emb = F.embedding(user_idx, user_graph_emb)
        
        # 3. 历史序列编码
        valid_his = (history_items > 0).long()
        his_vectors = self.i_embeddings(history_items)
        
        # 位置编码
        seq_len = history_items.size(1)
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors
        
        # 4. Transformer + 因果掩码
        causality_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=self.device, dtype=torch.long))
        for block in self.transformer_blocks:
            his_vectors = block(his_vectors, causality_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()
        
        # 5. 平均池化
        his_vector = his_vectors.sum(dim=1) / (lengths[:, None].float() + 1e-8)
        
        # 6. 特征融合
        user_vector = self.user_fusion(torch.cat([his_vector, batch_user_graph_emb], dim=-1))
        
        # 7. 预测
        batch_answer_graph_emb = F.embedding(answer_idx, answer_graph_emb)
        prediction_input = torch.cat([user_vector, batch_answer_graph_emb], dim=-1)
        prediction = self.predictor(prediction_input).squeeze(-1)
        prediction = torch.sigmoid(prediction)
        
        return {"prediction": prediction}


class SimpleRunner:
    """简化版训练器：修复checkpoint加载 + 评估逻辑"""
    def __init__(self, args: Dict):
        self.args = args
        self.device = args.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.epoch = args.get('epoch', 10)
        self.lr = args.get('lr', 1e-3)
        self.batch_size = args.get('batch_size', 256)
        self.patience = args.get('patience', 1000)
        self.save_every_n_epochs = args.get('save_every_n_epochs', 5)
        self.save_dir = args.get('save_dir', './diffusion_ckpt')
        os.makedirs(self.save_dir, exist_ok=True)

    def find_latest_checkpoint(self):
        """查找最新的checkpoint文件"""
        ckpt_pattern = os.path.join(self.save_dir, 'ckpt_epoch_*.pth')
        ckpt_files = glob.glob(ckpt_pattern)
        
        if not ckpt_files:
            return None, 0
        
        valid_ckpts = []
        for file_path in ckpt_files:
            try:
                file_name = os.path.basename(file_path)
                epoch_num = int(file_name.split('_')[-1].split('.')[0])
                valid_ckpts.append((file_path, epoch_num))
            except (ValueError, IndexError):
                continue
        
        if not valid_ckpts:
            return None, 0
        
        valid_ckpts.sort(key=lambda x: x[1], reverse=True)
        latest_ckpt, latest_epoch = valid_ckpts[0]
        return latest_ckpt, latest_epoch

    def train(self, model: DiffusionAgent, train_data: list, dev_data: list, 
              user2idx: dict, answer2idx: dict) -> nn.Module:
        """训练模型：支持从最新checkpoint继续训练"""
        # 构建超图
        diffusion_graph = utils.build_hypergraph(
            user_item_data=train_data,
            user_user_data=train_data,
            user2idx=user2idx,
            answer2idx=answer2idx
        )
        # 超图移到设备
        diffusion_graph = [g.to(self.device) for g in diffusion_graph]
        
        dev_diffusion_graph = utils.build_hypergraph(
            user_item_data=dev_data,
            user_user_data=dev_data,
            user2idx=user2idx,
            answer2idx=answer2idx
        )
        dev_diffusion_graph = [g.to(self.device) for g in dev_diffusion_graph]
        
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.BCELoss()
        
        # 加载最佳模型和最新checkpoint
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        best_f1 = 0.0
        
        if os.path.exists(best_model_path):
            try:
                best_ckpt = torch.load(best_model_path, map_location=self.device)
                best_f1 = best_ckpt.get('best_f1', 0.0)
                print(f"加载历史最佳模型F1: {best_f1:.4f}")
            except Exception as e:
                print(f"加载最佳模型失败: {e}，使用默认F1=0.0")
        
        latest_ckpt, start_epoch = self.find_latest_checkpoint()
        if latest_ckpt:
            try:
                ckpt = torch.load(latest_ckpt, map_location=self.device)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt['epoch']
                ckpt_dev_f1 = ckpt.get('dev_f1', 0.0)
                if ckpt_dev_f1 > best_f1:
                    best_f1 = ckpt_dev_f1
                print(f"成功加载最新checkpoint: {latest_ckpt}")
                print(f"从epoch {start_epoch} 继续训练，当前最佳F1: {best_f1:.4f}")
            except Exception as e:
                print(f"加载checkpoint失败: {e}，从头开始训练")
                start_epoch = 0
        else:
            print("未找到任何checkpoint，从头开始训练")
            start_epoch = 0
        
        # 构建数据集（修正：移除 diffusion_graph 传参）
        train_dataset = DiffusionDataset(train_data, user2idx, answer2idx, phase='train', max_his=self.args.get('max_his', 20))
        dev_dataset = DiffusionDataset(dev_data, user2idx, answer2idx, phase='dev', max_his=self.args.get('max_his', 20))
        
        # 训练循环
        for epoch in range(start_epoch, self.epoch):
            current_epoch = epoch + 1
            model.train()
            total_loss = 0.0
            
            dl = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                collate_fn=DiffusionDataset.collate_batch
            )
            
            # DiffusionAgent.py - SimpleRunner.train 方法（约第738行附近）
            for batch in tqdm(dl, desc=f"Epoch {current_epoch}/{self.epoch}"):
                optimizer.zero_grad()
                # 批数据移到设备：仅移动Tensor，跳过phase字符串
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):  # 新增类型判断
                        batch[k] = batch[k].to(self.device)

                batch['phase'] = 'train'
                
                out_dict = model(batch, diffusion_graph)
                prediction = out_dict['prediction']
                target = batch['is_click']
                
                loss = loss_fn(prediction, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            # 验证
            dev_metrics = self.evaluate(model, dev_dataset, dev_diffusion_graph, loss_fn)
            print(f"Epoch {current_epoch} | 训练损失: {total_loss/len(dl):.4f} | 验证指标: {utils.format_metric(dev_metrics)}")
            
            # 保存checkpoint
            if current_epoch % self.save_every_n_epochs == 0:
                ckpt_path = os.path.join(self.save_dir, f'ckpt_epoch_{current_epoch}.pth')
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dev_f1': dev_metrics['F1'],
                    'loss': total_loss/len(dl)
                }, ckpt_path)
                print(f"保存checkpoint到: {ckpt_path}")
            
            # 保存最佳模型
            if dev_metrics['F1'] > best_f1:
                best_f1 = dev_metrics['F1']
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'best_f1': best_f1
                }, best_model_path)
                print(f"更新最佳模型（F1: {best_f1:.4f}）到: {best_model_path}")
        
        # 加载最佳模型
        best_ckpt = torch.load(best_model_path, map_location=self.device)
        model.load_state_dict(best_ckpt['model_state_dict'])
        return model

    # DiffusionAgent.py - SimpleRunner 类的 evaluate 方法修正（补充 diffusion_graph 传参）
    def evaluate(self, model: DiffusionAgent, dataset: DiffusionDataset, 
                diffusion_graph: List, loss_fn) -> dict:
        """评估模型：对齐输入输出"""
        model.eval()
        total_loss = 0.0
        all_pred = []
        all_true = []
        
        dl = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=DiffusionDataset.collate_batch
        )
        
        with torch.no_grad():
            for batch in dl:
                # 批数据移到设备：仅移动Tensor，跳过phase字符串
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):  # 新增类型判断
                        batch[k] = batch[k].to(self.device)
                batch['phase'] = dataset.phase
                
                # 修正：确保 diffusion_graph 已移到对应设备，且传递给 model.forward
                out_dict = model(batch, diffusion_graph)
                prediction = out_dict['prediction']
                target = batch['is_click']
                
                loss = loss_fn(prediction, target)
                total_loss += loss.item()
                
                all_pred.extend(prediction.cpu().numpy())
                all_true.extend(target.cpu().numpy())
        
        metrics = utils.evaluate_metrics(np.array(all_true), np.array(all_pred))
        metrics['loss'] = total_loss / len(dl)
        return metrics

    def evaluate_from_checkpoint(self, model: DiffusionAgent, test_data: list, seed_data: list,
                                 user2idx: dict, answer2idx: dict, load_ckpt_path: str, save_path: str) -> dict:
        """从checkpoint加载模型并推理"""
        # 构建超图
        diffusion_graph = utils.build_hypergraph(
            user_item_data=seed_data,
            user_user_data=test_data,
            user2idx=user2idx,
            answer2idx=answer2idx
        )
        diffusion_graph = [g.to(self.device) for g in diffusion_graph]
        
        # 加载checkpoint
        ckpt = torch.load(load_ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from {load_ckpt_path} (best F1: {ckpt.get('best_f1', 'N/A')})")
        
        # 构建测试集
        test_dataset = DiffusionDataset(test_data, user2idx, answer2idx, phase='test')
        
        model.eval()
        all_pred = []
        all_true = []
        all_user_ids = [item['user_id'] for item in test_data]
        all_answer_ids = [item['answer_id'] for item in test_data]
        
        dl = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=DiffusionDataset.collate_batch
        )
        
        with torch.no_grad():
            for batch in tqdm(dl, desc="Predicting"):
                # 批数据移到设备：仅移动Tensor，跳过phase字符串
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):  # 新增类型判断
                        batch[k] = batch[k].to(self.device)
                batch['phase'] = 'test'
                
                out_dict = model(batch, diffusion_graph)
                prediction = out_dict['prediction']
                target = batch['is_click']
                
                all_pred.extend(prediction.cpu().numpy())
                all_true.extend(target.cpu().numpy())
        
        # 计算指标并保存
        metrics = utils.evaluate_metrics(np.array(all_true), np.array(all_pred))
        pred_results = [
            {
                "user_id": all_user_ids[i],
                "answer_id": all_answer_ids[i],
                "true_click": int(all_true[i]),
                "pred_click_prob": float(all_pred[i]),
                "pred_click": 1 if all_pred[i] >= 0.5 else 0
            }
            for i in range(len(all_true))
        ]
        detailed_save_path = save_path.replace('.json', '_detailed.json')
        utils.save_json(pred_results, detailed_save_path)
        utils.save_json(metrics, save_path)
        print(f"Evaluation metrics saved to {save_path}")
        print(f"Detailed predictions saved to {detailed_save_path}")
        return metrics