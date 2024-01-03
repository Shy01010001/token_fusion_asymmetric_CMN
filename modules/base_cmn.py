# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:07:43 2023

@author: hongyu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from .att_model import pack_wrapper, AttModel

class TokenFussion(nn.Module):
    def __init__(self, d_model, num_classes, dropout = 0.1):
        super(TokenFussion, self).__init__()
        self.input_dimension = d_model // 8 # 8 head number
        self.attn1 = nn.Sequential(nn.Linear(self.input_dimension, 2048), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2048, num_classes), nn.ReLU())
        self.norm = LayerNorm(512)
    def forward(self, x):
        x = rearrange(x, 'b p (h d) -> b h p d', h = 8) # head number = 8
        s_x = self.attn1(x)
        s_x = s_x.permute(0, 1, 3, 2)
        s_x = rearrange(s_x, 'b h p d -> (b h) p d')
        x = rearrange(x, 'b h p d -> (b h) p d')
        x = torch.bmm(s_x, x)
        x = rearrange(x, '(b h) p d -> b p (h d)', h = 8)
        return x

def min_max_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = torch.div(torch.sub(tensor, min_val), max_val - min_val)
    return normalized_tensor

class tsne:
    def __init__(self):
        self.tsne = TSNE(n_components=2, perplexity=5, learning_rate=200)
        self.scaler = MinMaxScaler(feature_range=(0, 1)) 
        
    def run(self, inpt, sign):
        # print(inpt.size())
        # exit()
        global img_id, count
        count = 0
        # print(img_id)
        # print(g_epoch)
        try:
            if (g_epoch -1)  % 4 == 0:
                
                if img_id is not None:
                    tsne_inpt = self.tensor2array((inpt))
                    # print(tsne_inpt)
                    for ids in img_id:
                        tsne_result = self.tsne.fit_transform(tsne_inpt[ids])
                        tsne_result_normalized = self.scaler.fit_transform(tsne_result)
                        self.plot_scatter(tsne_result_normalized, f'./t_sne/{g_name[ids]}/{g_epoch}_{sign}.jpg')
        except:
            pass
            
    def tensor2array(self, x):  # x : (batch x 98 x 512)
        l = []
        for i in x:
            l.append(i.detach().cpu().numpy())
            
        return l    
    def plot_scatter(self, data, save_path):
        fig, ax = plt.subplots()
    
        x = [point[0] for point in data]
        y = [point[1] for point in data]
    
        ax.scatter(x, y)
        ax.set_xlim(0, 1)  # 设置横坐标范围为 -1 到 1
        ax.set_ylim(0, 1)  # 设置纵坐标范围为 -1 到 1
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Scatter Plot')
        plt.tight_layout()

        # 调整图像边界，使其能够完整显示
        plt.tight_layout()
    
        # 保存图像
        plt.savefig(save_path)
        
class set_flag:
    def set_epoch(e):
        global g_epoch
        g_epoch = e
    def set_image_id(i = None, name = None):
        global img_id, g_name
        img_id = i
        g_name = name

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size): # 生成(1, size, size)的对角矩阵, 左下为True右上为False
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None): # [batch_size, head, 98, dim_perhead(64)]
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32): # [16, 8, 98, 64], [16, 8, 2048, 64], [16, 8, 2048, 64]
    d_k = query.size(-1) # 512/8
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # [16, 8, 98, 2048] # 公式(7, 8) # D_s_i, D_t_i
    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    selected_scores, idx = scores.topk(topk) # 选择最相关的K个记忆向量 # [16, 8, 98, 32(k)]
    # print('all zeros jdge',scores.any())
    # torch.save(scores, 'scores.pt')
    # vmax = 1e-10
    # vmin = -1e-10
    # for i in range(selected_scores.size(1)): # heads number
    #     heat_map = selected_scores[0, i, :, :].detach().cpu().numpy() # fussing token number
    #     plt.imshow(heat_map, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    #     plt.colorbar()
    #     if selected_scores.size(2) != 20:
    #         plt.savefig(f'text_{i}heatmap.png')
    #     else:
    #         plt.savefig(f'image_{i}heatmap.png')
    #     # plt.show()
    #     plt.clf()
    
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1)) # [16, 8, 98, 2048, 64]
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1)) # [16, 8, 98, 32, 64]
    selected_value = torch.gather(dummy_value, 3, dummy_idx) # [16, 8, 98, 32, 64]
    p_attn = F.softmax(selected_scores, dim=-1) # [16, 8, 98, 32] # 公式(9, 10) # w_s_i, w_t_i
    # print('p_attn', p_attn[1, 1,:,:])
    # print('value', selected_value[1,1,:,:])
    # exit()
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn # [16, 8, 98, 64] # 公式(12, 13) # r_x_s, r_y_t


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, flag = 0):
        if flag != 0:
            return sublayer(self.norm(x))
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab): # vocab_size+1
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class MultiThreadMemory(nn.Module):
    def __init__(self, head, d_model, dropout=0.1, topk=1):
        super(MultiThreadMemory, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.h = head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None): # [batch_size, 98, d_model], [batch_size, cmm_size, cmm_dim]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for x in [query, key, value]] # [batch_size, head, 98, d_k], [batch_size, head, cmm_size, d_k], [batch_size, head, cmm_size, d_k]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk) # [batch_size, head, 98, d_k], [batch_size, head, 98, 32(k)] # 公式(12, 9) # r, w

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) # [batch_size, 98, d_model]
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x) # memory responses for visual features, encoder input


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.h = head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for x in [query, key, value]] # (batch_size, head, seq_len, dim_perhead)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# !!! mapping by disease classes
class SemanticMap(nn.Module):
    def __init__(self, d_model, dropout = 0, execute_norm = True):
        super(SemanticMap, self).__init__()
        self.attn = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(), nn.Linear(2048, 512))
        self.Tanh = nn.Tanh()
        self.tsne = tsne()
        self.execute_norm = execute_norm
        # self.SoftMax = nn.Softmax()
        self.norm = LayerNorm(512)
        
    def forward(self, x, layer = 0):
        if self.execute_norm:
            x = self.norm(x)
        self.tsne.run(x, f'{layer}')
        x = self.attn(x)
        sm_x = self.Tanh(x)
        # tanh_x = self.Tanh(x)
        return min_max_normalize(sm_x)
    
class TokenFussion(nn.Module):
    def __init__(self, d_model, num_classes, dropout = 0.1):
        super(TokenFussion, self).__init__()
        self.input_dimension = d_model // 8 # 8 head number
        self.attn1 = nn.Sequential(nn.Linear(self.input_dimension, 2048), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2048, num_classes), nn.Sigmoid())
        self.norm = LayerNorm(512)
    def forward(self, x):
        x = rearrange(x, 'b p (h d) -> b h p d', h = 8) # head number = 8
        s_x = self.attn1(x)
        s_x = s_x.permute(0, 1, 3, 2)
        s_x = rearrange(s_x, 'b h p d -> (b h) p d')
        x = rearrange(x, 'b h p d -> (b h) p d')
        x = torch.bmm(s_x, x)
        x = rearrange(x, '(b h) p d -> b p (h d)', h = 8)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # attn: [batch_size, head, 98, 98]
        return self.sublayer[-1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None, token_fusion = None): # report词向量[batch_size, seq_len, d_model], encoder输出[batch_size, 98, d_model]
        m = memory
        if layer_past is None: # train
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward) # [batch_size, seq_len, d_model]
        else: # sample
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0])) # attn: [batch_size, head, seq_len, seq_len]
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1])) # attn: [batch_size, head, seq_len, 98]
            return self.sublayer[2](x, self.feed_forward), present


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # N个encoder layer
        self.forget_gates = clones(SemanticMap(512, 0, False), N) ## 512 是feature token的dimensions
        self.norm = LayerNorm(layer.size)
        self.token_fusion = TokenFussion(512, 20)
        # self.semantic_map1 = SemanticMap(512, 70)
        # self.semantic_map2 = SemanticMap(512, 60)
        # self.semantic_map3 = SemanticMap(512, 50)

    def forward(self, x, mask):
        # no_layer = 1
        dummy = torch.zeros_like(x)
        mesh = []
        for layer in self.layers:
            x = layer(x, mask)
            mesh.append(x)
            # gate = forget_gate(x, layer = no_layer)
            # x = gate * x
            # no_layer += 1
            # if flag == 0:
                # x = self.semantic_map(x)
            # outputs.append(x.unsqueeze(1))
            # flag += 1
        fuses = self.token_fusion(x)
        dummy[:, :20] = fuses
        mesh.append(dummy)
        mesh = torch.stack(mesh, dim = 0)

        # outputs = torch.cat(outputs, 1)
        # return outputs
        # return self.norm(x)
        return mesh


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None): # report词向量[batch_size, seq_len, d_model], encoder输出[batch_size, 98, d_model]
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers) # [None, None, None], num_layers
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask, layer_past) # [batch_size, seq_len, d_model]
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x) # [batch_size, seq_len, d_model]
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)] # 输出完整序列


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, cmn):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.cmn = cmn

    def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix):
        return self.decode(self.encoder(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix)

    def encode(self, src, src_mask): # [batch_size, 98, d_model], [batch_size, 1, 98]
        # !!! mapping by disease classes
        return self.encoder(src, src_mask) # [batch_size, 98, d_model]

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None, memory_matrix=None):
        # print(tgt)
        embeddings = self.tgt_embed(tgt) # [batch_size, seq_len, 768]
        # print('embeddings',embeddings.size())
        # exit()
        
        # Memory querying and responding for textual features
        dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0),
                                                                memory_matrix.size(1)) # [batch_size, cmm_size, cmm_dim]

        responses = self.cmn(embeddings, dummy_memory_matrix, dummy_memory_matrix) # [batch_size, seq_len, 768]
        embeddings = embeddings + responses
        # Memory querying and responding for textual features

        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)

class SrcMultiHeadedAttention(nn.Module):
    def __init__(self, head, d_model, N, dropout=0.1):
        super(SrcMultiHeadedAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.h = head
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layer_num = N + 1 # 魔鬼数字 1 1层 token fusion
        self.query_map = clones(nn.Linear(d_model, d_model), 1)
        self.key_map = clones(nn.Linear(d_model, d_model), 4) # 魔鬼数字 4 3层self attention + 1层 token fusion
        self.value_map = clones(nn.Linear(d_model, d_model), 4) # 魔鬼数字 4 3层self attention + 1层 token fusion
        self.sigma_map = clones(nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid()), self.layer_num) # 2d([Y, Attention(Y, Src)]) -> 1d
        self.fw_map = clones(nn.Linear(d_model, d_model), 1)
        self.norm = nn.LayerNorm(512)

    def forward(self, query, key, value, mask=None, layer_past=None):
        _query = query
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.query_map[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else: 
            query = self.query_map[0](query)
            for lyr in range(self.layer_num):
                key[:, lyr] = self.key_map[lyr](key[:, lyr])
                value[:, lyr] = self.value_map[lyr](value[:, lyr])

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            # print(key.size())
            # print(value.size())            
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            # print(key.size())
            # print(value.size())
            
            present = torch.stack([key, value])
            # print(present.size())
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2).unsqueeze(1).repeat(1, key.size(1), 1, 1, 1)
        key, value = [x.view(nbatches, self.layer_num, -1, self.h, self.d_k).transpose(2, 3) for x in [key, value]] # (batch_size, head, seq_len, dim_perhead)
        x, self.attn = attention(query, key, value, mask=None, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous().view(nbatches, self.layer_num, -1, self.h * self.d_k)
        query = query.transpose(2, 3).contiguous().view(nbatches, self.layer_num, -1, self.h * self.d_k)
        sigma = torch.zeros_like(query)
        for lyr in range(self.layer_num):
            sigma[:, lyr] = self.sigma_map[lyr](torch.cat((_query, x[:, lyr]), dim = -1))
        # print('qual?',torch.equal(query[:, 0], query[:, 1]))
        # exit()
        x = (torch.sum(sigma * x, dim = 1)) / np.sqrt(3) #### 1111!!!
        # print('prensentL:',present.size())
        if layer_past is not None:
            return self.fw_map[0](x), present
        else:
            return self.fw_map[0](x)

class BaseCMN(AttModel):
    def make_model(self, tgt_vocab, cmn):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        src_attn = SrcMultiHeadedAttention(self.num_heads, self.d_model, self.num_layers, dropout = self.dropout)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), 3), # only 1 layer for image feature extraction
            Decoder(DecoderLayer(self.d_model, c(attn), src_attn, c(ff), self.dropout), self.num_layers),
            nn.Sequential(c(position)), nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)), cmn)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(BaseCMN, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        # !!! mapping by disease classes
        self. num_classes = args.num_classes
        self.dropout = args.dropout
        self.topk = args.topk
        self.norm = nn.LayerNorm(512)
        self.tsne = tsne()
        
        tgt_vocab = self.vocab_size + 1

        # !!! mapping by disease classes
        self.semantic = SemanticMap(self.d_model, self.dropout)
        # self.token_fussion = TokenFussion(512, 20)

        # self.semantic = SemanticMap(self.d_model, 60, self.dropout)
        self.cmn = MultiThreadMemory(args.num_heads, args.d_model, topk=args.topk)

        self.model = self.make_model(tgt_vocab, self.cmn)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        # self.memory_matrix = nn.Parameter(torch.FloatTensor(args.d_vf, args.cmm_dim)) # 多模态记忆 memory matrix
        self.memory_matrix = nn.Parameter(torch.FloatTensor(98 * 4, args.cmm_dim))
        nn.init.normal_(self.memory_matrix, 0, 1 / args.cmm_dim)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        att_masks = None
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None): # [batch_size, 98, d_vf], None, [batch_size, max_seq_len]
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks) # [batch_size, 98, d_model]
        # !!! mapping by disease classes
        image_memory_matrix = self.memory_matrix.view(98, 4, -1) # 20 融合之后的token数
        dummy_memory_matrix = image_memory_matrix.unsqueeze(0)
        dummy_memory_matrix = torch.sum(dummy_memory_matrix, dim = 2)
        att_feats = (att_feats + dummy_memory_matrix/4) / np.sqrt(2)
        # forget_gate = self.semantic(att_feats)
        
        # att_feats = forget_gate * att_feats
        # self.tsne.run(att_feats, 'l')
        # dummy_memory_matrix = self.memory_matrix.unsqueeze(0).expand(att_feats.size(0), self.memory_matrix.size(0),
        #                                                               self.memory_matrix.size(1)) # [batch_size, cmm_size, cmm_dim]
        # responses = self.cmn(att_feats, dummy_memory_matrix, dummy_memory_matrix) # [batch_size, 98, d_model]    
        # att_feats = att_feats + responses
        # Memory querying and responding for visual features

        # Memory querying and responding for visual features
        
        
        
        if att_masks is None:
            att_masks = att_feats.new_ones(50, dtype=torch.long) # [batch_size, 98]  50 is the number of fused token
        att_masks = att_masks.unsqueeze(-2) # [batch_size, 98, 1]
        if seq is not None:
            seq = seq[:, :-1] # [batch_size, max_seq_len-1]
            seq_mask = (seq.data > 0) # [batch_size, max_seq_len-1]
            seq_mask[:, 0] += True # [batch_size]

            seq_mask = seq_mask.unsqueeze(-2) # [batch_size, 1, max_seq_len-1]
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask) # [1, max_seq_len-1, max_seq_len-1] 左下为True右上为False的对角矩阵
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask # [batch_size, 98, d_model], [batch_size, max_seq_len-1], [batch_size, 98, 1], [1, max_seq_len-1, max_seq_len-1]

    def _forward(self, fc_feats, att_feats, seq, att_masks=None): # seq: (batch_size, seq_len)
        # print(self.memory_matrix.size())
        # heat_map = torch.matmul(self.memory_matrix, self.memory_matrix.transpose(1, 0)).detach().cpu().numpy()
        
        # print(self.memory_matrix)
        # plt.imshow(heat_map, cmap='hot', interpolation='nearest')
        # plt.colorbar()      
        # plt.show()
        # plt.savefig(f'memory_matrix_similarity.png')
        # exit()
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        # print(seq.size())
        # print(seq_mask.size())
        # exit()
        att_masks = None
        out = self.model(att_feats, seq, att_masks, seq_mask, memory_matrix=self.memory_matrix) # [batch_size, seq_len, d_model]
        outputs = F.log_softmax(self.logit(out), dim=-1) # [batch_size, max_seq_len-1, vocab_size+1]

        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, 98, self.d_model)]
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        out, past = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device), past=past)
        return out[:, -1], [ys.unsqueeze(0)] + past
