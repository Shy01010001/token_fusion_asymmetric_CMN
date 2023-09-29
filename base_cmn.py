# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:30:44 2023

@author: hongyu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange

from .att_model import pack_wrapper, AttModel

def get_dict(data_dict):
    global pos_dict
    pos_dict = data_dict

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


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32, tgt = None): # [16, 8, 98, 64], [16, 8, 2048, 64], [16, 8, 2048, 64]

    d_k = query.size(-1) # 512/8
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # [16, 8, 98, 2048] # 公式(7, 8) # D_s_i, D_t_i
    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    if tgt == None:
        # print('NONE!!:',scores.size())
        selected_scores, idx = scores.topk(topk) # 选择最相关的K个记忆向量 # [16, 8, 98, 32(k)]
        dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1)) # [16, 8, 98, 2048, 64]
        dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1)) # [16, 8, 98, 32, 64]
        selected_value = torch.gather(dummy_value, 3, dummy_idx) # [16, 8, 98, 32, 64]
        p_attn = F.softmax(selected_scores, dim=-1) # [16, 8, 98, 32] # 公式(9, 10) # w_s_i, w_t_i
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn        
        
    else:
        # print('NONE:',scores.size())
        selected_scores = torch.zeros_like(scores)
    
        selected_value = torch.zeros_like(value).unsqueeze(2).expand(value.size(0), value.size(1), scores.size(2), value.size(2), value.size(3))        
    # print(scores.size())
    # print(tf_idf.size())
        if mask != None:
            for i in range(scores.size(0)):
                for j in range(scores.size(2)):
                    # try:
                    
                    key = str(tgt[i][j].item())
                    if key == '0':
                        topk = 64
                    elif key in ['38', '337', '745', '746', '747', '757', '759']:
                        topk = 32
                    elif key in ['127', '232', '379', '518', '589', '676', '686']:
                        topk = 16
                    else:
                        pos = pos_dict[key]
                        if pos in ['VERB', 'PROPN', 'ADJ', 'AUX', 'ADV', 'NUM', 'INTJ', 'SCONJ']:
                            topk = 32
                        elif pos in  ['NOUN']:
                            topk = 16
                        else:
                            topk = 0
                    if topk != 0:
                        selected_scores[i, :, j, : topk], idx = scores[i, :, j].topk(topk) # 选择最相关的K个记忆向量 # [16, 8, 98, 32(k)]
                        for k in range(8):  ## # of heads 8
                            selected_value[i, :, j][k, :idx.size(1), :] = torch.gather(value[i][k], 0, idx[k][:idx.size(1)].unsqueeze(1).expand(idx.size(1), 64))
        else:
            for i in range(scores.size(0)):
                key = str(tgt[i][-1].item())
                if key == '0':
                    topk = 64
                elif key in ['38', '337', '745', '746', '747', '757', '759']:
                    topk = 32
                elif key in ['127', '232', '379', '518', '589', '676', '686']:
                    topk = 16
                else:
                    pos = pos_dict[key]
                    if pos in ['VERB', 'PROPN', 'ADJ', 'AUX', 'ADV', 'NUM', 'INTJ', 'SCONJ']:
                        topk = 32
                    elif pos in  ['NOUN']:
                        topk = 16
                    else:
                        topk = 0
                if topk != 0:
                    selected_scores[i, :, -1, : topk], idx = scores[i, :, -1].topk(topk) # 选择最相关的K个记忆向量 # [16, 8, 98, 32(k)]
                    for k in range(8):  ## # of heads 8
                        selected_value[i, :, -1][k, :idx.size(1), :] = torch.gather(value[i][k], 0, idx[k][:idx.size(1)].unsqueeze(1).expand(idx.size(1), 64))                
            
                # except Exception as ex:
                #     if key == '0':
                #         topk = 640
                    
                #     if key in ['38', '337', '745', '746', '747', '759']:
                #         topk = 32
                #     else:
                #         topk = 16
                #     selected_scores[i, :, j, : topk], idx = scores[i, :, j].topk(topk) # 选择最相关的K个记忆向量 # [16, 8, 98, 32(k)]
                #     for k in range(8):  ## # of heads 8
                #         selected_value[i, :, j][k, :idx.size(1), :] = torch.gather(value[i][k], 0, idx[k][:idx.size(1)].unsqueeze(1).expand(idx.size(1), 64))          
                # print(idx.unsqueeze(-1).size())
    
                
                # print(dummy_value)
                # exit()
        # print(selected_scores)
        # exit()
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
        
        # dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1)) # [16, 8, 98, 2048, 64]
        # dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1)) # [16, 8, 98, 32, 64]
        # selected_value = torch.gather(dummy_value, 3, dummy_idx) # [16, 8, 98, 32, 64]
            p_attn = F.softmax(selected_scores, dim=-1) # [16, 8, 98, 32] # 公式(9, 10) # w_s_i, w_t_i
            # print(p_attn)
            # print('p_attn', p_attn[1, 1,:,:])
            # print('value', selected_value[1,1,:,:])
            # exit()
            # if dropout is not None:
            res = torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3)
            if topk == 0:
                return torch.zeros_like(res), p_attn
                # p_attn = dropout(p_attn)
            return res, p_attn # [16, 8, 98, 64] # 公式(12, 13) # r_x_s, r_y_t


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

    def forward(self, x, sublayer):
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
    def __init__(self, head, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.h = head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None, tgt = None): # [batch_size, 98, d_model], [batch_size, cmm_size, cmm_dim]
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
        # print('query.size()',query.size())
        # exit()
        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk, tgt = tgt) # [batch_size, head, 98, d_k], [batch_size, head, 98, 32(k)] # 公式(12, 9) # r, w

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
    def __init__(self, d_model, num_classes, dropout = 0):
        super(SemanticMap, self).__init__()
        self.attn = nn.Sequential(nn.Linear(d_model, num_classes), nn.Softmax())

    def forward(self, x):
        s_x = self.attn(x)
        s_x = s_x.permute(0,2,1)
        x = torch.bmm(s_x, x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # attn: [batch_size, head, 98, 98]
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None): # report词向量[batch_size, seq_len, d_model], encoder输出[batch_size, 98, d_model]
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
        self.norm = LayerNorm(layer.size)
        self.semantic_map1 = SemanticMap(512, 70)
        self.semantic_map2 = SemanticMap(512, 60)
        self.semantic_map3 = SemanticMap(512, 50)
    def forward(self, x, mask):
        # for GPT
        # outputs = []
        flag = 0
       # print(mask.size())
        for layer in self.layers:
            x = layer(x, mask)
            # outputs.append(x.unsqueeze(1))
            if flag == 0:
                x = self.semantic_map1(x)
                mask = mask[:, :, :x.size(1)]
            elif flag == 1:
                x = self.semantic_map2(x)
                mask = mask[:, :, :x.size(1)]
            else :
                x = self.semantic_map3(x)
                mask = mask[:, :,:x.size(1)]
            flag += 1
        # outputs = torch.cat(outputs, 1)
        # return outputs
        return self.norm(x)


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
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix)

    def encode(self, src, src_mask): # [batch_size, 98, d_model], [batch_size, 1, 98]
        # !!! mapping by disease classes
        return self.encoder(src, src_mask) # [batch_size, 98, d_model]

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None, memory_matrix=None):
        # print(tgt)
        src_mask = src_mask[:, :, :50]
        embeddings = self.tgt_embed(tgt) # [batch_size, seq_len, 768]
        # print('embeddings',embeddings.size())
        # exit()
        
        # Memory querying and responding for textual features
        dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0),
                                                                memory_matrix.size(1)) # [batch_size, cmm_size, cmm_dim]
        responses = self.cmn(embeddings, dummy_memory_matrix, dummy_memory_matrix, tgt = tgt) # [batch_size, seq_len, 768]
        embeddings = embeddings + responses
        # Memory querying and responding for textual features

        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)


class BaseCMN(AttModel):
    def make_model(self, tgt_vocab, cmn):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
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

        tgt_vocab = self.vocab_size + 1

        # !!! mapping by disease classes
        self.semantic = SemanticMap(self.d_model, 80, self.dropout)
        self.cmn = MultiThreadMemory(args.num_heads, args.d_model, topk=args.topk)

        self.model = self.make_model(tgt_vocab, self.cmn)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        self.memory_matrix = nn.Parameter(torch.FloatTensor(args.d_vf, args.cmm_dim)) # 多模态记忆 memory matrix 2048个patches 
        nn.init.normal_(self.memory_matrix, 0, 1 / args.cmm_dim)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None): # [batch_size, 98, d_vf], None, [batch_size, max_seq_len]
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks) # [batch_size, 98, d_model]
        # !!! mapping by disease classes


        
        # Memory querying and responding for visual features
        # image_memory_matrix = self.memory_matrix.view(98, 32, -1) # 20 融合之后的token数
        dummy_memory_matrix = self.memory_matrix.unsqueeze(0).expand(att_feats.size(0), self.memory_matrix.size(0),
                                                                      self.memory_matrix.size(1)) # [batch_size, cmm_size, cmm_dim]
        responses = self.cmn(att_feats, dummy_memory_matrix, dummy_memory_matrix) # [batch_size, 98, d_model]
        # dummy_memory_matrix = image_memory_matrix.unsqueeze(0)
        # dummy_memory_matrix = torch.sum(dummy_memory_matrix, dim = 2)
        att_feats = att_feats + responses
        att_feats = self.semantic(att_feats)
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long) # [batch_size, 98]
        # Memory querying and responding for visual features
        
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
        # print(seq_mask.size())
        # exit()
        out = self.model(att_feats, seq, att_masks, seq_mask, memory_matrix=self.memory_matrix) # [batch_size, seq_len, d_model]
        outputs = F.log_softmax(self.logit(out), dim=-1) # [batch_size, max_seq_len-1, vocab_size+1]

        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1) # [batch_size, 1, emb_dim]
            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model)]
        else:
            # print('0ys',ys.size())
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        out, past = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device),
                                      past=past, memory_matrix=self.memory_matrix)
        return out[:, -1], [ys.unsqueeze(0)] + past
