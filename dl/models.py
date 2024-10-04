import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F

from dataclasses import dataclass
from dl.modules import NewGELU, cross_entropy_logits

import tqdm
import math
import os
import urllib


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nheads = config.nheads
        self.hdim = config.hdim
        # q, k, v combined in 1 matrix (to be split later).
        # We assume input dimension is same as output dimension after projections
        self.qkv = nn.parameter.Parameter(torch.rand(size=(config.hdim, 3 * config.nheads * config.hdim), dtype=torch.float32) / math.sqrt(config.hdim))
        self.projection = nn.parameter.Parameter(torch.rand(size=(config.hdim * config.nheads, config.hdim), dtype=torch.float32) / math.sqrt(config.hdim))
        # which cells to mask? the upper left by setting them to 0 (diagonal is not masked)
        mask = torch.tril(torch.ones(config.seq_len,config.seq_len))
        self.register_buffer("mask", mask)
        self.dropout = nn.Dropout(p=config.att_drop)
        

    def forward(self, x):
        # x: (batch x sequence x hdim)
        B, S, H = x.size()
        
        # split into Q K V matrices. Each of shape [hdim, nheads * hdim]. 
        # alternatively: x =  x @ self.qkv
        Q, K, V = torch.split(self.qkv, H * self.nheads, dim=1)

        # project input -> q,k,v (batch x nheads x seq x hdim)
        # [B, S, H] x [H, nheads x H] -> [B, S, nheads x H] -> view -> [B, nheads, S, H]
        q = (x @ Q).view(B,self.nheads,S,H)
        k = (x @ K).view(B,self.nheads,S,H)
        v = (x @ V).view(B,self.nheads,S,H)

        # calculate dot product (batch x heads x seq(queries) x seq(keys))
        att = torch.matmul(q, k.transpose(3,2)) * (1.0 / math.sqrt(self.hdim))

        # Only difference of causal self-att.
        # Causal masking (in place):  set all items to the right of index i,i to zero (-inf before softmax)
        att.masked_fill_(torch.eq(self.mask.view(1,1,S,S), torch.scalar_tensor(0)), -torch.inf)

        # Apply softmax then droptout
        att = F.softmax(att, dim=-1) # same shape
        att = self.dropout(att)
        
        out = att @ v # batch x heads x seq x hdim

        # Project from all heads back to hdim
        out = out.view(B,S,-1) # concat all heads: batch x seq x hdim*nheads
        out = out @ self.projection # batch x seq x hdim
        #print(out.mean())
        return out


class AttBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hdim)
        self.layer_norm2 = nn.LayerNorm(config.hdim)
        
        ffexpand = nn.Linear(config.hdim, 4 * config.hdim, bias=False)
        gelu = NewGELU()
        ffshrink = nn.Linear(4 * config.hdim, config.hdim, bias=False)
        self.res_drop = nn.Dropout(config.res_drop)
        
        self.att = SelfAttention(config)
        self.mlp = nn.Sequential(ffexpand, gelu, ffshrink, self.res_drop)
        

    def forward(self, x):
        # x: (batch x sequence)
        #print(x.size())

        out = self.att(self.layer_norm1(x))
        out = self.res_drop(out)
        out = self.layer_norm2(x + out)
        out = self.mlp(out)
        
        return out

# todo: positional encoding
class MyModel(nn.Module):
    def __init__(self, config, layers=3):
        super().__init__()
        self.vocab_size = config.vocab
        self.emb = torch.nn.Embedding(config.vocab, config.hdim)
        self.emb_drop = nn.Dropout(config.emb_drop)
        self.layers = nn.Sequential(*[AttBlock(config) for _ in range(config.layers)])
        self.last_layer_norm = nn.LayerNorm(config.hdim)
        self.llm_head = nn.parameter.Parameter(torch.rand(size=(config.hdim, config.vocab), dtype=torch.float32) / math.sqrt(config.vocab))
        self.loss = cross_entropy_logits
        print(f"Total parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, x):
        outs = self.emb(x[:,:-1])
        outs = self.emb_drop(outs)
        B, T, nh = outs.size()
        outs = self.layers(outs)
        outs = self.last_layer_norm(outs)
        outs = outs @ self.llm_head # B x S X vocab_size
        #print(f"Output shape: {outs.shape}")
        loss = self.loss(outs.view(-1,self.vocab_size), x[:,1:].contiguous().view(-1), reduce=True)
        return loss, torch.argmax(outs, dim=-1)
