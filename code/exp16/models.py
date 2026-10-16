# ============================================================
# [열람용 사본] 2026-10-16 시점 스냅샷. 이 파일은 실행용이 아님.
# 실제로 돌아가는 원본 경로:
#   /home/user/onsesang/material_span_grounding/experiments/16_strong_recommender_tactile/models.py
# 원본은 이후 실험으로 바뀔 수 있음.
# 역할: Exp16 비교군 BPR·LightGCN과 기준 순차 모델 SASRec을 정의한다.
# ============================================================
"""Transparent BPR-MF, normalized LightGCN, causal SASRec baselines.

Equations follow the original model papers. These are local implementations,
not claims to reproduce authors' published Amazon Clothing benchmark metrics.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

# 사용자·상품 ID embedding의 pairwise ranking 기준 모델이다.
class BPR(nn.Module):
    def __init__(self,nusers,nitems,dim=64,**kwargs):
        super().__init__()
        self.user=nn.Embedding(nusers,dim); self.item=nn.Embedding(nitems,dim)
        nn.init.normal_(self.user.weight,std=.01);nn.init.normal_(self.item.weight,std=.01)

    def representations(self): return self.user.weight,self.item.weight

    def loss(self,u,p,n,reg=1e-6):
        ue,ie=self.representations();a,b,c=ue[u],ie[p],ie[n]
        rank=F.softplus(-(a*(b-c)).sum(-1)).mean()
        penalty=(self.user(u).square().sum()+self.item(p).square().sum()+self.item(n).square().sum())/len(u)
        return rank+reg*penalty

# 사용자-상품 그래프에서 층별 전파 표현을 평균하는 LightGCN 기준 모델이다.
class LightGCN(BPR):
    def __init__(self,nusers,nitems,edges,dim=64,layers=3,**kwargs):
        super().__init__(nusers,nitems,dim)
        self.nusers=nusers;self.layers=layers
        u,p=edges
        src=torch.cat([u,p+nusers]);dst=torch.cat([p+nusers,u])
        deg=torch.bincount(src,minlength=nusers+nitems).float().clamp_min(1)
        weights=deg[src].rsqrt()*deg[dst].rsqrt()
        self.register_buffer('adj',torch.sparse_coo_tensor(torch.stack([src,dst]),weights,(nusers+nitems,nusers+nitems)).coalesce())

    def representations(self):
        x=torch.cat([self.user.weight,self.item.weight]); xs=[x]
        for _ in range(self.layers):
            x=torch.sparse.mm(self.adj,x);xs.append(x)
        x=torch.stack(xs).mean(0)
        return x[:self.nusers],x[self.nusers:]

# causal self-attention과 FFN으로 SASRec 한 층을 구성한다.
class SASBlock(nn.Module):
    def __init__(self,dim,heads,dropout):
        super().__init__()
        self.attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.ln1=nn.LayerNorm(dim);self.ln2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(dim,dim),nn.Dropout(dropout))

    def forward(self,x,padding):
        q=self.ln1(x)
        causal=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),1)
        x=q+self.attn(q,x,x,attn_mask=causal,need_weights=False)[0]
        x=self.ln2(x);x=x+self.ff(x)
        return x.masked_fill(padding.unsqueeze(-1),0)

# 상품 ID 시퀀스로 다음 상품을 예측하는 Exp16 SASRec 기준 모델이다.
class SASRec(nn.Module):
    def __init__(self,nitems,dim=64,layers=2,heads=2,maxlen=50,dropout=.2,**kwargs):
        super().__init__(); self.dim=dim;self.maxlen=maxlen
        # zero denotes padding, real global catalog IDs are shifted by one.
        self.item=nn.Embedding(nitems+1,dim,padding_idx=0)
        self.pos=nn.Embedding(maxlen,dim);self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([SASBlock(dim,heads,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim)
        nn.init.normal_(self.item.weight,std=.02)
        with torch.no_grad(): self.item.weight[0].zero_()

    def forward(self,seq):
        # 0 padding을 유지한 채 상품·위치 embedding을 causal 블록에 입력한다.
        pad=seq==0
        x=self.drop(self.item(seq)*math.sqrt(self.dim)+self.pos(torch.arange(seq.shape[1],device=seq.device)))
        x=x.masked_fill(pad.unsqueeze(-1),0)
        for block in self.blocks:x=block(x,pad)
        return self.norm(x)

    def loss(self,seq,pos,neg):
        # 각 위치의 다음 상품 양성과 균등 sampled 음성을 분리하도록 학습한다.
        z=self(seq);valid=pos>0
        a=(z*self.item(pos)).sum(-1)[valid]
        b=(z*self.item(neg)).sum(-1)[valid]
        return F.softplus(-a).mean()+F.softplus(b).mean()
