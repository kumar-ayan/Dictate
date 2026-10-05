from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self,width,eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(width)); self.eps=eps
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight

def _rope(x):
    d=x.shape[-1]; half=d//2
    freq=torch.exp(-math.log(10000.)*torch.arange(half,device=x.device,dtype=torch.float32)/max(half,1))
    pos=torch.arange(x.shape[-2],device=x.device,dtype=torch.float32)
    angle=pos[:,None]*freq[None,:]; c=angle.cos().to(x.dtype)[None,None,:,:]; s=angle.sin().to(x.dtype)[None,None,:,:]
    a,b=x[...,:half],x[...,half:2*half]
    rotated=torch.cat((a*c-b*s,a*s+b*c),dim=-1)
    return torch.cat((rotated,x[...,2*half:]),dim=-1)

class Attention(nn.Module):
    def __init__(self,width,heads):
        super().__init__(); self.heads=heads; self.head_dim=width//heads
        self.q=nn.Linear(width,width,bias=False); self.k=nn.Linear(width,width,bias=False); self.v=nn.Linear(width,width,bias=False); self.out=nn.Linear(width,width,bias=False)
    def forward(self,query,context=None,key_padding=None,causal=False):
        context=query if context is None else context; b,t,d=query.shape; source=context.shape[1]
        q=self.q(query).view(b,t,self.heads,self.head_dim).transpose(1,2)
        k=self.k(context).view(b,source,self.heads,self.head_dim).transpose(1,2)
        v=self.v(context).view(b,source,self.heads,self.head_dim).transpose(1,2)
        q=_rope(q); k=_rope(k); scores=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        if causal: scores=scores.masked_fill(torch.ones((t,source),device=query.device,dtype=torch.bool).triu(1)[None,None],torch.finfo(scores.dtype).min)
        if key_padding is not None: scores=scores.masked_fill(key_padding[:,None,None,:],torch.finfo(scores.dtype).min)
        probs=torch.softmax(scores.float(),dim=-1).to(scores.dtype); value=probs@v
        return self.out(value.transpose(1,2).contiguous().view(b,t,d))

class SwiGLU(nn.Module):
    def __init__(self,width,mult=4):
        super().__init__(); self.in_proj=nn.Linear(width,width*mult*2); self.out_proj=nn.Linear(width*mult,width)
    def forward(self,x):
        gate,value=self.in_proj(x).chunk(2,dim=-1)
        return self.out_proj(F.silu(gate)*value)

class EncoderBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__(); self.n1=RMSNorm(width); self.attn=Attention(width,heads); self.n2=RMSNorm(width); self.ff=SwiGLU(width)
    def forward(self,x,pad):
        x=x+self.attn(self.n1(x),key_padding=pad); return x+self.ff(self.n2(x))

class DecoderBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__(); self.n1=RMSNorm(width); self.self_attn=Attention(width,heads); self.n2=RMSNorm(width); self.cross=Attention(width,heads); self.n3=RMSNorm(width); self.ff=SwiGLU(width)
    def forward(self,x,memory,src_pad,tgt_pad):
        x=x+self.self_attn(self.n1(x),key_padding=tgt_pad,causal=True)
        x=x+self.cross(self.n2(x),context=memory,key_padding=src_pad)
        return x+self.ff(self.n3(x))

class CleanupTransformer(nn.Module):
    """Small hand-written encoder-decoder with RoPE, RMSNorm, and SwiGLU."""
    def __init__(self,vocab_size,d_model=512,encoder_layers=3,decoder_layers=3,heads=8,max_len=512):
        super().__init__(); self.vocab_size=vocab_size; self.pad_id=vocab_size
        self.embedding=nn.Embedding(vocab_size+1,d_model,padding_idx=self.pad_id)
        self.encoder=nn.ModuleList([EncoderBlock(d_model,heads) for _ in range(encoder_layers)])
        self.decoder=nn.ModuleList([DecoderBlock(d_model,heads) for _ in range(decoder_layers)])
        self.norm=RMSNorm(d_model); self.max_len=max_len
    def forward(self,source,decoder_input):
        if source.shape[1]>self.max_len or decoder_input.shape[1]>self.max_len: raise ValueError("Cleanup sequence exceeds configured max_len")
        src_pad=source.eq(self.pad_id); tgt_pad=decoder_input.eq(self.pad_id)
        memory=self.embedding(source)
        for layer in self.encoder: memory=layer(memory,src_pad)
        hidden=self.embedding(decoder_input)
        for layer in self.decoder: hidden=layer(hidden,memory,src_pad,tgt_pad)
        return F.linear(self.norm(hidden),self.embedding.weight[:self.vocab_size])
    def param_count(self): return sum(p.numel() for p in self.parameters())
