from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class ConvSubsample(nn.Module):
    def __init__(self,d):
        super().__init__(); self.conv=nn.Sequential(nn.Conv2d(1,d,3,2,1),nn.GELU(),nn.Conv2d(d,d,3,2,1),nn.GELU())
    def forward(self,x):
        # B,T,F -> B,T/4,D
        x=self.conv(x.unsqueeze(1)); b,d,t,f=x.shape
        return x.transpose(1,2).reshape(b,t,d*f)

class FeedForward(nn.Module):
    def __init__(self,d,m,drop):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,m),nn.SiLU(),nn.Dropout(drop),nn.Linear(m,d),nn.Dropout(drop))
    def forward(self,x): return self.net(x)

class RotarySelfAttention(nn.Module):
    def __init__(self,d_model,heads,dropout):
        super().__init__(); self.heads=heads; self.head_dim=d_model//heads
        self.qkv=nn.Linear(d_model,3*d_model); self.out=nn.Linear(d_model,d_model); self.dropout=nn.Dropout(dropout)
    def _rotate(self,x):
        d=x.shape[-1]; half=d//2
        freq=torch.exp(-torch.log(torch.tensor(10000.,device=x.device))*torch.arange(half,device=x.device,dtype=torch.float32)/max(half,1))
        pos=torch.arange(x.shape[-2],device=x.device,dtype=torch.float32); phase=pos[:,None]*freq[None,:]
        c=phase.cos().to(x.dtype)[None,None]; s=phase.sin().to(x.dtype)[None,None]; a,b=x[...,:half],x[...,half:2*half]
        return torch.cat((a*c-b*s,a*s+b*c,x[...,2*half:]),dim=-1)
    def forward(self,x,key_padding_mask=None):
        b,t,d=x.shape; q,k,v=self.qkv(x).view(b,t,3,self.heads,self.head_dim).permute(2,0,3,1,4)
        q=self._rotate(q); k=self._rotate(k); scores=(q@k.transpose(-2,-1))/self.head_dim**.5
        if key_padding_mask is not None: scores=scores.masked_fill(key_padding_mask[:,None,None,:],torch.finfo(scores.dtype).min)
        probs=torch.softmax(scores.float(),dim=-1).to(scores.dtype); y=self.dropout(probs)@v
        return self.out(y.transpose(1,2).contiguous().view(b,t,d))

class ConformerBlock(nn.Module):
    def __init__(self,d,h,m,drop):
        super().__init__(); self.ff1=FeedForward(d,m,drop); self.attn=RotarySelfAttention(d,h,drop); self.an=nn.LayerNorm(d); self.cn=nn.LayerNorm(d); self.pw1=nn.Conv1d(d,2*d,1); self.dw=nn.Conv1d(d,d,15,padding=7,groups=d); self.pw2=nn.Conv1d(d,d,1); self.ff2=FeedForward(d,m,drop); self.out=nn.LayerNorm(d)
    def forward(self,x,key_padding_mask=None):
        x=x+.5*self.ff1(x); x=x+self.attn(self.an(x),key_padding_mask); z=self.cn(x).transpose(1,2); a,b=self.pw1(z).chunk(2,dim=1); z=self.dw(a*torch.sigmoid(b)); x=x+self.pw2(z).transpose(1,2); x=x+.5*self.ff2(x); return self.out(x)

class ConformerCTC(nn.Module):
    def __init__(self,vocab_size,d_model=512,layers=8,heads=8,ff_mult=4,dropout=.1,n_mels=80,gradient_checkpointing=False):
        super().__init__(); self.blank_id=vocab_size; self.subsample=ConvSubsample(d_model); self.project=nn.Linear(d_model*((n_mels+3)//4),d_model); self.blocks=nn.ModuleList([ConformerBlock(d_model,heads,d_model*ff_mult,dropout) for _ in range(layers)]); self.classifier=nn.Linear(d_model,vocab_size+1); self.ctc=nn.CTCLoss(blank=self.blank_id,zero_infinity=True); self.gradient_checkpointing=gradient_checkpointing
    def forward(self,features,lengths=None):
        x=self.project(self.subsample(features))
        if lengths is None: lengths=torch.full((features.shape[0],),features.shape[1],device=features.device,dtype=torch.long)
        out_lengths=(lengths+3)//4; mask=torch.arange(x.shape[1],device=x.device)[None,:]>=out_lengths[:,None]
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x=checkpoint(block,x,mask,use_reentrant=False)
            else: x=block(x,mask)
        return self.classifier(x).log_softmax(-1)
    def param_count(self): return sum(p.numel() for p in self.parameters())
