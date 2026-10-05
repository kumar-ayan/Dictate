from __future__ import annotations
import csv
from pathlib import Path
from ..util import ROOT
from ..data.parquet import normalize_text
from .rules import clean_rules

def _distance(left,right):
    prev=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        curr=[i]
        for j,b in enumerate(right,1): curr.append(min(curr[-1]+1,prev[j]+1,prev[j-1]+(a!=b)))
        prev=curr
    return prev[-1]

def generate(model,sp,text,device,max_length=256):
    import torch
    source=sp.encode(text,out_type=int)[:max_length-1]+[sp.eos_id()]
    decode_limit=min(max_length,max(16,len(source)+16))
    src=torch.tensor(source,dtype=torch.long,device=device)[None,:]
    decoder=torch.tensor([[sp.bos_id()]],dtype=torch.long,device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(decode_limit-1):
            logits=model(src,decoder); token=int(logits[0,-1].argmax())
            if token==sp.eos_id(): break
            decoder=torch.cat((decoder,torch.tensor([[token]],device=device)),dim=1)
    return normalize_text(sp.decode(decoder[0,1:].tolist()))

def evaluate_pairs(model,sp,pairs,device,max_length=256):
    exact=cer_edits=chars=clean_count=over_edits=violations=count=0; samples=[]; baseline_exact=baseline_edits=0
    for source,target in pairs:
        count+=1
        output=generate(model,sp,source,device,max_length)
        baseline=normalize_text(clean_rules(source)); target=normalize_text(target)
        exact+=output==target; baseline_exact+=baseline==target
        cer_edits+=_distance(list(output),list(target)); chars+=len(target)
        baseline_edits+=_distance(list(baseline),list(target))
        clean=normalize_text(source)==target
        if clean:
            clean_count+=1; over_edits+=output!=target
        ratio=len(output)/max(len(source),1); violations+=ratio<.5 or ratio>1.5
        if len(samples)<40: samples.append({"input":source,"target":target,"model":output,"rules":baseline})
    n=max(count,1)
    return {"count":count,"exact_match":exact/n,"cer":cer_edits/max(chars,1),"clean_input_count":clean_count,"over_edit_rate":over_edits/max(clean_count,1),"length_ratio_violations":violations/n,"rule_baseline":{"exact_match":baseline_exact/n,"cer":baseline_edits/max(chars,1)}},samples

def real_eval_pairs(path=None):
    path=Path(path) if path else ROOT/"evals/cleanup_real.tsv"
    if not path.exists(): return []
    with path.open(encoding="utf-8",newline="") as f:
        return [(row["input"],row["target"]) for row in csv.DictReader(f,delimiter="\t") if row.get("input") and row.get("target")]
