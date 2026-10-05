from __future__ import annotations
import hashlib, json, random, signal, time
from pathlib import Path
import torch
from .. import registry
from ..train.trainer import save_checkpoint, load_checkpoint, retain_checkpoints
from ..util import ROOT, disk_guard, write_json, git_hash
from .data import iter_cleanup_texts, split_for_eval, punctuation_stats
from .corrupt import corrupt
from .evaluation import evaluate_pairs, real_eval_pairs
from .model import CleanupTransformer

def _plots(run_dir):
    import matplotlib.pyplot as plt
    rows=[json.loads(x) for x in (run_dir/"metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    eval_rows=[json.loads(x) for x in (run_dir/"eval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for key,label in (("loss","Loss"),("lr","Learning rate")):
        plt.figure(); plt.plot([x["step"] for x in rows],[x.get(key,0) for x in rows]); plt.xlabel("Step"); plt.ylabel(label); plt.tight_layout(); plt.savefig(run_dir/f"{key}.png"); plt.close()
    plt.figure(); plt.plot([x["step"] for x in eval_rows],[x["cer"] for x in eval_rows]); plt.xlabel("Step"); plt.ylabel("Held-out CER"); plt.tight_layout(); plt.savefig(run_dir/"cer.png"); plt.close()

def _seed(text,epoch=0):
    digest=hashlib.sha256(f"{epoch}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8],"big")

def _pair(text,epoch):
    rng=random.Random(_seed(text,epoch))
    if rng.random()<.15: return text,text
    return corrupt(text,rng),text

def _encoded_batches(texts,sp,batch_size,max_length,epoch,validation=False):
    batch=[]; bos=sp.bos_id(); eos=sp.eos_id(); pad=sp.get_piece_size()
    for text in texts:
        if split_for_eval(text)!=validation: continue
        source,target=_pair(text,0 if validation else epoch)
        src=sp.encode(source,out_type=int)[:max_length-1]+[eos]
        tgt=sp.encode(target,out_type=int)[:max_length-1]+[eos]
        decoder=[bos]+tgt[:-1]
        batch.append((src,decoder,tgt,source,target))
        if len(batch)>=batch_size:
            yield _collate(batch,pad); batch=[]
    if batch: yield _collate(batch,pad)

def _collate(batch,pad):
    from torch.nn.utils.rnn import pad_sequence
    src=[torch.tensor(x[0],dtype=torch.long) for x in batch]; dec=[torch.tensor(x[1],dtype=torch.long) for x in batch]; tgt=[torch.tensor(x[2],dtype=torch.long) for x in batch]
    return pad_sequence(src,batch_first=True,padding_value=pad),pad_sequence(dec,batch_first=True,padding_value=pad),pad_sequence(tgt,batch_first=True,padding_value=pad),[(x[3],x[4]) for x in batch]

def _validation_pairs(epoch=0):
    for text in iter_cleanup_texts():
        if split_for_eval(text): yield _pair(text,0)

def train_cleanup(config,resume=False,max_minutes=None):
    from ..tokenizer import load
    sp=load(); db=registry.connect(); cfg=config["cleanup"]
    rows=db.execute("SELECT * FROM shards WHERE consumed_cleanup=0 ORDER BY name").fetchall()
    if not rows and not any((ROOT/"data/text").glob("**/*")): raise SystemExit("No cleanup text. Scan speech shards or add clean text under data/text/.")
    stats=punctuation_stats(iter_cleanup_texts())
    if stats["sentences"]==0: raise SystemExit("No registered transcripts or text corpus found.")
    device="cuda" if torch.cuda.is_available() else "cpu"; model_cfg={k:cfg[k] for k in ("d_model","encoder_layers","decoder_layers","heads") if k in cfg}
    stage_no=1+db.execute("SELECT COUNT(*) FROM stages WHERE kind='cleanup'").fetchone()[0]; stage=f"cleanup-{stage_no:03d}"
    model=CleanupTransformer(sp.get_piece_size(),max_len=int(cfg.get("max_length",256)),**model_cfg).to(device)
    if stage_no>1:
        previous=ROOT/"checkpoints"/f"cleanup-{stage_no-1:03d}"/"best.pt"
        if previous.exists(): model.load_state_dict(torch.load(previous,map_location=device,weights_only=False)["model"])
    run_dir=ROOT/"runs"/stage; ckpt_dir=ROOT/"checkpoints"/stage; run_dir.mkdir(parents=True,exist_ok=True); ckpt_dir.mkdir(parents=True,exist_ok=True)
    write_json(run_dir/"config.json",config); write_json(run_dir/"hardware.json",{"device":device,"cuda":torch.cuda.get_device_name() if device=="cuda" else None}); write_json(run_dir/"shards.json",[{"name":r["name"],"sha256":r["sha256"],"rows":r["rows"],"hours":r["hours"]} for r in rows]); write_json(run_dir/"corpus.json",stats)
    model_cfg["max_len"]=int(cfg.get("max_length",256)); write_json(run_dir/"model.json",{"parameters":model.param_count(),"config":model_cfg,"tokenizer_hash":registry.get_meta("tokenizer_hash"),"tokenizer_choice":"Reuse frozen ASR SentencePiece tokenizer; Hindi, Latin, and byte fallback already covered."})
    peak_lr=float(cfg.get("learning_rate",2e-4))*(float(cfg.get("stage_lr_scale",.3)) if stage_no>1 else 1.)
    optimizer=torch.optim.AdamW(model.parameters(),lr=peak_lr)
    epochs=int(cfg.get("epochs",1)); batch_size=int(cfg.get("batch_size",32)); accum=max(1,int(cfg.get("grad_accum",1))); warmup=max(1,int(cfg.get("warmup_steps",100)))
    total=max(warmup+1,int(stats["sentences"]*.9/max(batch_size*accum,1))*epochs)
    def lr_scale(n):
        if n<warmup:return max(.01,(n+1)/warmup)
        return .5*(1+__import__("math").cos(__import__("math").pi*min(1.,(n-warmup)/(total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale)
    scaler=torch.amp.GradScaler("cuda",enabled=False); step=0; epoch_start=0; batch_start=0; last_loss=None; ckpts=sorted(ckpt_dir.glob("step-*.pt"),reverse=True)
    if resume:
        if not ckpts: raise SystemExit("No cleanup checkpoint to resume.")
        state=load_checkpoint(ckpts[0],model,optimizer,scheduler,scaler,map_location=device)
        if state.get("tokenizer_hash")!=registry.get_meta("tokenizer_hash"): raise SystemExit("Cleanup checkpoint tokenizer hash mismatch.")
        if state.get("train_config")!=config: raise SystemExit("Resume config differs from cleanup checkpoint; restore its YAML values first.")
        step=state["step"]; last_loss=state.get("loss"); cursor=state.get("data_state") or {}; epoch_start=cursor.get("epoch",0); batch_start=cursor.get("batch",0)
    began=time.monotonic(); metrics=run_dir/"metrics.jsonl"; eval_file=run_dir/"eval.jsonl"; max_len=int(cfg.get("max_length",256)); optimizer.zero_grad(set_to_none=True)
    stop_requested=[False]
    old_sigint=signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT,lambda *_: stop_requested.__setitem__(0,True))
    for epoch in range(epoch_start,int(cfg.get("epochs",1))):
        skipped=batch_start if epoch==epoch_start else 0; last_batch_index=None
        for batch_index,batch in enumerate(_encoded_batches(iter_cleanup_texts(),sp,batch_size,max_len,epoch)):
            if batch_index<skipped: continue
            last_batch_index=batch_index
            step_started=time.monotonic()
            source,decoder,target,_=batch; source=source.to(device); decoder=decoder.to(device); target=target.to(device)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device=="cuda"):
                logits=model(source,decoder); loss=torch.nn.functional.cross_entropy(logits.reshape(-1,sp.get_piece_size()),target.reshape(-1),ignore_index=sp.get_piece_size())/accum
            loss.backward(); last_loss=float(loss.detach())*accum
            if (batch_index+1)%accum==0:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step+=1
                elapsed=time.monotonic()-step_started
                row={"step":step,"epoch":epoch,"loss":last_loss,"lr":optimizer.param_groups[0]["lr"],"grad_norm":float(norm),"tokens_per_sec":int(target.ne(sp.get_piece_size()).sum())/max(elapsed,1e-6),"gpu_memory_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,"step_seconds":elapsed}
                with metrics.open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
                should_stop=stop_requested[0] or max_minutes is not None and (time.monotonic()-began)/60>=max_minutes
                if step%int(cfg.get("checkpoint_every",100))==0 or should_stop:
                    path=ckpt_dir/f"step-{step:08d}.pt"; save_checkpoint(path,model,optimizer,scheduler,scaler,step,{"epoch":epoch,"batch":batch_index+1},last_loss,{"tokenizer_hash":registry.get_meta("tokenizer_hash"),"model_config":model_cfg,"train_config":config}); retain_checkpoints(ckpt_dir)
                if should_stop:
                    signal.signal(signal.SIGINT,old_sigint)
                    return f"Saved cleanup checkpoint at step {step}; shards remain unconsumed. Resume with --resume."
        if last_batch_index is not None and (last_batch_index+1)%accum:
            remainder=(last_batch_index+1)%accum
            for parameter in model.parameters():
                if parameter.grad is not None: parameter.grad.mul_(accum/remainder)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step+=1
            row={"step":step,"epoch":epoch,"loss":last_loss,"lr":optimizer.param_groups[0]["lr"],"grad_norm":float(norm),"tokens_per_sec":0,"gpu_memory_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,"step_seconds":0.0}
            with metrics.open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
            path=ckpt_dir/f"step-{step:08d}.pt"; save_checkpoint(path,model,optimizer,scheduler,scaler,step,{"epoch":epoch+1,"batch":0},last_loss,{"tokenizer_hash":registry.get_meta("tokenizer_hash"),"model_config":model_cfg,"train_config":config}); retain_checkpoints(ckpt_dir)
            if stop_requested[0] or max_minutes is not None and (time.monotonic()-began)/60>=max_minutes:
                signal.signal(signal.SIGINT,old_sigint)
                return f"Saved cleanup checkpoint at step {step}; shards remain unconsumed. Resume with --resume."
        batch_start=0
    signal.signal(signal.SIGINT,old_sigint)
    if step:
        final_state=ckpt_dir/f"step-{step:08d}.pt"
        save_checkpoint(final_state,model,optimizer,scheduler,scaler,step,{"epoch":int(cfg.get("epochs",1)),"batch":0},last_loss,{"tokenizer_hash":registry.get_meta("tokenizer_hash"),"model_config":model_cfg,"train_config":config})
        retain_checkpoints(ckpt_dir)
    scores,samples=evaluate_pairs(model,sp,_validation_pairs(),device,max_len)
    if not scores["count"]: raise SystemExit("Held-out cleanup split is empty; add more distinct sentences.")
    real_pairs=real_eval_pairs(); real_scores,_=evaluate_pairs(model,sp,real_pairs,device,max_len) if real_pairs else ({"count":0},[])
    best={"model":model.state_dict(),"model_config":model_cfg,"tokenizer_hash":registry.get_meta("tokenizer_hash"),"evaluation":scores}
    disk_guard(ckpt_dir/"best.pt",needed_bytes=model.param_count()*4,reserve_bytes=500_000_000)
    torch.save(best,ckpt_dir/"best.pt")
    (ROOT/"checkpoints").mkdir(exist_ok=True)
    disk_guard(ROOT/"checkpoints/best-cleanup.pt",needed_bytes=model.param_count()*4,reserve_bytes=500_000_000)
    torch.save(best,ROOT/"checkpoints/best-cleanup.pt")
    eval_record={"step":step,"cer":scores["cer"],"exact_match":scores["exact_match"],"over_edit_rate":scores["over_edit_rate"],"length_ratio_violations":scores["length_ratio_violations"]}
    with eval_file.open("a",encoding="utf-8") as f: f.write(json.dumps(eval_record)+"\n")
    write_json(run_dir/"eval.json",{"heldout":scores,"real_pairs":real_scores}); (run_dir/"samples.txt").write_text("\n".join(str(x) for x in samples),encoding="utf-8")
    beats=scores["exact_match"]>scores["rule_baseline"]["exact_match"] or scores["cer"]<scores["rule_baseline"]["cer"]
    record={"steps":step,"parameters":model.param_count(),"loss":last_loss,"punctuation":stats,"evaluation":scores,"real_pairs_evaluation":real_scores,"beats_rule_baseline":beats,"tokenizer_hash":registry.get_meta("tokenizer_hash"),"wall_seconds":time.monotonic()-began,"git_hash":git_hash(),"config":config,"hardware":{"device":device,"cuda":torch.cuda.get_device_name() if device=="cuda" else None},"shards":[{"name":r["name"],"sha256":r["sha256"],"rows":r["rows"],"hours":r["hours"]} for r in rows],"metrics":[json.loads(x) for x in metrics.read_text(encoding="utf-8").splitlines() if x.strip()],"evaluations":[json.loads(x) for x in eval_file.read_text(encoding="utf-8").splitlines() if x.strip()]}
    write_json(run_dir/"metrics.json",record)
    punct_msg="Most corpus sentences lack punctuation; add clean, punctuated material to data/text/." if stats["punctuation_rate"]<.5 else "Corpus contains punctuated sentences."
    baseline_msg="Cleanup model beats rule baseline." if record["beats_rule_baseline"] else "Cleanup model does not beat rule baseline."
    (run_dir/"report.md").write_text(f"# {stage}\n\nSentences: {stats['sentences']}\n\nPunctuation rate: {stats['punctuation_rate']:.3f}\n\n{punct_msg}\n\nHeld-out exact match: {scores['exact_match']:.3f}\n\nHeld-out CER: {scores['cer']:.3f}\n\nReal-pair exact match: {real_scores.get('exact_match')}\n\nReal-pair CER: {real_scores.get('cer')}\n\n{baseline_msg}\n",encoding="utf-8")
    _plots(run_dir)
    db.execute("UPDATE shards SET consumed_cleanup=1 WHERE name IN ("+",".join("?" for _ in rows)+")",[r["name"] for r in rows]) if rows else None
    db.execute("INSERT OR REPLACE INTO stages(name,kind,record_json) VALUES(?,?,?)",(stage,"cleanup",json.dumps(record))); db.commit()
    return f"Finished {stage}: exact {scores['exact_match']:.3f}, CER {scores['cer']:.3f}, rule exact {scores['rule_baseline']['exact_match']:.3f}. {punct_msg}"
