from __future__ import annotations
import hashlib, json, math, random, shutil, signal, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .. import registry
from ..data.parquet import iter_rows, normalize_text, decode_audio, log_mel
from ..util import ROOT, disk_guard, write_json, git_hash

def _speaker_excluded(row, audio_column=None,speakers=None,hashes=None):
    speakers=set(json.loads(registry.get_meta("dev_speakers","[]") or "[]")) if speakers is None else speakers
    speaker=row.get("speaker_id",row.get("speaker"))
    if speaker is not None and str(speaker) in speakers: return True
    if audio_column:
        hashes=set(json.loads(registry.get_meta("dev_audio_hashes","[]") or "[]")) if hashes is None else hashes
        if hashes:
            from ..data.parquet import hash_audio_payload
            try: return hash_audio_payload(row.get(audio_column)) in hashes
            except (TypeError,ValueError): return False
    return False

def _feature_batch(rows, sp, device):
    feats=[]; targets=[]; durations=[]; positions=[]
    for row,pos in rows:
        text=normalize_text(str(row.get("__text","") or ""))
        if not text: continue
        try:
            wave,sr=row["__wave"],row["__sr"]; duration=len(wave)/sr
            if duration<1 or duration>30: continue
            feature=log_mel(wave,sr,device=device).squeeze(0).transpose(0,1).contiguous()
            ids=sp.encode(text,out_type=int)
            if not ids: continue
            feats.append(feature); targets.append(torch.tensor(ids,dtype=torch.long)); durations.append(duration); positions.append(pos)
        except Exception:
            continue
    if not feats: return None
    feature_lengths=torch.tensor([x.shape[0] for x in feats],dtype=torch.long)
    return pad_sequence(feats,batch_first=True),targets,feature_lengths

def _decode(log_probs, blank):
    out=[]
    for row in log_probs.argmax(-1).tolist():
        tokens=[]; last=None
        for idx in row:
            if idx!=last and idx!=blank: tokens.append(idx)
            last=idx
        out.append(tokens)
    return out

def _edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        curr=[i]
        for j,y in enumerate(b,1): curr.append(min(curr[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=curr
    return prev[-1]

def _latest_checkpoint(paths):
    paths=list(paths)
    return max(paths,key=lambda path:(path.stat().st_mtime_ns,path.name)) if paths else None

def _validate_dev_manifest_sources():
    manifest_path=ROOT/"data/dev"/"manifest.json"
    if not manifest_path.exists(): return
    refs=json.loads(manifest_path.read_text(encoding="utf-8")).get("rows",[])
    missing=sorted({ref["path"] for ref in refs if not Path(ref["path"]).is_file()})
    if missing:
        names=", ".join(Path(path).name for path in missing)
        raise SystemExit(f"Fixed dev set references missing parquet file(s): {names}. Restore the original file(s) at their recorded paths; the fixed dev set cannot be rebuilt during training.")

def _evaluate_dev(model, sp, device):
    import pyarrow.parquet as pq
    manifest_path=ROOT/"data/dev"/"manifest.json"
    if not manifest_path.exists(): return None
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")); refs=manifest.get("rows",[])
    if not refs: return None
    _validate_dev_manifest_sources()
    groups={}
    for ref in refs: groups.setdefault((ref["path"],ref["row_group"],ref["audio_column"]),[]).append(ref)
    words=chars=wer_edits=cer_edits=examples=0; samples=[]; failures=[]; skipped=0; model.eval()
    with torch.no_grad():
        for (path,rg,acol),items in groups.items():
            pf=pq.ParquetFile(path); tab=pf.read_row_group(rg,columns=[acol]).column(0).to_pylist()
            for ref in items:
                try:
                    wave,sr=decode_audio(tab[ref["row_offset"]]); feat=log_mel(wave,sr,device=device).squeeze(0).transpose(0,1)
                    # _decode expects [batch, time, vocabulary]; keep the
                    # singleton batch dimension until greedy CTC decoding.
                    pred=sp.decode(_decode(model(feat.unsqueeze(0),torch.tensor([feat.shape[0]],device=device)),sp.get_piece_size())[0]); truth=normalize_text(ref["text"])
                    ww=truth.split(); pw=pred.split(); cc=list(truth); pc=list(pred)
                    examples+=1
                    wer_edits+=_edit_distance(ww,pw); cer_edits+=_edit_distance(cc,pc); words+=len(ww); chars+=len(cc)
                    if len(samples)<20: samples.append((truth,pred))
                except Exception as exc:
                    skipped+=1
                    if len(failures)<3: failures.append(f"{type(exc).__name__}: {exc}")
                    continue
    model.train()
    if examples==0:
        detail="; ".join(failures) if failures else "no dev rows"
        raise RuntimeError(f"Dev evaluation could not decode or infer any of {len(refs)} rows ({skipped} skipped). First errors: {detail}")
    result={"wer":wer_edits/max(words,1),"cer":cer_edits/max(chars,1),"examples":examples,"skipped":skipped}
    return result,samples

def _checkpoint_state(model,optimizer,scheduler,scaler,step,data_state,loss,model_cfg,best_wer,accumulated=0,train_config=None):
    gradients={name:p.grad.detach().cpu() for name,p in model.named_parameters() if p.grad is not None}
    return {"sampler_version":2,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"step":step,"data_state":data_state,"loss":loss,"best_wer":best_wer,"model_config":model_cfg,"train_config":train_config,"accumulated":accumulated,"gradients":gradients,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,"git_hash":git_hash()}

def _save_training_state(directory,state,step,param_count):
    path=Path(directory)/f"step-{step:08d}.pt"; size=param_count*16
    disk_guard(path,needed_bytes=size,reserve_bytes=500_000_000)
    temp=path.with_suffix(".tmp"); torch.save(state,temp); temp.replace(path)
    checkpoints=sorted(Path(directory).glob("step-*.pt"),key=lambda p:(p.stat().st_mtime_ns,p.name),reverse=True)
    for old in checkpoints[3:]: old.unlink()
    return path

def _save_asr_best(stage_path,global_path,state,score,param_count):
    disk_guard(stage_path,needed_bytes=param_count*4,reserve_bytes=500_000_000)
    torch.save(state,stage_path)
    previous_score=float("inf")
    if global_path.exists():
        previous=torch.load(global_path,map_location="cpu",weights_only=False)
        previous_score=float(previous.get("dev_wer",float("inf")))
    if score<previous_score:
        disk_guard(global_path,needed_bytes=stage_path.stat().st_size,reserve_bytes=500_000_000)
        shutil.copy2(stage_path,global_path)

def _write_plots(run_dir):
    import matplotlib.pyplot as plt
    metrics=[]
    for line in (run_dir/"metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip(): metrics.append(json.loads(line))
    evals=[]
    p=run_dir/"eval.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip(): evals.append(json.loads(line))
    for key,label in (("loss","Loss"),("lr","Learning rate")):
        plt.figure(); plt.plot([x["step"] for x in metrics],[x.get(key,0) for x in metrics]); plt.xlabel("Step"); plt.ylabel(label); plt.tight_layout(); plt.savefig(run_dir/f"{key}.png"); plt.close()
    plt.figure(); plt.plot([x["step"] for x in evals],[x["wer"] for x in evals]); plt.xlabel("Step"); plt.ylabel("Dev WER"); plt.tight_layout(); plt.savefig(run_dir/"wer.png"); plt.close()

def _jsonl(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _estimate_optimizer_steps(shards,max_seconds,epochs,grad_accum,replay_fraction=0.):
    audio_seconds=sum(float(shard["hours"] or 0)*3600 for shard in shards)
    source_budget=max(1.,float(max_seconds)*(1-min(max(float(replay_fraction),0.),.95)))
    batches=math.ceil(audio_seconds/source_budget)
    return max(2,math.ceil(batches*int(epochs)/max(1,int(grad_accum))))

def _stage_replay_fraction(config,stage_no,repeat_last_stage=False):
    if stage_no<=1 or repeat_last_stage: return 0.
    return float(config["data"].get("replay_fraction",.15))

def _all_registered_parquet_shards(db):
    from ..data.parquet import hash_file
    folder=ROOT/"data/parquet"
    rows=db.execute("SELECT * FROM shards ORDER BY name").fetchall()
    by_path={Path(row["path"]).resolve():row for row in rows}
    files=sorted(folder.glob("*.parquet"))
    if not files: raise SystemExit("No parquet shards found under data/parquet.")
    selected=[]
    for path in files:
        shard=by_path.get(path.resolve())
        if shard is None: raise SystemExit(f"{path.name} is not registered. Run data scan first.")
        if hash_file(path)!=shard["sha256"]: raise SystemExit(f"{path.name} changed since data scan. Run data scan before repeating all shards.")
        selected.append(shard)
    return selected

def _replay_refs():
    refs=[]
    for manifest_path in sorted((ROOT/"data/replay").glob("*.json")):
        try: manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception: continue
        if not Path(manifest.get("source","" )).exists(): continue
        shard=registry.connect().execute("SELECT sha256 FROM shards WHERE path=?",(manifest.get("source"),)).fetchone()
        if not shard or shard["sha256"]!=manifest.get("sha256"): continue
        for ref in manifest.get("rows",[]): refs.append({**ref,"source":manifest["source"],"audio_column":manifest["audio_column"],"text_column":manifest["text_column"]})
    return refs

def _replay_signature(refs=None):
    refs=_replay_refs() if refs is None else refs
    raw=json.dumps(refs,sort_keys=True,separators=(",",":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def _batches(shards,sp,device,max_seconds,start=None,replay_fraction=0.,seed=1729,shuffle_buffer=128,epoch=0):
    from ..data.batching import LengthBucketBatcher
    shard_signature=[{"name":s["name"],"sha256":s["sha256"]} for s in shards]
    dev_speakers=set(json.loads(registry.get_meta("dev_speakers","[]") or "[]")); dev_hashes=set(json.loads(registry.get_meta("dev_audio_hashes","[]") or "[]"))
    replay_refs=_replay_refs() if replay_fraction>0 else []; replay_index=int((start or {}).get("replay_index",0)); cached_key=None; cached_rows=None
    replay_sig=_replay_signature(replay_refs)
    def add_replay(batch,base_seconds):
        nonlocal replay_index,cached_key,cached_rows
        if not replay_refs or replay_fraction<=0: return batch
        goal=min(base_seconds*replay_fraction/max(1-replay_fraction,1e-6),max(0.,max_seconds-base_seconds)); added=0.; attempts=0
        while added<goal and attempts<len(replay_refs)*2:
            ref=replay_refs[replay_index%len(replay_refs)]; current_index=replay_index; replay_index+=1; attempts+=1
            key=(ref["source"],ref["row_group"],ref["audio_column"],ref["text_column"])
            if key!=cached_key:
                import pyarrow.parquet as pq
                pf=pq.ParquetFile(ref["source"]); cols=[c for c in (ref["audio_column"],ref["text_column"],"speaker_id","speaker") if c in pf.schema_arrow.names]
                cached_rows=pf.read_row_group(ref["row_group"],columns=cols).to_pylist(); cached_key=key
            row=cached_rows[ref["row_offset"]]; text=normalize_text(str(row.get(ref["text_column"]) or ""))
            if not text or _speaker_excluded(row,ref["audio_column"],dev_speakers,dev_hashes): continue
            payload=row.get(ref["audio_column"])
            if isinstance(payload,dict): payload=payload.get("bytes",payload.get("audio",payload.get("data")))
            try: wave,sr=decode_audio(payload)
            except Exception: continue
            duration=len(wave)/sr
            if duration<1 or duration>30: continue
            row["__text"],row["__wave"],row["__sr"]=text,wave,sr
            batch.append((row,{"replay":True,"replay_index":current_index,"row_group":ref["row_group"],"row_offset":ref["row_offset"]})); added+=duration
        for _,position in reversed(batch):
            if not position.get("replay"):
                position["replay_index"]=replay_index
                break
        return batch
    for shard_index,shard in enumerate(shards):
        if start and shard_index<start.get("shard",0): continue
        stats=json.loads(shard["stats_json"] or "{}"); acol,tcol=stats.get("audio_column"),stats.get("text_column")
        if not acol or not tcol: raise ValueError(f"Missing detected columns for {shard['name']}; rescan shard")
        begin_group=start.get("row_group",0) if start and shard_index==start.get("shard",0) else 0
        begin_offset=start.get("row_offset",0) if start and shard_index==start.get("shard",0) else 0
        pool_index=int(start.get("pool_index",0)) if start and shard_index==start.get("shard",0) else 0
        pool=[]; cursor=None
        def flush_pool():
            nonlocal pool,pool_index,cursor
            if not pool: return
            pool_seed=seed+pool_index*1009+shard_index*100003+epoch*1000003
            source_budget=max(1.,max_seconds*(1-replay_fraction))
            bucketer=LengthBucketBatcher(pool,max_seconds=source_budget,pool_size=min(32,len(pool)),seed=pool_seed)
            packed=list(bucketer)
            for batch_index,items in enumerate(packed):
                batch=[item["pair"] for item in items]; total=sum(item["duration"] for item in items)
                batch=add_replay(batch,total)
                if batch_index==len(packed)-1:
                    for _,position in batch:
                        if not position.get("replay"):
                            position["pool_end"]=True
                            position["pool_cursor"]={"shard":shard_index,"row_group":cursor["row_group"],"row_offset":cursor["row_offset"],"pool_index":pool_index+1,"seed":seed,"epoch":epoch,"shuffle_buffer":shuffle_buffer,"max_batch_seconds":max_seconds,"replay_fraction":replay_fraction,"replay_signature":replay_sig,"shards":shard_signature}
                yield batch
            pool=[]; pool_index+=1
        for row,pos in iter_rows(Path(shard["path"]),acol,tcol,begin_group,begin_offset):
            if start and shard_index==start.get("shard",0) and (pos["row_group"],pos["row_offset"]) <= (start.get("row_group",0),start.get("row_offset",0)): continue
            row["__text"]=normalize_text(str(row.get(tcol) or ""))
            if not row["__text"] or _speaker_excluded(row,acol,dev_speakers,dev_hashes): continue
            payload=row.get(acol)
            if isinstance(payload,dict): payload=payload.get("bytes",payload.get("audio",payload.get("data")))
            try: wave,sr=decode_audio(payload)
            except Exception: continue
            duration=len(wave)/sr
            if duration<1 or duration>30: continue
            row["__wave"],row["__sr"]=wave,sr
            pool.append({"duration":duration,"pair":(row,{**pos,"shard":shard_index})}); cursor=pos
            if len(pool)>=max(1,shuffle_buffer):
                yield from flush_pool()
        yield from flush_pool()

def train_asr(config,resume=False,max_minutes=None,repeat_last_stage=False,repeat_all_shards=False):
    from ..data.parquet import initialize_dev_set
    from ..asr.model import ConformerCTC
    from ..tokenizer import load
    initialize_dev_set(config["data"].get("dev_max_hours",2),config["data"].get("dev_fraction",.03))
    _validate_dev_manifest_sources()
    db=registry.connect()
    completed_count=db.execute("SELECT COUNT(*) FROM stages WHERE kind='asr'").fetchone()[0]
    if repeat_all_shards:
        shards=_all_registered_parquet_shards(db)
    elif repeat_last_stage:
        if completed_count<1: raise SystemExit("No completed ASR stage exists to repeat.")
        previous_name=f"asr-{completed_count:03d}"
        previous_row=db.execute("SELECT record_json FROM stages WHERE name=? AND kind='asr'",(previous_name,)).fetchone()
        previous_record=json.loads(previous_row["record_json"] or "{}") if previous_row else {}
        previous_shards=previous_record.get("shards",[])
        if not previous_shards: raise SystemExit(f"The previous stage {previous_name} has no recorded shard list to repeat.")
        selected=[]
        for item in previous_shards:
            shard=db.execute("SELECT * FROM shards WHERE name=?",(item["name"],)).fetchone()
            if not shard or shard["sha256"]!=item["sha256"] or not Path(shard["path"]).is_file():
                raise SystemExit(f"Cannot repeat {item['name']}: its registered file is missing or its hash changed.")
            selected.append(shard)
        shards=selected
    else:
        shards=db.execute("SELECT * FROM shards WHERE consumed_asr=0 ORDER BY name").fetchall()
        if not shards: raise SystemExit("No unconsumed registered shards. Run data scan first, or use --repeat-last-stage for another pass over the previous ASR stage.")
    if not registry.get_meta("tokenizer_path"): raise SystemExit("Run tokenizer train first.")
    dev_manifest=ROOT/"data/dev"/"manifest.json"
    if not dev_manifest.exists() or not json.loads(dev_manifest.read_text(encoding="utf-8")).get("rows"):
        raise SystemExit("Fixed dev set is empty. Add shards under data/parquet and run data scan first.")
    sp=load(); device="cuda" if torch.cuda.is_available() else "cpu"; seed=int(config["data"].get("seed",1729))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    cfg=config["asr"]; model_cfg={k:cfg[k] for k in ("d_model","layers","heads","ff_mult","dropout","gradient_checkpointing") if k in cfg}
    model=ConformerCTC(sp.get_piece_size(),**model_cfg).to(device)
    stage_no=1+completed_count
    if stage_no>1:
        previous=ROOT/"checkpoints"/f"asr-{stage_no-1:03d}"
        previous_checkpoint=_latest_checkpoint(previous.glob("step-*.pt"))
        if previous_checkpoint:
            previous_state=torch.load(previous_checkpoint,map_location=device,weights_only=False); model.load_state_dict(previous_state["model"])
        elif (previous/"best.pt").exists():
            previous_state=torch.load(previous/"best.pt",map_location=device,weights_only=False); model.load_state_dict(previous_state["model"])
    stage=f"asr-{stage_no:03d}"; run_dir=ROOT/"runs"/stage; ckpt_dir=ROOT/"checkpoints"/stage
    run_dir.mkdir(parents=True,exist_ok=True); ckpt_dir.mkdir(parents=True,exist_ok=True)
    write_json(run_dir/"config.json",config); write_json(run_dir/"shards.json",[{"name":s["name"],"sha256":s["sha256"],"rows":s["rows"],"hours":s["hours"],"oov_rate":json.loads(s["stats_json"] or "{}").get("oov_rate")} for s in shards])
    write_json(run_dir/"hardware.json",{"device":device,"cuda":torch.cuda.get_device_name() if device=="cuda" else None})
    write_json(run_dir/"model.json",{"parameters":model.param_count(),"config":model_cfg})
    base_lr=float(cfg.get("learning_rate",5e-4)); peak_lr=base_lr*(float(cfg.get("stage_lr_scale",.3)) if stage_no>1 else 1.)
    optimizer=torch.optim.AdamW(model.parameters(),lr=peak_lr)
    epochs=int(cfg.get("epochs",1)); budget=max(1,int(config["data"].get("max_batch_seconds",20)))
    accum=max(1,int(cfg.get("grad_accum",1)))
    replay_mix=_stage_replay_fraction(config,stage_no,repeat_last_stage or repeat_all_shards)
    estimated=_estimate_optimizer_steps(shards,budget,epochs,accum,replay_mix)
    warmup=max(1,int(cfg.get("warmup_steps",1000))); total=max(warmup+1,estimated)
    def lr_scale(n):
        if n<warmup: return max(.01,(n+1)/warmup)
        return .5*(1+math.cos(math.pi*min(1.,(n-warmup)/(total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale)
    scaler=torch.amp.GradScaler("cuda",enabled=False)
    state_path=_latest_checkpoint(ckpt_dir.glob("step-*.pt"))
    step=0; start_state=None; best_wer=float("inf"); last_loss=None; saved_gradients={}; saved_accumulated=0
    if resume:
        if state_path is None: raise SystemExit("No ASR checkpoint to resume.")
        state=torch.load(state_path,map_location=device,weights_only=False); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        if state.get("sampler_version")!=2: raise SystemExit("Checkpoint uses an older data sampler; finish that stage or start a new stage before resuming.")
        if state.get("train_config")!=config: raise SystemExit("Resume config differs from the checkpoint; restore its YAML values first.")
        step=state["step"]; start_state=state.get("data_state"); best_wer=state.get("best_wer",float("inf")); last_loss=state.get("loss"); saved_gradients=state.get("gradients",{}); saved_accumulated=int(state.get("accumulated",0))
        if start_state and start_state.get("seed",seed)!=seed: raise SystemExit("Resume seed differs from checkpoint.")
        if start_state and start_state.get("shuffle_buffer",int(config["data"].get("shuffle_buffer",128)))!=int(config["data"].get("shuffle_buffer",128)): raise SystemExit("Resume shuffle buffer differs from checkpoint.")
        if start_state and start_state.get("max_batch_seconds",budget)!=budget: raise SystemExit("Resume batch-seconds budget differs from checkpoint.")
        expected_replay=_stage_replay_fraction(config,stage_no,repeat_last_stage)
        if start_state and start_state.get("replay_fraction",expected_replay)!=expected_replay: raise SystemExit("Resume replay fraction differs from checkpoint.")
        replay_sig=_replay_signature(_replay_refs() if expected_replay>0 else [])
        if start_state and start_state.get("replay_signature",replay_sig)!=replay_sig: raise SystemExit("Replay manifests changed since checkpoint; restore the original replay set before resuming.")
        current_shards=[{"name":s["name"],"sha256":s["sha256"]} for s in shards]
        if start_state and start_state.get("shards",current_shards)!=current_shards: raise SystemExit("Registered shard set changed since checkpoint; restore the original stage shard set before resuming.")
        from .trainer import restore_rng_state
        restore_rng_state(state)
    best_path=ckpt_dir/"best.pt"
    if best_path.exists():
        prior=torch.load(best_path,map_location="cpu",weights_only=False); best_wer=float(prior.get("dev_wer",best_wer))
    blank=sp.get_piece_size(); loss_fn=torch.nn.CTCLoss(blank=blank,zero_infinity=True)
    metrics=run_dir/"metrics.jsonl"; eval_file=run_dir/"eval.jsonl"; began=time.monotonic(); accumulated=saved_accumulated; max_epoch=int(start_state.get("epoch",0)) if start_state else 0
    committed_pos=start_state; pending_pos=start_state
    committed_rng={"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    model.train(); optimizer.zero_grad(set_to_none=True); accum_started=None
    for name,param in model.named_parameters():
        if name in saved_gradients: param.grad=saved_gradients[name].to(device=device,dtype=param.dtype)
    def checkpoint(position,loss_value):
        state=_checkpoint_state(model,optimizer,scheduler,scaler,step,position,loss_value,model_cfg,best_wer,accumulated,config)
        _save_training_state(ckpt_dir,state,step,model.param_count())
    stop_requested=[False]; old_sigint=signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT,lambda *_: stop_requested.__setitem__(0,True))
    try:
        for epoch in range(max_epoch,epochs):
            cursor=start_state if epoch==max_epoch else None
            for raw_batch in _batches(shards,sp,device,budget,cursor,replay_mix,seed,int(config["data"].get("shuffle_buffer",128)),epoch):
                step_started=time.monotonic()
                prepared=_feature_batch(raw_batch,sp,device)
                source_positions=[p for _,p in raw_batch if not p.get("replay")]
                source_pos=source_positions[-1]
                pool_cursor=next((p.get("pool_cursor") for p in source_positions if p.get("pool_cursor")),None)
                pool_end=pool_cursor is not None
                if pool_end: pending_pos={**pool_cursor,"replay_index":source_pos.get("replay_index",0)}
                if prepared is None:
                    if pool_end:
                        committed_pos=pending_pos
                        should_stop=stop_requested[0] or max_minutes is not None and (time.monotonic()-began)/60>=max_minutes
                        if step%int(cfg.get("checkpoint_every",100))==0 or should_stop: checkpoint(committed_pos,last_loss)
                        if should_stop:
                            signal.signal(signal.SIGINT,old_sigint)
                            return f"Saved resumable checkpoint at step {step}; shard remains unconsumed. Resume with --resume."
                    continue
                features,targets,feature_lengths=prepared
                if accumulated==0: accum_started=time.monotonic()
                if model.training and float(cfg.get("specaugment",.0))>0:
                    for i in range(features.shape[0]):
                        if random.random()<float(cfg.get("specaugment",.0)):
                            width=min(10,features.shape[2]); f0=random.randrange(max(1,features.shape[2]-width+1)); features[i,:,f0:f0+width]=0
                            width=min(30,features.shape[1]); t0=random.randrange(max(1,features.shape[1]-width+1)); features[i,t0:t0+width,:]=0
                input_lengths=((feature_lengths+3)//4).to(device); target_lengths=torch.tensor([len(t) for t in targets],device=device,dtype=torch.long); flat=torch.cat(targets).to(device)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device=="cuda"):
                    output=model(features,feature_lengths.to(device)).transpose(0,1); loss=loss_fn(output,flat,input_lengths,target_lengths)/accum
                loss.backward(); accumulated+=1; last_loss=float(loss.detach())*accum
                if accumulated>=accum:
                    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); accumulated=0; step+=1
                    committed_rng={"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
                    step_seconds=max(time.monotonic()-(accum_started or step_started),1e-6); accum_started=None
                    rec={"step":step,"epoch":epoch,"loss":last_loss,"lr":optimizer.param_groups[0]["lr"],"grad_norm":float(grad_norm),"tokens_per_sec":int(target_lengths.sum())/step_seconds,"gpu_memory_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,"step_seconds":step_seconds}
                    with metrics.open("a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")
                    eval_every=int(cfg.get("eval_every_steps",500)); should_eval=step%eval_every==0
                    if should_eval:
                        evaluated=_evaluate_dev(model,sp,device)
                        if evaluated:
                            ev,samples=evaluated; ev.update({"step":step,"epoch":epoch})
                            with eval_file.open("a",encoding="utf-8") as f: f.write(json.dumps(ev)+"\n")
                            (run_dir/"samples.txt").write_text("\n".join(f"REF: {a}\nHYP: {b}" for a,b in samples),encoding="utf-8")
                            if ev["wer"]<best_wer:
                                best_wer=ev["wer"]; _save_asr_best(best_path,ROOT/"checkpoints/best-asr.pt",{"model":model.state_dict(),"model_config":model_cfg,"blank_id":blank,"tokenizer_hash":registry.get_meta("tokenizer_hash"),"dev_wer":best_wer},best_wer,model.param_count())
                if pool_end:
                    committed_pos=pending_pos
                    committed_rng={"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
                    should_stop=stop_requested[0] or max_minutes is not None and (time.monotonic()-began)/60>=max_minutes
                    if step%int(cfg.get("checkpoint_every",100))==0 or should_stop: checkpoint(committed_pos,last_loss)
                    if should_stop:
                        signal.signal(signal.SIGINT,old_sigint)
                        return f"Saved resumable checkpoint at step {step}; shard remains unconsumed. Resume with --resume."
        if accumulated:
            remainder=accumulated
            if remainder<accum:
                for parameter in model.parameters():
                    if parameter.grad is not None: parameter.grad.mul_(accum/remainder)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); accumulated=0; step+=1
            committed_rng={"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        final_eval=_evaluate_dev(model,sp,device)
        if final_eval:
            ev,samples=final_eval; ev.update({"step":step,"epoch":epochs-1})
            with eval_file.open("a",encoding="utf-8") as f: f.write(json.dumps(ev)+"\n")
            (run_dir/"samples.txt").write_text("\n".join(f"REF: {a}\nHYP: {b}" for a,b in samples),encoding="utf-8")
            if ev["wer"]<best_wer:
                best_wer=ev["wer"]; _save_asr_best(best_path,ROOT/"checkpoints/best-asr.pt",{"model":model.state_dict(),"model_config":model_cfg,"blank_id":blank,"tokenizer_hash":registry.get_meta("tokenizer_hash"),"dev_wer":best_wer},best_wer,model.param_count())
        checkpoint(committed_pos,last_loss)
    except KeyboardInterrupt:
        from .trainer import restore_rng_state
        restore_rng_state(committed_rng)
        state=_checkpoint_state(model,optimizer,scheduler,scaler,step,committed_pos,last_loss,model_cfg,best_wer,accumulated,config); state.update(committed_rng)
        _save_training_state(ckpt_dir,state,step,model.param_count())
        signal.signal(signal.SIGINT,old_sigint)
        return f"Interrupted safely at step {step}; shard remains unconsumed. Resume with --resume."
    signal.signal(signal.SIGINT,old_sigint)
    record={"steps":step,"parameters":model.param_count(),"loss":last_loss,"dev_wer":None if best_wer==float("inf") else best_wer,"git_hash":git_hash(),"wall_seconds":time.monotonic()-began,"checkpoints":[p.name for p in sorted(ckpt_dir.glob("*.pt"))],"config":config,"hardware":{"device":device,"cuda":torch.cuda.get_device_name() if device=="cuda" else None},"shards":[{"name":s["name"],"sha256":s["sha256"],"rows":s["rows"],"hours":s["hours"],"oov_rate":json.loads(s["stats_json"] or "{}").get("oov_rate")} for s in shards],"metrics":_jsonl(metrics),"evaluations":_jsonl(eval_file)}
    write_json(run_dir/"metrics.json",record)
    (run_dir/"report.md").write_text(f"# {stage}\n\nSteps: {step}\n\nParameters: {model.param_count()}\n\nBest dev WER: {record['dev_wer']}\n\nWall time: {record['wall_seconds']:.1f} seconds\n",encoding="utf-8")
    _write_plots(run_dir)
    db.execute("UPDATE shards SET consumed_asr=1 WHERE name IN ("+",".join("?" for _ in shards)+")",[s["name"] for s in shards]); db.execute("INSERT OR REPLACE INTO stages(name,kind,record_json) VALUES(?,?,?)",(stage,"asr",json.dumps(record))); db.commit()
    return f"Finished {stage}: {step} steps, {model.param_count()} parameters, dev WER {record['dev_wer']}."
