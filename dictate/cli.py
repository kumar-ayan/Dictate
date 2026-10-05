from __future__ import annotations
import argparse, json
from pathlib import Path
from .util import ROOT, ensure_layout, load_config, setup_logging

def parser():
    p=argparse.ArgumentParser(prog="dictate"); p.add_argument("--config",default=None); sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("data").add_subparsers(dest="action",required=True); d.add_parser("scan")
    t=sub.add_parser("tokenizer").add_subparsers(dest="action",required=True); t.add_parser("train")
    tr=sub.add_parser("train").add_subparsers(dest="stage",required=True)
    for name in ("asr","cleanup"):
        q=tr.add_parser(name); q.add_argument("--resume",action="store_true"); q.add_argument("--max-minutes",type=float)
        if name=="asr": q.add_argument("--repeat-last-stage",action="store_true",help="Train another stage on the latest completed ASR stage's shards")
    e=sub.add_parser("eval").add_subparsers(dest="stage",required=True); e.add_parser("asr"); e.add_parser("cleanup")
    r=sub.add_parser("replay").add_subparsers(dest="action",required=True); rb=r.add_parser("build"); rb.add_argument("shard",nargs="?"); rb.add_argument("--max-hours",type=float,default=3)
    sub.add_parser("report"); sub.add_parser("run"); sub.add_parser("test"); sub.add_parser("fake-data")
    return p

def main(argv=None):
    args=parser().parse_args(argv); ensure_layout(); setup_logging(); cfg=load_config(args.config)
    if args.command=="data":
        from .data import scan
        print(json.dumps(scan(audio_column=cfg["data"].get("audio_column"),text_column=cfg["data"].get("text_column")),ensure_ascii=False,indent=2))
    elif args.command=="tokenizer":
        from .tokenizer import train
        print(train(cfg["tokenizer"]["vocab_size"]))
    elif args.command=="fake-data":
        from .testing import make_fake_parquet
        print(make_fake_parquet())
    elif args.command=="train": train_stage(args,cfg)
    elif args.command=="eval": evaluate_stage(args.stage)
    elif args.command=="replay": build_replay(args)
    elif args.command=="report": report()
    elif args.command=="run":
        from .app.run import run
        run(cfg)
    elif args.command=="test":
        import pytest
        raise SystemExit(pytest.main([str(ROOT/"tests"),"-q"]))

def train_stage(args,cfg):
    from . import registry
    rows=registry.connect().execute(f"SELECT * FROM shards WHERE consumed_{args.stage}=0").fetchall()
    if not rows and args.stage=="asr" and not args.repeat_last_stage: raise SystemExit("No unconsumed registered shards. Run data scan first, or use --repeat-last-stage for another pass over the previous ASR stage.")
    if args.stage=="asr":
        from .train.asr import train_asr
        print(train_asr(cfg,resume=args.resume,max_minutes=args.max_minutes,repeat_last_stage=args.repeat_last_stage))
    else:
        from .cleanup.training import train_cleanup
        print(train_cleanup(cfg,resume=args.resume,max_minutes=args.max_minutes))

def evaluate_stage(stage):
    import torch
    from .tokenizer import load
    sp=load(); device="cuda" if torch.cuda.is_available() else "cpu"
    if stage=="asr":
        import shutil
        from . import registry
        from .asr.model import ConformerCTC
        from .train.asr import _evaluate_dev
        path=ROOT/"checkpoints/best-asr.pt"
        recovered=False
        if not path.exists():
            # Completed training can have a final checkpoint even when dev
            # evaluation failed. Evaluate that checkpoint instead of requiring
            # an expensive retrain.
            row=registry.connect().execute("SELECT name FROM stages WHERE kind='asr' ORDER BY created_at DESC LIMIT 1").fetchone()
            if row:
                stage_dir=ROOT/"checkpoints"/row["name"]
                candidates=sorted(stage_dir.glob("step-*.pt"),key=lambda p:p.name,reverse=True)
                if candidates:
                    path=candidates[0]; recovered=True
            if not recovered: raise SystemExit("No completed ASR checkpoint found. Train ASR first.")
        state=torch.load(path,map_location=device,weights_only=False); model=ConformerCTC(sp.get_piece_size(),**state["model_config"]).to(device); model.load_state_dict(state["model"])
        result=_evaluate_dev(model,sp,device)
        if not result: raise SystemExit("Fixed dev set is empty. Run data scan first.")
        scores,samples=result
        if recovered:
            best={"model":model.state_dict(),"model_config":state["model_config"],"blank_id":sp.get_piece_size(),"tokenizer_hash":registry.get_meta("tokenizer_hash"),"dev_wer":scores["wer"]}
            from .util import disk_guard
            stage_best=ROOT/"checkpoints"/row["name"]/"best.pt"
            disk_guard(stage_best,needed_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),reserve_bytes=500_000_000)
            torch.save(best,stage_best)
            disk_guard(ROOT/"checkpoints/best-asr.pt",needed_bytes=stage_best.stat().st_size,reserve_bytes=500_000_000)
            shutil.copy2(stage_best,ROOT/"checkpoints/best-asr.pt")
            eval_record={**scores,"step":state.get("step"),"source":"post-training recovery"}
            run_dir=ROOT/"runs"/row["name"]
            with (run_dir/"eval.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(eval_record)+"\n")
            (run_dir/"samples.txt").write_text("\n".join(f"REF: {a}\nHYP: {b}" for a,b in samples),encoding="utf-8")
            record_row=registry.connect().execute("SELECT record_json FROM stages WHERE name=?",(row["name"],)).fetchone()
            record=json.loads(record_row["record_json"] or "{}") if record_row else {}
            record["dev_wer"]=scores["wer"]; record["dev_cer"]=scores["cer"]; record["evaluations"]=[*record.get("evaluations",[]),eval_record]
            with registry.connect() as db: db.execute("UPDATE stages SET record_json=? WHERE name=?",(json.dumps(record),row["name"]))
            report=run_dir/"report.md"
            if report.exists():
                report.write_text(report.read_text(encoding="utf-8").replace("Best dev WER: None",f"Best dev WER: {scores['wer']:.4f}\n\nDev CER: {scores['cer']:.4f}"),encoding="utf-8")
        print(json.dumps(scores,ensure_ascii=False,indent=2)); print("\n".join(f"REF: {a}\nHYP: {b}" for a,b in samples))
    else:
        from .cleanup.model import CleanupTransformer
        from .cleanup.evaluation import evaluate_pairs,real_eval_pairs
        path=ROOT/"checkpoints/best-cleanup.pt"
        if not path.exists(): raise SystemExit("No best cleanup checkpoint. Train cleanup first.")
        state=torch.load(path,map_location=device,weights_only=False); model=CleanupTransformer(sp.get_piece_size(),**state["model_config"]).to(device); model.load_state_dict(state["model"])
        from .cleanup.training import _validation_pairs
        heldout,samples=evaluate_pairs(model,sp,_validation_pairs(),device,int(state["model_config"].get("max_len",256)))
        real,_=evaluate_pairs(model,sp,real_eval_pairs(),device,int(state["model_config"].get("max_len",256)))
        print(json.dumps({"heldout":heldout,"real_pairs":real},ensure_ascii=False,indent=2)); print("\n".join(str(x) for x in samples))

def build_replay(args):
    import json, math, random
    import pyarrow.parquet as pq
    from . import registry
    from .data.parquet import decode_audio, normalize_text
    db=registry.connect(); row=db.execute("SELECT * FROM shards WHERE name=?",(args.shard,)).fetchone() if args.shard else db.execute("SELECT * FROM shards ORDER BY name DESC LIMIT 1").fetchone()
    if not row: raise SystemExit("No registered shard")
    # AGENTS.md forbids copying audio bytes from source parquet. Keep a compact reference manifest.
    out=ROOT/"data/replay"/(Path(row["name"]).stem+".json")
    stats=json.loads(row["stats_json"] or "{}"); acol=stats.get("audio_column"); tcol=stats.get("text_column")
    if not acol or not tcol: raise SystemExit("Shard lacks detected audio/text columns; rescan it.")
    pf=pq.ParquetFile(row["path"]); total=pf.metadata.num_rows; total_hours=float(row["hours"] or 0)
    estimate=min(total,max(1,math.ceil(total*min(1,args.max_hours/max(total_hours,1e-6))*1.25)))
    rng=random.Random(int(row["sha256"][:16],16)); chosen=set(rng.sample(range(total),estimate)) if estimate<total else set(range(total))
    refs=[]; used=0.; global_row=0; cap=args.max_hours*3600
    for rg in range(pf.num_row_groups):
        count=pf.metadata.row_group(rg).num_rows
        offsets=[i for i in range(count) if global_row+i in chosen]
        if offsets:
            cols=[c for c in (acol,tcol,"speaker_id","speaker") if c in pf.schema_arrow.names]
            rows=pf.read_row_group(rg,columns=cols).to_pylist()
            for off in offsets:
                current=rows[off]; text=normalize_text(str(current.get(tcol) or ""))
                if not text: continue
                try:
                    payload=current.get(acol)
                    if isinstance(payload,dict): payload=payload.get("bytes",payload.get("audio",payload.get("data")))
                    wave,sr=decode_audio(payload); duration=len(wave)/sr
                except Exception: continue
                if duration<1 or duration>30 or used+duration>cap: continue
                refs.append({"row_group":rg,"row_offset":off,"duration":duration}); used+=duration
                if used>=cap: break
        global_row+=count
        if used>=cap: break
    out.write_text(json.dumps({"source":row["path"],"sha256":row["sha256"],"audio_column":acol,"text_column":tcol,"rows":refs,"hours":used/3600,"hours_cap":args.max_hours},indent=2),encoding="utf-8")
    print(out)

def report():
    from . import registry
    print(f"{'Stage':<18} {'Type':<9} {'Steps':>8} {'Loss':>10} {'Dev metric':>16} {'Params':>12}")
    for r in registry.connect().execute("SELECT name,kind,record_json FROM stages ORDER BY created_at"):
        value=json.loads(r["record_json"] or "{}")
        steps=value.get("steps",0); loss=value.get("loss"); params=value.get("parameters",0)
        metric=value.get("dev_wer") if r["kind"]=="asr" else value.get("evaluation",{}).get("cer")
        print(f"{r['name']:<18} {r['kind']:<9} {steps:>8} {str(loss):>10} {str(metric):>16} {params:>12}")

if __name__=="__main__": main()
