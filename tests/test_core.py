import io
import sys
import types
import numpy as np
import pytest

from dictate.data.parquet import normalize_text, decode_audio, scan
from dictate.data.batching import LengthBucketBatcher
from dictate.cleanup.corrupt import pairs
from dictate.cleanup.rules import guarded_cleanup
from dictate.app.pipeline import energy_trim
from dictate.testing import make_fake_parquet

def test_normalize_text():
    assert normalize_text("\u200b नमस्ते   दुनिया ") == "नमस्ते दुनिया"

def test_batch_seconds_and_state():
    rows=[{"duration":x} for x in (1,4,2,3)]
    batches=list(LengthBucketBatcher(rows,max_seconds=5,seed=4))
    assert all(sum(x["duration"] for x in b)<=5 for b in batches)
    assert LengthBucketBatcher(rows,seed=4).state_dict()["seed"]==4

def test_asr_bounded_bucket_resume_recreates_next_pool(tmp_path,monkeypatch):
    import json
    import dictate.train.asr as asr
    from dictate.testing import make_fake_parquet
    path=tmp_path/"bucket.parquet"; make_fake_parquet(path,rows=8)
    monkeypatch.setattr(asr,"_speaker_excluded",lambda *args:False)
    shard={"name":path.name,"path":str(path),"sha256":"fixture","stats_json":json.dumps({"audio_column":"audio","text_column":"text"})}
    complete=list(asr._batches([shard],None,"cpu",3,seed=19,shuffle_buffer=4,epoch=0))
    boundary=next(i for i,batch in enumerate(complete) if any(pos.get("pool_end") for _,pos in batch))
    cursor=next(pos["pool_cursor"] for _,pos in complete[boundary] if pos.get("pool_cursor"))
    resumed=list(asr._batches([shard],None,"cpu",3,start=cursor,seed=19,shuffle_buffer=4,epoch=0))
    def positions(batches): return [pos["row_offset"] for batch in batches for _,pos in batch if not pos.get("replay")]
    assert positions(resumed)==positions(complete[boundary+1:])
    assert sorted(positions(complete))==list(range(1,9))

def test_audio_decode_wav_bytes():
    import soundfile as sf
    data=io.BytesIO(); sf.write(data,np.zeros(8000,dtype=np.float32),8000,format="WAV")
    wave,sr=decode_audio(data.getvalue())
    assert sr==16000 and len(wave)==16000

def test_fake_parquet_scan(tmp_path):
    path=tmp_path/"tiny.parquet"; make_fake_parquet(path,rows=4)
    results=scan(path.parent)
    assert results and results[0]["rows"]==4
    assert results[0]["speakers"]==4

def test_corruption_keeps_original_targets():
    source="आज मौसम अच्छा है।"
    assert list(pairs([source],seed=1,clean_fraction=1))==[(source,source)]

def test_cleanup_guard_fallback():
    raw="मुझे पानी चाहिए"
    assert guarded_cleanup(raw,lambda _: "x") == raw

def test_cleanup_guard_can_restore_original_asr_text():
    assert guarded_cleanup("preferred spelling",lambda _: "",fallback="raw ASR") == "raw ASR"

def test_energy_trim():
    audio=np.r_[np.zeros(4000),np.ones(8000,dtype=np.float32)*.1,np.zeros(4000)]
    trimmed=energy_trim(audio,margin=0)
    assert 7900<=len(trimmed)<=8100

def test_clipboard_restore(monkeypatch):
    clipboard={"text":"previous"}; pasted=[]
    clip=types.ModuleType("pyperclip"); clip.paste=lambda:clipboard["text"]; clip.copy=lambda value:clipboard.update(text=value)
    keyboard=types.ModuleType("pynput.keyboard")
    class Controller:
        def press(self,key): pass
        def release(self,key):
            if key=="v": pasted.append(clipboard["text"])
    keyboard.Controller=Controller; keyboard.Key=types.SimpleNamespace(ctrl="ctrl")
    pynput=types.ModuleType("pynput"); pynput.keyboard=keyboard
    monkeypatch.setitem(sys.modules,"pyperclip",clip); monkeypatch.setitem(sys.modules,"pynput",pynput); monkeypatch.setitem(sys.modules,"pynput.keyboard",keyboard)
    from dictate.app.pipeline import inject
    inject("inserted")
    assert pasted==["inserted"] and clipboard["text"]=="previous"

def test_tiny_asr_forward_and_ctc_blank():
    import torch
    from dictate.asr.model import ConformerCTC
    model=ConformerCTC(12,d_model=16,layers=1,heads=2,ff_mult=2,dropout=0)
    output=model(torch.randn(2,32,80),torch.tensor([32,20]))
    assert output.shape==(2,8,13)
    assert model.blank_id==12

def test_greedy_ctc_decode_keeps_batch_dimension():
    import torch
    from dictate.train.asr import _decode
    # IDs 1,1,blank,2 collapse to [1,2] for one batch item.
    logits=torch.full((1,4,4),-10.0)
    logits[0,0,1]=10; logits[0,1,1]=10; logits[0,2,3]=10; logits[0,3,2]=10
    assert _decode(logits,blank=3)==[[1,2]]

def test_asr_schedule_estimate_accounts_for_gradient_accumulation():
    from dictate.train.asr import _estimate_optimizer_steps, _stage_replay_fraction
    shards=[{"hours":320/3600}]
    assert _estimate_optimizer_steps(shards,20,epochs=2,grad_accum=4)==8
    cfg={"data":{"replay_fraction":.15}}
    assert _stage_replay_fraction(cfg,2,repeat_last_stage=True)==0
    assert _stage_replay_fraction(cfg,2)==.15

def test_asr_cli_accepts_repeat_last_stage():
    from dictate.cli import parser
    args=parser().parse_args(["train","asr","--repeat-last-stage"])
    assert args.repeat_last_stage and not args.resume

def test_asr_cli_accepts_repeat_all_shards():
    from dictate.cli import parser
    args=parser().parse_args(["train","asr","--repeat-all-shards"])
    assert args.repeat_all_shards and not args.repeat_last_stage

def test_cli_accepts_web_command_and_port():
    from dictate.cli import parser
    args=parser().parse_args(["web","--port","9000"])
    assert args.command=="web" and args.port==9000

def test_web_transcribes_decoded_audio_in_memory(monkeypatch):
    import numpy as np
    from dictate.app import web
    from dictate.data import audio
    from dictate.app import pipeline
    wave=np.full(16000,.1,dtype=np.float32)
    monkeypatch.setattr(audio,"decode_audio",lambda payload:(wave,16000))
    monkeypatch.setattr(pipeline,"energy_trim",lambda value,**kwargs:value)
    seen=[]
    result=web._transcribe_payload(b"wav bytes",lambda value:seen.append(value.copy()) or "नमस्ते")
    assert result["text"]=="नमस्ते"
    assert result["seconds"]==1
    assert len(seen)==1 and np.array_equal(seen[0],wave)

def test_asr_inference_chunks_long_audio_to_bound_attention_memory():
    import torch
    from dictate.app.run import _chunked_model_output
    class FakeModel:
        def __init__(self): self.feature_lengths=[]
        def __call__(self,features,lengths):
            self.feature_lengths.append(features.shape[1])
            return torch.zeros(1,(features.shape[1]+3)//4,5)
    model=FakeModel()
    wave=torch.zeros(21*16000)
    def fake_log_mel(part,sample_rate,device): return torch.zeros(80,len(part)//160)
    output=_chunked_model_output(wave,model,"cpu",fake_log_mel)
    assert model.feature_lengths==[1000,1000,300]
    assert max(model.feature_lengths)<=10*100
    assert output.shape==(523,5)

def test_repeat_all_selects_registered_parquet_files(tmp_path,monkeypatch):
    import hashlib,sqlite3
    import dictate.train.asr as asr
    folder=tmp_path/"data"/"parquet"; folder.mkdir(parents=True)
    paths=[folder/f"train-{i}.parquet" for i in range(4)]
    for path in paths: path.write_bytes(path.name.encode())
    db=sqlite3.connect(":memory:"); db.row_factory=sqlite3.Row
    db.execute("CREATE TABLE shards(path TEXT, sha256 TEXT, name TEXT)")
    for path in paths:
        db.execute("INSERT INTO shards VALUES(?,?,?)",(str(path.resolve()),hashlib.sha256(path.read_bytes()).hexdigest(),path.name))
    monkeypatch.setattr(asr,"ROOT",tmp_path)
    selected=asr._all_registered_parquet_shards(db)
    assert [row["name"] for row in selected]==[path.name for path in paths]

def test_latest_checkpoint_uses_newest_save_time(tmp_path):
    import os
    from dictate.train.asr import _latest_checkpoint
    old=tmp_path/"step-00007423.pt"; new=tmp_path/"step-00000000.pt"
    old.touch(); new.touch()
    os.utime(old,ns=(1_000_000_000,1_000_000_000))
    os.utime(new,ns=(2_000_000_000,2_000_000_000))
    assert _latest_checkpoint([old,new])==new

def test_missing_fixed_dev_source_fails_clearly(tmp_path,monkeypatch):
    import json
    import dictate.train.asr as asr
    dev=tmp_path/"data"/"dev"; dev.mkdir(parents=True)
    missing=tmp_path/"data"/"parquet"/"missing.parquet"
    (dev/"manifest.json").write_text(json.dumps({"rows":[{"path":str(missing)}]}),encoding="utf-8")
    monkeypatch.setattr(asr,"ROOT",tmp_path)
    with pytest.raises(SystemExit,match="missing.parquet"):
        asr._validate_dev_manifest_sources()

def test_asr_new_stage_preserves_global_best_checkpoint(tmp_path,monkeypatch):
    import torch
    import dictate.train.asr as asr
    monkeypatch.setattr(asr,"disk_guard",lambda *args,**kwargs:None)
    stage=tmp_path/"stage-best.pt"; global_best=tmp_path/"best-asr.pt"
    torch.save({"dev_wer":.5,"tag":"older"},global_best)
    asr._save_asr_best(stage,global_best,{"dev_wer":.7,"tag":"new-stage"},.7,param_count=1)
    assert torch.load(global_best,weights_only=False)["tag"]=="older"
    asr._save_asr_best(stage,global_best,{"dev_wer":.4,"tag":"better"},.4,param_count=1)
    assert torch.load(global_best,weights_only=False)["tag"]=="better"

def test_tiny_cleanup_encoder_decoder_forward():
    import torch
    from dictate.cleanup.model import CleanupTransformer
    model=CleanupTransformer(20,d_model=32,encoder_layers=1,decoder_layers=1,heads=4,max_len=16)
    logits=model(torch.tensor([[1,2,20]]),torch.tensor([[1,3,4]]))
    assert logits.shape==(1,3,20)

def test_cleanup_real_eval_file_has_30_pairs():
    from dictate.cleanup.evaluation import real_eval_pairs
    assert len(real_eval_pairs())==30

def test_dev_speaker_exclusion(monkeypatch):
    from dictate import registry
    from dictate.train.asr import _speaker_excluded
    monkeypatch.setattr(registry,"get_meta",lambda key,default=None:'["held-out"]' if key=="dev_speakers" else ("[]" if key=="dev_audio_hashes" else default))
    assert _speaker_excluded({"speaker_id":"held-out"})
    assert not _speaker_excluded({"speaker_id":"train-speaker"})

def test_tokenizer_refuses_retrain_after_freeze(monkeypatch):
    from dictate import registry
    from dictate.tokenizer.spm import train
    monkeypatch.setattr(registry,"get_meta",lambda key,default=None:"frozen" if key=="tokenizer_hash" else default)
    with pytest.raises(RuntimeError,match="frozen"):
        train()
