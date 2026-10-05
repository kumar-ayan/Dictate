from __future__ import annotations
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from ..util import ROOT
from .pipeline import energy_trim, process_audio, inject

def _dictionary(path):
    result={}
    file=ROOT/path
    if file.exists():
        for line in file.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                wrong,right=line.split("\t",1); result[wrong]=right
    return result

def _file_hash(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def _load_asr(device,sp):
    from .. import registry
    from ..asr.model import ConformerCTC
    from ..train.asr import _decode
    path=ROOT/"checkpoints/best-asr.pt"
    if not path.exists(): raise SystemExit("No trained ASR checkpoint. Run tokenizer train and train asr first.")
    state=torch.load(path,map_location=device,weights_only=False)
    if state.get("tokenizer_hash")!=registry.get_meta("tokenizer_hash"):
        raise SystemExit("ASR checkpoint tokenizer hash does not match frozen tokenizer.")
    model=ConformerCTC(sp.get_piece_size(),**state["model_config"]).to(device); model.load_state_dict(state["model"]); model.eval()
    def infer(audio):
        wave=torch.as_tensor(np.asarray(audio,dtype=np.float32),device="cpu")
        if wave.numel()==0: return ""
        from ..data.parquet import log_mel
        feature=log_mel(wave,16000,device=device).transpose(0,1).unsqueeze(0)
        with torch.no_grad(): output=model(feature,torch.tensor([feature.shape[1]],device=device))[0]
        return sp.decode(_decode(output.unsqueeze(0),sp.get_piece_size())[0])
    # Initialize CUDA kernels and feature path before listening.
    infer(np.zeros(16000,dtype=np.float32))
    return model,infer,_file_hash(path)

def _load_cleanup(device,sp):
    from .. import registry
    from ..cleanup.model import CleanupTransformer
    from ..cleanup.evaluation import generate
    path=ROOT/"checkpoints/best-cleanup.pt"
    if not path.exists(): return None,None,None
    state=torch.load(path,map_location=device,weights_only=False)
    if state.get("tokenizer_hash")!=registry.get_meta("tokenizer_hash"):
        raise SystemExit("Cleanup checkpoint tokenizer hash does not match frozen tokenizer.")
    model_cfg=state["model_config"]
    model=CleanupTransformer(sp.get_piece_size(),**model_cfg).to(device); model.load_state_dict(state["model"]); model.eval()
    fn=lambda text: generate(model,sp,text,device,int(model_cfg.get("max_len",256)))
    fn("नमस्ते")
    return model,fn,_file_hash(path)

def run(config):
    import sounddevice as sd
    from pynput import keyboard
    from .. import registry
    from ..tokenizer import load
    mode=config.get("app",{}).get("cleanup","rules")
    device="cuda" if torch.cuda.is_available() else "cpu"; sp=load(); asr_model,asr,asr_version=_load_asr(device,sp)
    cleanup_model,cleanup,cleanup_version=_load_cleanup(device,sp) if mode=="model" else (None,None,None)
    if mode=="model" and cleanup is None: print("No cleanup checkpoint found; using rules."); mode="rules"
    dictionary=_dictionary(config.get("app",{}).get("dictionary","data/dictionary.txt"))
    key_name=config.get("app",{}).get("hotkey","right_ctrl").lower()
    if key_name in ("right_ctrl","ctrl_r"): key=keyboard.Key.ctrl_r
    elif key_name in ("left_ctrl","ctrl_l"): key=keyboard.Key.ctrl_l
    elif key_name.startswith("f") and key_name[1:].isdigit(): key=getattr(keyboard.Key,key_name)
    elif len(key_name)==1: key=keyboard.KeyCode.from_char(key_name)
    else: raise ValueError("app.hotkey must be right_ctrl, left_ctrl, F-key, or one character")
    chunks=[]; pressed=[None]
    print(f"Ready. Hold {key_name} to dictate; release to insert. Press Ctrl+C to quit.")
    def process_recording():
        if not chunks: return
        audio=energy_trim(np.concatenate(chunks,axis=0).reshape(-1))
        if audio.size==0: print("No speech detected."); return
        final,timings=process_audio(audio,asr,cleanup,mode,dictionary,{"asr":asr_version,"cleanup":cleanup_version or mode,"tokenizer":registry.get_meta("tokenizer_hash")})
        print(f"ASR {timings['asr']:.2f}s; cleanup {timings['cleanup']:.2f}s; total {sum(timings.values()):.2f}s")
        if final: inject(final)
    def on_press(event):
        if event==key and pressed[0] is None:
            pressed[0]=time.monotonic(); chunks.clear()
    def on_release(event):
        if event==key and pressed[0] is not None:
            duration=time.monotonic()-pressed[0]; pressed[0]=None
            if duration>=.3: process_recording()
            else: chunks.clear()
    def callback(indata,frames,timing,status):
        if pressed[0] is not None: chunks.append(indata.copy())
    try:
        with sd.InputStream(samplerate=16000,channels=1,dtype="float32",callback=callback):
            with keyboard.Listener(on_press=on_press,on_release=on_release) as listener: listener.join()
    except KeyboardInterrupt:
        print("Dictation stopped.")
