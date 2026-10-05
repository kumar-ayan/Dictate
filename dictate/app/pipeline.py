from __future__ import annotations
import json, time
from ..cleanup.rules import guarded_cleanup, clean_rules
from .. import registry

def energy_trim(audio,threshold=.012,margin=.2,sample_rate=16000):
    import numpy as np
    a=np.asarray(audio,dtype=np.float32); frame=max(1,int(.02*sample_rate)); n=len(a)//frame
    if not n:return a
    rms=np.array([np.sqrt(np.mean(a[i*frame:(i+1)*frame]**2)) for i in range(n)])
    active=np.flatnonzero(rms>=threshold)
    if not len(active):return a[:0]
    lo=active[0]*frame; hi=min(len(a),(active[-1]+1)*frame)
    # Refine frame boundaries at sample level so partial edge frames do not add
    # up to 40 ms of silence around otherwise clean clips.
    within=np.flatnonzero(np.abs(a[lo:hi])>=threshold)
    if not len(within): return a[:0]
    pad=int(margin*sample_rate)
    return a[max(0,lo+int(within[0])-pad):min(len(a),lo+int(within[-1])+1+pad)]

def process_audio(audio,asr,cleanup=None,mode="rules",dictionary=None,model_versions=None):
    started=time.perf_counter(); raw=asr(audio); t_asr=time.perf_counter()-started
    personalized=raw
    for word,preferred in (dictionary or {}).items(): personalized=personalized.replace(word,preferred)
    started=time.perf_counter(); fn=cleanup if mode=="model" and cleanup else clean_rules
    cleaned=guarded_cleanup(personalized,fn,fallback=raw); t_cleanup=time.perf_counter()-started; final=cleaned
    with registry.connect() as db: db.execute("INSERT INTO dictations(raw,cleaned,final,timings_json,model_versions) VALUES(?,?,?,?,?)",(raw,cleaned,final,json.dumps({"asr":t_asr,"cleanup":t_cleanup}),json.dumps(model_versions or {"asr":"local","cleanup":mode})))
    return final,{"asr":t_asr,"cleanup":t_cleanup}

def inject(text):
    import pyperclip
    from pynput.keyboard import Controller,Key
    old=pyperclip.paste()
    try:
        pyperclip.copy(text); keyboard=Controller(); keyboard.press(Key.ctrl); keyboard.press("v"); keyboard.release("v"); keyboard.release(Key.ctrl)
        time.sleep(.15)
    finally: pyperclip.copy(old)
