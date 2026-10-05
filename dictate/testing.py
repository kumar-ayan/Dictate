from __future__ import annotations
import io
from pathlib import Path
from .util import ROOT

def make_fake_parquet(path=None,rows=24):
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq, soundfile as sf
    path=Path(path) if path else ROOT/"data/parquet/fake_hindi.parquet"; path.parent.mkdir(parents=True,exist_ok=True)
    audio=[]; text=[]; speaker=[]
    for i in range(rows):
        t=np.arange(16000,dtype=np.float32)/16000; wave=.08*np.sin(2*np.pi*(180+i)*t)
        buf=io.BytesIO(); sf.write(buf,wave,16000,format="WAV"); audio.append({"bytes":buf.getvalue()}); text.append(["नमस्ते दुनिया।","आज मौसम अच्छा है।","मुझे पानी चाहिए।"][i%3]); speaker.append(f"speaker-{i%8}")
    table=pa.table({"audio":pa.array(audio,type=pa.struct([("bytes",pa.binary())])),"text":text,"speaker_id":speaker,"speech_style":["read"]*rows})
    pq.write_table(table,path); return str(path)
