from __future__ import annotations
import hashlib
from pathlib import Path
from .. import registry
from ..util import ROOT, disk_guard

def texts():
    import pyarrow.parquet as pq
    with registry.connect() as db:
      for r in db.execute("SELECT path,stats_json FROM shards ORDER BY name"):
        pf=pq.ParquetFile(r["path"]); names=pf.schema_arrow.names
        import json
        stats=json.loads(r["stats_json"] or "{}")
        text=stats.get("text_column") or next((n for n in ("text","sentence","transcript","normalized_text") if n in names),None)
        if text:
          for rg in range(pf.num_row_groups):
            for x in pf.read_row_group(rg,columns=[text]).column(0).to_pylist():
              if x and str(x).strip(): yield str(x).strip()

def train(vocab_size=3000):
    if registry.get_meta("tokenizer_hash"): raise RuntimeError("Tokenizer is frozen; refusing retraining")
    import sentencepiece as spm
    root=ROOT/"checkpoints/tokenizer"; root.mkdir(parents=True,exist_ok=True)
    corpus=root/"corpus.txt"
    row_count=sum(r[0] or 0 for r in registry.connect().execute("SELECT rows FROM shards"))
    disk_guard(corpus,needed_bytes=row_count*256,reserve_bytes=1_000_000_000)
    with corpus.open("w",encoding="utf-8") as f:
      for t in texts(): f.write(t.replace("\n"," ")+"\n")
    if corpus.stat().st_size==0: raise ValueError("No registered transcripts")
    prefix=root/"hindi"
    spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=str(prefix),model_type="bpe",vocab_size=vocab_size,character_coverage=0.9995,byte_fallback=True,hard_vocab_limit=False)
    digest=hashlib.sha256((root/"hindi.model").read_bytes()).hexdigest()
    registry.set_meta("tokenizer_hash",digest); registry.set_meta("tokenizer_path",str(root/"hindi.model"))
    corpus.unlink()
    return digest

def load():
    import sentencepiece as spm
    model=registry.get_meta("tokenizer_path")
    if not model or not Path(model).exists(): raise FileNotFoundError("Run tokenizer train first")
    sp=spm.SentencePieceProcessor(model_file=model)
    return sp

def oov_rate(text):
    sp=load(); ids=sp.encode(text,out_type=int)
    return sum(i==sp.unk_id() for i in ids)/max(len(ids),1)
