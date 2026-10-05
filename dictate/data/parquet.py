from __future__ import annotations
import hashlib, json, re, unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator
from .. import registry
from ..util import ROOT

ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).translate(ZW)).strip()

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def hash_audio_payload(value) -> str:
    value=_audio_payload(value)
    if isinstance(value,memoryview): value=value.tobytes()
    if not isinstance(value,(bytes,bytearray)): raise TypeError("Audio payload must be embedded bytes")
    return hashlib.sha256(value).hexdigest()

def _columns(schema, audio_override=None, text_override=None):
    names = schema.names
    audio = audio_override
    text = text_override
    for field in schema:
        typ = str(field.type).lower()
        if audio is None and ("binary" in typ or "struct" in typ and any(k in typ for k in ("bytes", "path"))): audio = field.name
        if text is None and field.name.lower() in ("text", "sentence", "transcript", "normalized_text"): text = field.name
    if audio not in names or text not in names:
        raise ValueError(f"Cannot identify audio/text columns. Schema: {schema}")
    return audio, text

def _audio_payload(value):
    if isinstance(value, dict):
        for k in ("bytes", "audio", "data"):
            if value.get(k) is not None: return value[k]
    return value

def scan(directory: Path | None = None, audio_column=None, text_column=None) -> list[dict]:
    import pyarrow.parquet as pq
    from .audio import decode_audio
    directory = directory or ROOT / "data/parquet"
    directory.mkdir(parents=True, exist_ok=True)
    results=[]
    tokenizer=None
    model_path=registry.get_meta("tokenizer_path")
    if model_path:
        try:
            import sentencepiece as spm
            tokenizer=spm.SentencePieceProcessor(model_file=model_path)
        except Exception: tokenizer=None
    with registry.connect() as db:
      for path in sorted(directory.glob("*.parquet")):
        digest=hash_file(path)
        old=db.execute("SELECT sha256,path FROM shards WHERE name=?",(path.name,)).fetchone()
        if old and old[0]==digest and Path(old[1]).resolve()==path.resolve(): continue
        pf=pq.ParquetFile(path); schema=pf.schema_arrow
        print(f"{path.name}:\n{schema}")
        acol,tcol=_columns(schema,audio_column,text_column)
        decode_errors=0; styles=Counter(); hist=Counter(); rows=0; seconds=0.; speakers=set(); chars=Counter(); latin=punct=nonempty=0; oov_tokens=token_count=0
        for rg in range(pf.num_row_groups):
          tab=pf.read_row_group(rg,columns=list(dict.fromkeys([acol,tcol]+[n for n in ("speaker_id","speaker","style","speech_style","scenario") if n in schema.names])))
          for row in tab.to_pylist():
            rows+=1; text=normalize_text(str(row.get(tcol) or ""))
            if text:
              nonempty+=1; chars.update(text); latin+=bool(re.search(r"[A-Za-z]",text)); punct+=bool(re.search(r"[.,!?;:।॥]",text))
              if tokenizer is not None:
                token_ids=tokenizer.encode(text,out_type=int); token_count+=len(token_ids); oov_tokens+=sum(x==tokenizer.unk_id() for x in token_ids)
            speaker=row.get("speaker_id",row.get("speaker")); speakers.add(str(speaker) if speaker is not None else "")
            style=row.get("speech_style") or row.get("style") or row.get("scenario")
            if style: styles[str(style)] += 1
            try:
              wave,sr=decode_audio(_audio_payload(row.get(acol)))
              dur=len(wave)/sr; seconds+=dur; hist[f"{int(dur)}-{int(dur)+1}s"]+=1
            except Exception: decode_errors+=1
        doc={"rows":rows,"hours":seconds/3600,"duration_histogram":dict(hist),"speakers":len(speakers-{""}),"speech_style":dict(styles),"audio_decode_errors":decode_errors,"latin_transcript_share":latin/max(nonempty,1),"punctuation_transcript_share":punct/max(nonempty,1),"top_characters":chars.most_common(40),"audio_column":acol,"text_column":tcol,"oov_rate":oov_tokens/max(token_count,1) if tokenizer is not None else None}
        db.execute("INSERT OR REPLACE INTO shards(name,path,sha256,rows,hours,schema_json,stats_json,consumed_asr,consumed_cleanup) VALUES (?,?,?,?,?,?,?,?,?)",(path.name,str(path.resolve()),digest,rows,seconds/3600,str(schema),json.dumps(doc,ensure_ascii=False),0,0))
        db.commit(); results.append(doc)
    if directory.resolve()==(ROOT/"data/parquet").resolve(): initialize_dev_set()
    return results

def initialize_dev_set(max_hours=2.0,fraction=.03):
    """Store fixed dev rows as references; never copy audio out of source shards."""
    from .. import registry
    from ..util import write_json
    if registry.get_meta("dev_initialized")=="1": return
    import pyarrow.parquet as pq
    from .audio import decode_audio
    db=registry.connect(); all_rows=db.execute("SELECT * FROM shards ORDER BY name").fetchall()
    parquet_root=(ROOT/"data/parquet").resolve()
    rows=[]
    for item in all_rows:
        try:
            if Path(item["path"]).resolve().is_relative_to(parquet_root): rows.append(item)
        except (OSError,ValueError): continue
    if not rows: return
    selected=[]; speakers=set(); hashes=[]; anonymous=[]; fallback_speaker=None; fallback=[]; used=0.; cap=max_hours*3600
    # First registered shard anchors fixed dev selection. Stable hash sampling
    # approximates the configured speaker share without scanning every shard.
    for shard in rows[:1]:
        stats=json.loads(shard["stats_json"] or "{}"); acol=stats.get("audio_column"); tcol=stats.get("text_column")
        if not acol or not tcol: continue
        pf=pq.ParquetFile(shard["path"])
        seen=0
        for rg in range(pf.num_row_groups):
            cols=[x for x in (acol,tcol,"speaker_id","speaker") if x in pf.schema_arrow.names]
            tab=pf.read_row_group(rg,columns=list(dict.fromkeys(cols)))
            for off,row in enumerate(tab.to_pylist()):
                seen+=1
                if seen>50000: break
                try:
                    payload=_audio_payload(row.get(acol)); wave,sr=decode_audio(payload); duration=len(wave)/sr
                    if not normalize_text(str(row.get(tcol) or "")) or duration<1 or duration>30: continue
                    ref={"path":shard["path"],"row_group":rg,"row_offset":off,"duration":duration,"text":normalize_text(str(row[tcol])),"audio_column":acol}
                    speaker=row.get("speaker_id",row.get("speaker"))
                    if speaker is not None and str(speaker).strip():
                        speaker=str(speaker)
                        if fallback_speaker is None: fallback_speaker=speaker
                        if len(fallback)<10000 and speaker==fallback_speaker: fallback.append(ref)
                        sample=int(hashlib.sha256(speaker.encode("utf-8")).hexdigest()[:8],16)/0xffffffff
                        if sample<fraction and used+duration<=cap:
                            speakers.add(speaker); selected.append(ref); used+=duration
                    else:
                        digest=hash_audio_payload(payload)
                        if int(digest[:8],16)/0xffffffff<fraction and used+duration<=cap:
                            anonymous.append((digest,ref)); used+=duration
                except Exception: continue
                if used>=cap: break
            if seen>50000 or used>=cap: break
    if not selected and not anonymous and fallback:
        # Tiny fixtures can hash-sample zero speakers. Hold one speaker out so
        # fixed dev behavior remains testable and deterministic.
        selected=fallback[:max(1,min(len(fallback),int(cap//max(fallback[0]["duration"],1))))]
        speakers={fallback_speaker}; used=sum(r["duration"] for r in selected)
    anonymous.sort(key=lambda x:x[0]); hashes=[x[0] for x in anonymous]
    selected.extend(x[1] for x in anonymous)
    manifest={"rows":selected,"speakers":sorted(speakers),"audio_hashes":hashes,"hours":used/3600,"source_shards":[r["name"] for r in rows[:1]]}
    write_json(ROOT/"data/dev"/"manifest.json",manifest)
    registry.set_meta("dev_speakers",json.dumps(sorted(speakers),ensure_ascii=False)); registry.set_meta("dev_audio_hashes",json.dumps(hashes)); registry.set_meta("dev_initialized","1")

def iter_rows(path: Path, audio_column: str, text_column: str, start_group=0, start_offset=0) -> Iterator[tuple[dict,dict]]:
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(path)
    cols=list(dict.fromkeys([audio_column,text_column,"speaker_id","speaker","style","speech_style"]))
    for rg in range(start_group,pf.num_row_groups):
      tab=pf.read_row_group(rg,columns=[c for c in cols if c in pf.schema_arrow.names])
      for off,row in enumerate(tab.to_pylist()):
        if rg==start_group and off<start_offset: continue
        yield row,{"row_group":rg,"row_offset":off+1}

def decode_audio(payload):
    from .audio import decode_audio
    return decode_audio(_audio_payload(payload))

def log_mel(wave, sample_rate=16000, device="cuda"):
    import torch, torchaudio
    if not torch.is_tensor(wave): wave=torch.as_tensor(wave,dtype=torch.float32)
    wave=wave.to(device)
    mel=torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,n_fft=400,hop_length=160,n_mels=80).to(device)
    return torch.log(mel(wave).clamp_min(1e-5))
