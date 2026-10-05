from __future__ import annotations
import csv, hashlib, re
from pathlib import Path
from .. import registry
from ..data.parquet import normalize_text
from ..util import ROOT

TEXT_COLUMNS=("text","sentence","transcript","normalized_text","clean_text")

def iter_cleanup_texts():
    import pyarrow.parquet as pq
    from ..tokenizer.spm import texts as shard_texts
    yield from (normalize_text(x) for x in shard_texts() if normalize_text(x))
    for path in sorted((ROOT/"data/text").glob("**/*")):
        if not path.is_file(): continue
        if path.suffix.lower()==".txt":
            with path.open(encoding="utf-8-sig") as f:
                for line in f:
                    value=line.rstrip("\r\n")
                    value=normalize_text(value)
                    if value: yield value
        elif path.suffix.lower() in (".tsv",".csv"):
            delimiter="\t" if path.suffix.lower()==".tsv" else ","
            with path.open(encoding="utf-8-sig",newline="") as f:
                reader=csv.DictReader(f,delimiter=delimiter)
                if reader.fieldnames:
                    text_key=next((n for n in reader.fieldnames if n.lower().strip() in TEXT_COLUMNS),reader.fieldnames[-1])
                    for row in reader:
                        value=normalize_text(str(row.get(text_key) or ""))
                        if value: yield value
        elif path.suffix.lower()==".parquet":
            pf=pq.ParquetFile(path); col=next((n for n in TEXT_COLUMNS if n in pf.schema_arrow.names),None)
            if col:
                for rg in range(pf.num_row_groups):
                    for value in pf.read_row_group(rg,columns=[col]).column(0).to_pylist():
                        value=normalize_text(str(value or ""))
                        if value: yield value

def split_for_eval(text,validation_fraction=.1):
    value=int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8],16)/0xffffffff
    return value<validation_fraction

def punctuation_stats(texts):
    total=punctuated=0
    for text in texts:
        total+=1; punctuated+=bool(re.search(r"[.,!?;:।॥]",text))
    return {"sentences":total,"punctuated":punctuated,"punctuation_rate":punctuated/max(total,1)}
