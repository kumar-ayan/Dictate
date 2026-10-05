from __future__ import annotations
import re

FILLERS=re.compile(r"\b(?:um+|uh+|like|you know|matlab|basically)\b|(?:मतलब|यानी|तो)",re.I)
def clean_rules(text):
    text=FILLERS.sub(" ",text)
    text=re.sub(r"\b(\w+)(?:\s+\1\b)+",r"\1",text,flags=re.I)
    text=re.sub(r"\b(.+?)\s+(?:sorry|actually|nahi)\s+(.+)",r"\2",text,flags=re.I)
    text=re.sub(r"\b.+?\s+(?:मेरा मतलब|मतलब)\s+(.+)",r"\1",text)
    return re.sub(r"\s+"," ",text).strip(" ,")

def guarded_cleanup(raw, fn, fallback=None):
    try:
        result=fn(raw); ratio=len(result)/max(len(raw),1)
        return result if .5<=ratio<=1.5 else (raw if fallback is None else fallback)
    except Exception: return raw if fallback is None else fallback
