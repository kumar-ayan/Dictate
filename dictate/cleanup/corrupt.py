from __future__ import annotations
import random, re

FILLERS=["um","uh","like","you know","matlab","मतलब","यानी","तो","basically"]
def corrupt(text, rng=None):
    rng=rng or random.Random(); words=text.split(); out=[]; i=0
    while i<len(words):
        w=words[i]
        if rng.random()<.12: out.append(rng.choice(FILLERS))
        out.append(w)
        if rng.random()<.08: out.append(w)
        if i+1<len(words) and rng.random()<.04: out.extend([rng.choice(["sorry","nahi","actually","मेरा मतलब"]),words[i+1]])
        if len(w)>4 and rng.random()<.025:
            at=rng.randrange(1,len(w)-1); out[-1]=w[:at]+w[at+1:]
        i+=1
    return re.sub(r"[.,!?;:।॥]","", " ".join(out)).lower()

def pairs(sentences,seed=0,clean_fraction=.15):
    rng=random.Random(seed)
    for sentence in sentences:
        if rng.random()<clean_fraction: yield sentence,sentence
        else: yield corrupt(sentence,rng),sentence
