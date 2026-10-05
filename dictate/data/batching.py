from __future__ import annotations
import random

class LengthBucketBatcher:
    """Deterministic shuffled pools, sorted by duration, packed under seconds budget."""
    def __init__(self, rows, max_seconds=120, pool_size=512, seed=0):
        self.rows=list(rows); self.max_seconds=max_seconds; self.pool_size=pool_size; self.seed=seed
    def __iter__(self):
        rng=random.Random(self.seed); indices=list(range(len(self.rows))); rng.shuffle(indices); batch=[]; total=0.
        for start in range(0,len(indices),self.pool_size):
            pool=sorted((self.rows[i] for i in indices[start:start+self.pool_size]),key=lambda x:float(x["duration"]))
            for row in pool:
                d=float(row["duration"])
                if batch and total+d>self.max_seconds: yield batch; batch=[]; total=0.
                batch.append(row); total+=d
        if batch: yield batch
    def state_dict(self): return {"seed":self.seed,"max_seconds":self.max_seconds,"pool_size":self.pool_size}
    def load_state_dict(self,state):
        if state["seed"]!=self.seed: raise ValueError("Batcher seed mismatch")
