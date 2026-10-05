from __future__ import annotations
import random, signal, time
from itertools import islice
from pathlib import Path
import numpy as np
import torch
from ..util import disk_guard

STOP=False
def _stop(*_):
    global STOP; STOP=True

def restore_rng_state(state):
    """RNG generators require CPU byte states, even when checkpoints load on CUDA."""
    random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])

def save_checkpoint(path,model,optimizer=None,scheduler=None,scaler=None,step=0,data_state=None,loss=None,extra=None):
    estimate=sum(p.numel()*p.element_size() for p in model.parameters())*(4 if optimizer is not None else 1)
    disk_guard(Path(path),needed_bytes=estimate,reserve_bytes=500_000_000)
    state={"model":model.state_dict(),"step":step,"data_state":data_state,"loss":loss,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    for name,obj in (("optimizer",optimizer),("scheduler",scheduler),("scaler",scaler)):
        if obj is not None: state[name]=obj.state_dict()
    if extra: state.update(extra)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp"); torch.save(state,temp); temp.replace(path)

def load_checkpoint(path,model,optimizer=None,scheduler=None,scaler=None,map_location="cpu"):
    state=torch.load(path,map_location=map_location,weights_only=False); model.load_state_dict(state["model"])
    for name,obj in (("optimizer",optimizer),("scheduler",scheduler),("scaler",scaler)):
        if obj is not None and name in state: obj.load_state_dict(state[name])
    restore_rng_state(state)
    return state

def retain_checkpoints(directory,best_name="best.pt",keep=3):
    p=Path(directory); files=sorted(p.glob("step-*.pt"),key=lambda x:x.name,reverse=True)
    for x in files[keep:]: x.unlink()

def train_steps(model,batches,loss_fn,optimizer,steps,checkpoint,device="cpu",resume=False,max_minutes=None,checkpoint_every=10,scheduler=None,scaler=None):
    """Train a deterministic iterable; checkpoint includes RNG and next batch position."""
    global STOP; STOP=False
    scheduler=scheduler or torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    scaler=scaler or torch.amp.GradScaler("cuda",enabled=False)
    start=offset=0; last_loss=None; data_state=None
    state=None
    if resume:
        if not Path(checkpoint).exists(): raise FileNotFoundError(checkpoint)
        state=load_checkpoint(checkpoint,model,optimizer,scheduler,scaler,map_location=device)
        start=state["step"]; data_state=state.get("data_state") or {}; offset=int(data_state.get("batch_index",start)); last_loss=state.get("loss")
    signal.signal(signal.SIGINT,_stop); began=time.monotonic(); model.to(device).train(); optimizer.zero_grad(set_to_none=True)
    # Stateful samplers restore their own cursor. Plain iterables replay skipped
    # batches; those batches must be deterministic and have no model side effects.
    stateful=hasattr(batches,"load_state_dict") and data_state and data_state.get("iterator_state") is not None
    if stateful: batches.load_state_dict(data_state["iterator_state"])
    iterator=iter(batches); batch_index=0
    if not stateful:
        for _ in islice(iterator,offset): batch_index+=1
        if state is not None:
            restore_rng_state(state)
    committed_rng={"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    step=start
    try:
        for batch in iterator:
            batch_index+=1
            if step>=start+steps or STOP: break
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
                loss=loss_fn(model,batch)
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step+=1; last_loss=float(loss.detach())
            iterator_state=batches.state_dict() if hasattr(batches,"state_dict") else None
            data_state={"batch_index":batch_index,"iterator_state":iterator_state}
            committed_rng={"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            if step%checkpoint_every==0 or STOP or max_minutes is not None and (time.monotonic()-began)/60>=max_minutes:
                save_checkpoint(checkpoint,model,optimizer,scheduler,scaler,step,data_state,last_loss)
            if STOP or max_minutes is not None and (time.monotonic()-began)/60>=max_minutes: break
    except KeyboardInterrupt:
        restore_rng_state(committed_rng)
    save_checkpoint(checkpoint,model,optimizer,scheduler,scaler,step,data_state,last_loss)
    return {"step":step,"loss":last_loss,"data_state":data_state}
