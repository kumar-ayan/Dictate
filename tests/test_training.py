import torch
from torch import nn
import pytest
import random
import numpy as np

from dictate.train import train_steps, retain_checkpoints
from dictate.train.trainer import restore_rng_state

def _run_segment(model,batches,checkpoint,steps,resume=False,monkeypatch=None):
    import dictate.train.trainer as trainer
    if monkeypatch: monkeypatch.setattr(trainer,"disk_guard",lambda *args,**kwargs: None)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.01)
    def loss_fn(m,batch): return torch.nn.functional.mse_loss(m(batch[0]),batch[1])
    return train_steps(model,batches,loss_fn,optimizer,steps,checkpoint,device="cpu",resume=resume,checkpoint_every=1)

def test_resume_matches_uninterrupted(tmp_path,monkeypatch):
    torch.manual_seed(17)
    batches=[(torch.randn(3,4),torch.randn(3,2)) for _ in range(6)]
    uninterrupted=nn.Sequential(nn.Linear(4,16),nn.Dropout(.2),nn.Linear(16,2))
    train_rng=torch.get_rng_state()
    initial={k:v.clone() for k,v in uninterrupted.state_dict().items()}
    full_result=_run_segment(uninterrupted,batches,tmp_path/"full.pt",6,monkeypatch=monkeypatch)

    split=nn.Sequential(nn.Linear(4,16),nn.Dropout(.2),nn.Linear(16,2)); split.load_state_dict(initial)
    torch.set_rng_state(train_rng)
    _run_segment(split,batches,tmp_path/"resume.pt",3,monkeypatch=monkeypatch)
    resumed=nn.Sequential(nn.Linear(4,16),nn.Dropout(.2),nn.Linear(16,2))
    result=_run_segment(resumed,batches,tmp_path/"resume.pt",3,resume=True,monkeypatch=monkeypatch)
    assert result["step"]==6 and result["data_state"]["batch_index"]==6
    assert result["loss"]==full_result["loss"]
    for name,value in uninterrupted.state_dict().items():
        assert torch.equal(value,resumed.state_dict()[name])

def test_checkpoint_retention_keeps_three_and_best(tmp_path):
    for step in range(1,6): (tmp_path/f"step-{step:08d}.pt").write_bytes(b"x")
    (tmp_path/"best.pt").write_bytes(b"best")
    retain_checkpoints(tmp_path)
    assert len(list(tmp_path.glob("step-*.pt")))==3
    assert {p.name for p in tmp_path.glob("step-*.pt")}=={"step-00000003.pt","step-00000004.pt","step-00000005.pt"}
    assert (tmp_path/"best.pt").exists()

def test_restore_rng_state_after_cuda_checkpoint_load(tmp_path):
    if not torch.cuda.is_available(): pytest.skip("requires CUDA to exercise map_location=cuda")
    state={"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}
    path=tmp_path/"rng.pt"; torch.save(state,path)
    loaded=torch.load(path,map_location="cuda",weights_only=False)
    assert loaded["torch_rng"].is_cuda
    restore_rng_state(loaded)

def test_tiny_asr_overfits_twenty_repeated_utterances():
    from dictate.asr.model import ConformerCTC
    torch.manual_seed(9); old_threads=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        model=ConformerCTC(3,d_model=16,layers=1,heads=2,ff_mult=2,dropout=0)
        features=torch.randn(20,40,80); lengths=torch.full((20,),40,dtype=torch.long); targets=torch.tensor([1,2]*20); target_lengths=torch.full((20,),2,dtype=torch.long)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01); loss_fn=torch.nn.CTCLoss(blank=3,zero_infinity=True)
        for _ in range(150):
            optimizer.zero_grad(set_to_none=True); log_probs=model(features,lengths).transpose(0,1); loss=loss_fn(log_probs,targets,lengths.new_full((20,),10),target_lengths); loss.backward(); optimizer.step()
        prediction=model(features[:1],lengths[:1])[0].argmax(-1).tolist(); decoded=[]; last=None
        for token in prediction:
            if token!=last and token!=3: decoded.append(token)
            last=token
        assert decoded==[1,2]
    finally: torch.set_num_threads(old_threads)
