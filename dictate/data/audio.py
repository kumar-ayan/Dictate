from __future__ import annotations
import io
import numpy as np

def decode_audio(payload):
    import torch, torchaudio
    if payload is None: raise ValueError("Missing audio bytes")
    if isinstance(payload,memoryview): payload=payload.tobytes()
    if not isinstance(payload,(bytes,bytearray)): raise TypeError("Audio payload must be embedded bytes")
    try:
        wave,sr=torchaudio.load(io.BytesIO(payload))
    except Exception:
        import soundfile as sf
        array,sr=sf.read(io.BytesIO(payload),dtype="float32",always_2d=True)
        wave=torch.from_numpy(array.T)
    wave=wave.float().mean(dim=0)
    if sr!=16000: wave=torchaudio.functional.resample(wave,sr,16000)
    return wave.contiguous(),16000
