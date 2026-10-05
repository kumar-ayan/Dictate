from __future__ import annotations
import json, logging, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_config(path: str | None = None) -> dict:
    import yaml
    p = Path(path) if path else ROOT / "config" / "default.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def ensure_layout() -> None:
    for p in ("data/parquet", "data/dev", "data/replay", "data/text", "runs", "checkpoints", "config", "docs"):
        (ROOT / p).mkdir(parents=True, exist_ok=True)

def setup_logging() -> logging.Logger:
    ensure_layout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("dictate")

def disk_guard(path: Path, needed_bytes: int = 0, reserve_bytes: int = 2_000_000_000) -> None:
    free = shutil.disk_usage(path.parent if path.parent.exists() else ROOT).free
    if free - needed_bytes < reserve_bytes:
        raise OSError(f"Disk guard: need {needed_bytes} bytes plus {reserve_bytes} byte reserve; free {free}")

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unavailable"
