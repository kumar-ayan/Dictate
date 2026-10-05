# nirmiti-dictate

Voice dictation tool for India (Wispr Flow alternative). Hold a hotkey, speak Hindi (later Punjabi, Hinglish), release, and cleaned text is typed into the focused app. Brand-new project, nothing carried over from older repos.

## Hard rules
- FROM SCRATCH. Both models (speech recognition and text cleanup) are written in plain PyTorch and trained here. No pretrained weights, no Whisper, no Hugging Face transformers classes, no Ollama, no LLM or speech APIs, anywhere, including as a placeholder or a teacher.
- Allowed libraries: torch, torchaudio (resampling, io), numpy, pyarrow, sentencepiece (tokenizer training only), soundfile, sounddevice, pynput, pywin32/pyperclip, pyyaml, matplotlib, tqdm, pytest.
- Do not touch or import from any other repo or folder outside this project.
- Never delete or modify user data in `data/`. Never copy audio out of the parquet files into new files. Only read them.
- Never print or log secrets.

## Machine
Windows, RTX 5050 8GB laptop, limited disk. Use `py` to run Python, a venv in `.venv`. bf16, gradient accumulation, batches sized by audio seconds. Check free disk before writing checkpoints or caches.

## Layout
- `dictate/` package: `data/`, `tokenizer/`, `asr/`, `cleanup/`, `train/` (shared trainer, resume, records), `app/` (hotkey app), `cli.py`
- `config/` YAML configs (`default.yaml`, `tiny.yaml` for tests)
- `data/parquet/` user drops speech `.parquet` shards here. `data/dev/` fixed dev set. `data/replay/` small replay subset. `data/text/` optional clean text for the cleanup model.
- `runs/<stage>/` logs, plots, report. `checkpoints/` weights. `registry.db` SQLite state.
- `docs/PROGRESS.md` running log. Update it after every phase.

## Commands (all through `py -m dictate ...`)
`data scan`, `tokenizer train`, `train asr [--resume] [--max-minutes N]`, `train cleanup [--resume]`, `eval asr`, `eval cleanup`, `replay build`, `report`, `run` (starts the dictation app), `test`.

## Invariants (do not break)
- Training state is saved so `--resume` continues exactly: model, optimizer, scheduler, scaler, RNG, step, shard id, row-group and row offset.
- A shard is "consumed" only when its stage finishes. Training state must not depend on old shards still being on disk.
- Tokenizer is trained once and frozen. Later shards reuse it. Log the out-of-vocabulary rate per new shard.
- Dev set is fixed after creation. Dev speakers (or audio hashes if no speaker id) are excluded from all later shards.
- Every training run writes records: config snapshot, git hash, hardware, shards used (name, hash, rows, hours), param count, loss, dev WER/CER, tokens/sec, GPU memory, wall time, checkpoints.
- Checkpoint retention: keep last 3 and best-by-dev-WER only.
- Cleanup model output is guarded at runtime: if length ratio vs input is outside 0.5-1.5, or it errors, inject the raw ASR text.

## Working style
- Build in phases. Run tests after each phase. Do not start the next phase if tests fail.
- Don't change tests or eval data to make numbers look better. Report real numbers, including bad ones.
- Don't ask questions for small choices. Decide, note it in `docs/PROGRESS.md`, and mention it in the final report.
- Keep code small and readable. Short, direct explanations. Use hyphens, not em-dashes. No filler words.