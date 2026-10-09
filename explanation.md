# Nirmiti Dictate - project explanation

This document describes the code and training state in this workspace as of 2026-10-04. The ASR model has completed its first training stage, but its real dev score still needs to be generated with the fixed evaluation command. Do not treat training completion as proof of recognition quality.

## Project goal and constraints

Nirmiti Dictate is a Windows hotkey dictation tool focused first on Hindi. The intended flow is: hold a key, speak, release, recognize speech locally, clean obvious disfluencies, and paste text into the focused application.

Both learned models are implemented in plain PyTorch and are intended to train from scratch. The project does not load pretrained weights, call speech or text APIs, or use an outside LLM or model framework. Audio remains inside the source parquet files; training and evaluation decode it in memory. The fixed dev and replay files contain references into those parquet files rather than copied audio.

## Project map

| Path | Contents |
|---|---|
| `dictate/data/` | Parquet schema detection, audio decoding/resampling, text normalization, dev/replay references, log-mel features, duration batching. |
| `dictate/tokenizer/` | SentencePiece BPE training, frozen tokenizer loading, transcript enumeration. |
| `dictate/asr/` | PyTorch Conformer encoder and CTC output layer. |
| `dictate/cleanup/` | Text corpus loading, synthetic corruption, rules baseline, encoder-decoder model, evaluation and training. |
| `dictate/train/` | ASR training, metrics, checkpoint state and shared trainer helpers. |
| `dictate/app/` | Hotkey recording, VAD-style energy trim, inference, cleanup guard, clipboard insertion. |
| `config/default.yaml` | Full training and app defaults. |
| `config/tiny.yaml` | Small CPU-friendly plumbing/test architecture. |
| `data/parquet/` | User-provided speech shards. |
| `data/dev/` | Fixed dev manifest with transcript and source row references. |
| `data/replay/` | Sample manifests that allow old shards to be mixed into later ASR stages. |
| `data/text/` | Optional clean text corpus for cleanup training. |
| `registry.db` | SQLite metadata for shards, tokenizer, stages, and dictation history. |
| `checkpoints/` | Resumable checkpoints and selected best model files. |
| `runs/<stage>/` | Config/hardware/shard snapshots, metrics, evaluations, prediction samples, plots, and report. |

## Data path

### Scan and registry

Put `.parquet` files in `data/parquet/`, then run `python -m dictate data scan`. The scanner finds unseen or changed files by name and SHA-256, prints the Arrow schema, detects an audio field and transcript field, and registers per-shard metadata. Audio-column or text-column overrides live under `data.audio_column` and `data.text_column` in YAML.

The scan records row count, decoded audio hours, a duration histogram, speaker count, speech-style counts when available, Latin-letter transcript share, punctuation share, common characters, decode failures, and tokenizer OOV rate when the tokenizer already exists. A later scan ignores a shard whose path and hash have not changed.

### Audio and transcript handling

Audio bytes are decoded from each parquet row in memory, converted to mono, and resampled to 16 kHz. The project does not write extracted audio files. Rows without usable audio, empty text, clips shorter than 1 second, and clips longer than 30 seconds are not used for ASR.

Text is normalized with Unicode NFC, zero-width character removal, whitespace collapse, and trimming. The normalized text is used for training while source parquet values remain unchanged. Hindi text is not translated or transliterated.

### Dev split and replay

On the first scan, the project creates a fixed dev manifest in `data/dev/`. It selects about 3% of speakers using stable hashes where speaker IDs are available, with a target cap near two hours. If no speaker ID exists, it samples by audio hash. Dev speakers or hashes are excluded from future ASR training. The manifest stores source path, row group, row offset, text, and audio-column name; the audio stays in parquet.

`python -m dictate replay build --max-hours 3` creates a deterministic sample manifest under `data/replay/`. Later ASR stages can mix replay examples; the default replay fraction is 15%. A replay manifest references its source and hash, so source shards must remain present and unmodified while the manifest is used.

### Batching and features

The ASR reader streams Arrow row groups. It decodes a bounded shuffle buffer, forms deterministic length buckets, and packs batches under a total-audio-seconds budget (`data.max_batch_seconds`, default 20). It stores the seed, shard, row group, row offset, pool, replay cursor/signature, and partial gradient accumulation in resumable checkpoints.

Log-mel spectrograms are generated in the training batch path and moved to CUDA when available. The current feature setup uses 80 mel bins, a 400-sample FFT window, and a 160-sample hop. CUDA training uses BF16 autocast, gradient accumulation, gradient clipping, and optional activation checkpointing.

## Tokenizer

`python -m dictate tokenizer train` trains SentencePiece BPE from every registered transcript. The default vocabulary size is 3,000 (configurable in YAML), with high character coverage and byte fallback. The model file hash is saved in `registry.db`; a second tokenizer-training attempt is rejected once frozen. New transcript scans compute OOV rate against the existing tokenizer.

Cleanup reuses the ASR tokenizer so both models share the same Hindi, Latin, and byte representation. This avoids a second vocabulary and keeps their text IDs compatible.

## ASR model and algorithm

The recognizer is a Conformer-CTC acoustic model written in PyTorch:

1. Input log-mel frames pass through two stride-2 convolution layers, reducing time resolution by approximately 4x.
2. A projection maps the subsampled audio into the model width.
3. Conformer blocks apply macaron-style feed-forward layers, rotary self-attention, a gated pointwise/depthwise convolution module, residual connections, and LayerNorm.
4. A linear classifier predicts tokenizer symbols plus a dedicated CTC blank symbol at each output frame.
5. CTC loss aligns frame-level outputs with transcripts without requiring hand-aligned word or character timestamps.
6. Inference uses greedy CTC decoding: take the most likely symbol per frame, merge repeated adjacent symbols, remove blank symbols, and decode the resulting token IDs.

The default architecture is width 512, 8 blocks, 8 attention heads, and a feed-forward multiplier of 4. For this workspace's 3,000-piece tokenizer, the completed model reports **57,558,457 parameters**. Default ASR training is 10 epochs, AdamW with peak learning rate `0.0005`, 1,000 warmup steps followed by cosine decay, gradient accumulation of 4, batch budget of 20 audio seconds, clipping at 1.0, and SpecAugment enabled with probability 0.5. These values can be changed in `config/default.yaml`; changing a config while resuming is rejected.

Training records loss, learning rate, gradient norm, token throughput, GPU memory, step duration, and dev WER/CER. The optimizer-step schedule is estimated from audio seconds, batch-seconds budget, epochs, and gradient accumulation. The previous estimate omitted gradient accumulation and left completed `asr-001` at a learning rate near its peak; the estimate is corrected for future runs. This may have contributed to the poor result, but does not prove it is the only cause.

The retained checkpoint policy is the most recent three step checkpoints plus a best-by-dev-WER model checkpoint. Disk space is checked before checkpoint writes.

## Cleanup model and algorithm

The cleanup model is a hand-written encoder-decoder Transformer, also in plain PyTorch. It uses RMSNorm, rotary positions, manually implemented attention, causal decoder self-attention, cross-attention, and SwiGLU feed-forward layers. Default dimensions are width 512 with 3 encoder and 3 decoder blocks; its actual parameter count is recorded in `runs/cleanup-*/model.json` after training.

Cleanup training draws sentence text from registered speech transcripts and optional `.txt`, `.tsv`, `.csv`, or `.parquet` files in `data/text/`. It creates synthetic pairs by adding fillers (`um`, `uh`, `like`, `you know`, `matlab`, `मतलब`, `यानी`, `तो`, `basically`), repeats, self-corrections, case/punctuation stripping, and light character deletion. About 15% of generated pairs are already clean, with input equal to target. Train/validation assignment is deterministic by sentence hash.

Evaluation reports held-out exact match, character error rate, over-edit rate on clean inputs, length-ratio violations, and a regex rules baseline. The file `evals/cleanup_real.tsv` is intended for 30 real pairs that can be extended. The app can use rules or a trained cleanup checkpoint. Runtime cleanup output is accepted only if it succeeds and stays within 0.5-1.5 times the input character length; otherwise it falls back to raw ASR text.

Punctuation restoration depends on punctuated targets. The cleanup report measures the punctuation rate of its text corpus; if most transcript sentences lack punctuation, add clean, punctuated examples to `data/text/`.

## Training stages and checkpoints

### First ASR stage

The first `python -m dictate train asr` trains a new model on all registered shards with `consumed_asr=0`, using the default 10 epochs. The stage is named `asr-001`, then those shards are marked consumed after successful completion. The optimizer and scheduler are created for that stage.

### Resume versus next stage

`python -m dictate train asr --resume` is only for a stage that stopped early through `--max-minutes` or Ctrl+C. It restores the exact saved training state and requires the same config and shard set. A stage that says `Finished ...` is complete and cannot be resumed as if it were interrupted.

When new unconsumed shards are scanned, a new ASR stage starts from the latest weights of the previous stage, with a fresh optimizer and a lower peak learning rate (`stage_lr_scale`, default 0.3). It receives a new name such as `asr-002`, and has its own metrics files and loss/WER/LR plots. `report` lists stages together. It does not append the new run to the old stage's plot.

With no new shards, plain `train asr` reports that there are no unconsumed shards. To deliberately repeat the most recently completed ASR stage's registered shards, run `python -m dictate train asr --repeat-last-stage`. This verifies the registered hashes and source files, creates a new stage, initializes from the previous stage's latest weights, and creates a fresh optimizer at the lower stage learning rate. Replay mixing is disabled for this stage because it already trains on the selected shards. Its plots live in a separate `runs/asr-NNN/` folder; `report` lists both stages. The global best checkpoint used by the app is replaced only if the new dev WER improves on the current global best. It does not edit data files. If a repeated stage is stopped with `--max-minutes`, resume it using both flags: `python -m dictate train asr --repeat-last-stage --resume`. Evaluate before repeating because another pass can overfit and does not guarantee better WER.

To deliberately repeat all registered parquet files currently present in `data/parquet/` together, use `python -m dictate train asr --repeat-all-shards`. Each file must already be registered and match its scanned hash. This creates a new stage from the latest ASR weights with a fresh optimizer and no replay mixing. Resume a time-limited run with `--repeat-all-shards --resume`.

Cleanup also tracks its consumed cleanup shards and creates new named stages. Interrupted ASR and cleanup jobs save resumable checkpoints. ASR saves at safe shuffle-pool boundaries, so `--max-minutes N` may run slightly beyond N minutes while it finishes its current pool.

## Dictation app

`python -m dictate run` loads the frozen tokenizer and ASR checkpoint once, checks tokenizer hashes, and warms up the model before listening. The default hotkey is Right Ctrl, configurable under `app.hotkey` (Left Ctrl, F-keys, and single characters are also supported). Recordings shorter than 300 ms are ignored.

The microphone stream is mono float audio at 16 kHz. A simple 20 ms RMS energy detector trims silence, with a 200 ms margin. The app runs ASR, applies `data/dictionary.txt` entries written as `heard spelling<TAB>preferred spelling`, then applies cleanup. It prints stage timings, saves each dictation and model-version hash in SQLite, pastes via the clipboard, and restores the clipboard's old contents.

Without `checkpoints/best-asr.pt`, the app exits with an explanation. It does not use an outside model as fallback. With `app.cleanup: model` but no cleanup checkpoint, it explains that it will use rules instead.

## Command reference

Run commands from the activated project environment:

```cmd
python -m dictate --help
python -m dictate data scan
python -m dictate tokenizer train
python -m dictate train asr
python -m dictate train asr --max-minutes 10
python -m dictate train asr --resume
python -m dictate train asr --repeat-last-stage
python -m dictate eval asr
python -m dictate replay build --max-hours 3
python -m dictate train cleanup
python -m dictate train cleanup --max-minutes 10
python -m dictate train cleanup --resume
python -m dictate eval cleanup
python -m dictate report
python -m dictate run
python -m dictate test
```

Use global config selection before the command, for example `python -m dictate --config config/tiny.yaml test`. `config/tiny.yaml` reduces model and data sizes for plumbing tests; it is not the production training config and should not be used to resume a default-config run.

`python -m dictate fake-data` writes a synthetic WAV-in-parquet fixture at `data/parquet/fake_hindi.parquet`. Run it only in a disposable workspace because scanning that location registers it like any other shard. Normal tests generate fixtures under temporary directories.

## Current real-data status

The workspace has a completed ASR stage named `asr-001`: 14,708 optimizer steps and 57,558,457 parameters using the default 10-epoch configuration. Its first dev evaluation failed because the batch dimension was dropped in greedy CTC decoding; the fix was made and all 126 rows then evaluated. The measured WER was 0.9989 and CER was 0.9919, with mostly blank or one-token predictions. This is unusable ASR quality. The run's final learning rate was 0.0004008 from a 0.0005 peak, showing that the old schedule had not decayed as intended. The schedule estimate now accounts for gradient accumulation. A same-data stage can be started with `--repeat-last-stage`, but improvement is not guaranteed; evaluate each stage and compare dev metrics.

Latest test result after the decoder, schedule-estimation, repeat-stage, and global-best regression tests: **23 passed**, two PyArrow deprecation warnings. The synthetic tests validate software behavior; they do not prove model quality. Real microphone, hotkey permissions, clipboard behavior on the target Windows laptop, and Hindi dictation quality still need user-side checks.
