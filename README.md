# Nirmiti Dictate

Nirmiti Dictate is a local Hindi speech dictation project for Windows. Hold a hotkey, speak, and release it to recognize and insert text in the focused app. Speech recognition and text cleanup use models implemented in plain PyTorch and trained from scratch here. There are no pretrained weights, hosted inference services, Whisper, Hugging Face model classes, or LLMs.

See [explanation.md](explanation.md) for architecture, algorithms, data handling, training behavior, and the current project status. See [docs/PROGRESS.md](docs/PROGRESS.md) for the phase log.

## Requirements and setup

- Windows and Python 3.11+
- An NVIDIA GPU is recommended. The target machine is an RTX 5050 Laptop GPU with 8 GB VRAM; CPU mode is available but training will be much slower.
- Keep source parquet files available while they are used by training, dev evaluation, or replay references.

From Command Prompt in the project folder:

```cmd
cd /d D:\Dictate
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e .
```

If `.venv` already exists and works, activate it instead of recreating it. The commands below use `python` after activation so they run inside that environment. If you prefer the Python launcher, use `py -m dictate ...` with the same Python 3.11 environment.

## First real-data training run

Put IndicVoices Hindi `.parquet` shards in `data/parquet/`. Then run these commands in order:

```cmd
python -m dictate data scan
python -m dictate tokenizer train
python -m dictate train asr
python -m dictate eval asr
python -m dictate report
```

`data scan` prints each new shard's schema, identifies the audio and transcript columns, registers its hash and statistics in `registry.db`, and creates the fixed dev manifest on the first scan. Set `data.audio_column` or `data.text_column` in a config YAML only if automatic detection chooses the wrong fields.

Train the tokenizer once, after scanning the first corpus. It uses all registered transcripts, defaults to a 3,000-piece SentencePiece BPE vocabulary, and is frozen after training. Later shards use the same tokenizer and receive an OOV-rate statistic during scan. Do not retrain it after an ASR checkpoint exists.

The default ASR config trains for 10 epochs. A completed run consumes its registered shards for ASR. The first ASR stage trains from random initialization. Later stages start from the preceding stage's latest weights, create a fresh optimizer, use a lower peak learning rate, and can mix replay data.

### Continue an interrupted ASR stage

To stop a long stage cleanly after a time limit:

```cmd
python -m dictate train asr --max-minutes 10
```

Resume that still-incomplete stage with:

```cmd
python -m dictate train asr --resume
```

Resume restores model, optimizer, scheduler, RNG, and the data cursor. Keep the original config and shard files unchanged. `--resume` is for an interrupted stage, not a completed one.

### Train on new shards

Add the new `.parquet` files to `data/parquet/`, then run:

```cmd
python -m dictate data scan
python -m dictate replay build --max-hours 3
python -m dictate train asr
python -m dictate eval asr
python -m dictate report
```

The replay command creates a small JSON manifest of row references. Audio remains embedded in the original parquet. Later ASR stages use the configured replay fraction, 15% by default. You can select a shard explicitly with `python -m dictate replay build SHARD_NAME.parquet`.

After a completed stage, a plain `train asr` uses only new unconsumed shards. Use the explicit repeat command below to deliberately make another pass over the latest completed stage's data. `--resume` is only for interrupted stages.

To intentionally train another stage on the most recently completed ASR stage's same shards, use:

```cmd
python -m dictate train asr --repeat-last-stage
python -m dictate eval asr
python -m dictate report
```

This validates the stored shard hashes, starts a new stage from the previous stage's latest weights, uses a fresh optimizer and lower stage learning rate, and writes a separate graph under the new `runs/asr-NNN/` directory. A repeat stage disables replay mixing, since it already trains on the selected original shards. The app's global best checkpoint is replaced only if the new stage's dev WER improves on it. The command does not change files under `data/`. Use this only after reviewing the current dev score; more epochs can overfit and do not guarantee better WER. If you time-limit the repeated stage, resume it with both flags: `python -m dictate train asr --repeat-last-stage --resume`.

## Cleanup training

Cleanup reads transcripts from registered speech shards and optional clean text under `data/text/`. Accepted text formats are `.txt` (one sentence per line), `.tsv`, `.csv`, and `.parquet` with a recognized text column. Punctuated clean text is useful for learning punctuation restoration.

```cmd
python -m dictate train cleanup
python -m dictate eval cleanup
python -m dictate report
```

Cleanup training synthesizes noisy-to-clean sentence pairs from the corpus. Evaluation compares the learned model with a simple rules baseline and reports exact match, character error rate, clean-input over-edit rate, length-ratio violations, and optional real-pair results. Extend `evals/cleanup_real.tsv` with real examples to measure behavior you care about.

## Run dictation

After `eval asr` has created `checkpoints/best-asr.pt`:

```cmd
python -m dictate run
```

The default hotkey is Right Ctrl. Hold it while speaking and release it to process the recording. Taps under 300 ms are ignored. The app loads models once at startup, trims quiet audio, runs ASR, applies the spelling dictionary at `data/dictionary.txt`, applies the configured cleanup mode, copies the result to the clipboard, pastes it, and restores the previous clipboard contents. It logs the raw, cleaned, and final text plus timings and model hashes in `registry.db`.

Cleanup defaults to `rules`. To use a trained cleanup model, set `app.cleanup: model` in `config/default.yaml`. If a cleanup checkpoint is missing, the app reports that and uses rules. If cleanup errors or changes output length beyond the 0.5 to 1.5 input ratio, the raw ASR text is used.

## Commands

Run `python -m dictate --help` for the command list. Implemented commands:

| Command | Purpose |
|---|---|
| `python -m dictate data scan` | Detect, hash, and register new parquet shards and corpus statistics. |
| `python -m dictate tokenizer train` | Train and freeze the tokenizer once. |
| `python -m dictate train asr` | Train on all unconsumed ASR shards. |
| `python -m dictate train asr --resume` | Resume an interrupted ASR stage exactly. |
| `python -m dictate train asr --repeat-last-stage` | Start a new stage on the latest completed ASR stage's shards, initialized from its latest weights. |
| `python -m dictate train asr --max-minutes N` | Save a resumable ASR checkpoint after the time limit is reached at a safe data boundary. |
| `python -m dictate train cleanup` | Train cleanup on registered transcripts and optional clean text. |
| `python -m dictate train cleanup --resume` | Resume an interrupted cleanup stage. |
| `python -m dictate train cleanup --max-minutes N` | Save a resumable cleanup checkpoint after the time limit. |
| `python -m dictate eval asr` | Evaluate on the fixed dev set and print WER, CER, and sample predictions. |
| `python -m dictate eval cleanup` | Evaluate cleanup against held-out and real pairs. |
| `python -m dictate replay build --max-hours 3` | Sample replay row references from a registered shard. |
| `python -m dictate report` | Print a table of completed stages. |
| `python -m dictate run` | Start the local hotkey dictation app. |
| `python -m dictate test` | Run the pytest suite. |
| `python -m pytest -q` | Run the pytest suite directly. |
| `python -m dictate fake-data` | Create a synthetic fixture at `data/parquet/fake_hindi.parquet`. Use only in a disposable/test workspace. |

To use a config other than `config/default.yaml`, put the global option before the command, for example:

```cmd
python -m dictate --config config/tiny.yaml train asr
```

`config/tiny.yaml` is for smoke tests, not real training. Do not use it to resume a default-config stage.

## Files and records

- `data/parquet/`: user-provided source shards. The program reads them and does not extract audio to separate files.
- `data/dev/manifest.json`: fixed dev references; audio stays in source parquet.
- `data/replay/`: replay reference manifests.
- `data/text/`: optional cleanup corpus.
- `registry.db`: shard, stage, tokenizer, and dictation records.
- `checkpoints/`: resumable stage checkpoints and selected best model checkpoints.
- `runs/<stage>/`: config and hardware snapshots, shard list, JSONL metrics/evaluations, predictions, plots, and report.

## Current status

The current ASR stage (`asr-001`) completed 14,708 optimizer steps with 57,558,457 parameters using the default 10-epoch configuration. Dev evaluation reports WER 0.9989 and CER 0.9919 on 126 examples, with mostly blank predictions. The run ended at learning rate 0.0004008 from a 0.0005 peak; the schedule estimate had failed to divide batch count by gradient accumulation. The estimate is fixed for future runs. That schedule issue may have contributed to the poor recognition, but does not prove the sole cause. Review `explanation.md` before choosing a same-data repeat or adding more data.

The current test suite result is **23 passed**, with two PyArrow deprecation warnings. Synthetic tests check plumbing and do not measure Hindi recognition quality. See [explanation.md](explanation.md) for limitations and algorithm details.
