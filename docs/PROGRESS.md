# Progress

## Phase 0 - scaffold

Package layout, YAML configs, CLI, SQLite registry, disk guard, logging, packaging, and docs are present. Registry tables are created lazily. `fake-data` is explicit so scans never generate files unexpectedly.

## Phase 1 - data layer

`data scan` prints each new parquet schema, detects embedded audio and transcript columns (or honors config overrides), hashes/registers shards, and records duration, speaker/style, language-script, punctuation, character, and tokenizer OOV statistics. Audio decodes in memory through torchaudio/soundfile and resamples to mono 16 kHz. Text normalization preserves the original parquet value and normalizes the training copy.

The fixed dev set stores row references and text, never copied audio. Speaker IDs or audio hashes are held out across later training shards. Replay manifests store sampled row references and mix old audio in later ASR stages. The ASR reader streams pyarrow row groups into bounded 128-row pools, shuffles each pool deterministically, sorts short length buckets, and packs batches by total seconds. Resume checkpoints are committed only at pool boundaries and save the seed, pool index, cursor, and any partial accumulation gradients, so a resumed run reconstructs the same next pool without audio caches.

## Phase 2 - tokenizer

SentencePiece BPE uses byte fallback, supports Devanagari and Latin, is frozen after the first successful training, and stores its hash/path in the registry. New shard scans calculate OOV rate against it. The cleanup model reuses this frozen tokenizer because it is shared Hindi transcript text and has byte fallback; this avoids incompatible vocabularies between ASR and cleanup.

## Phase 3 - ASR

Implemented a from-scratch PyTorch Conformer-CTC with 4x convolutional subsampling, rotary self-attention, macaron feed-forward layers, convolution modules, optional gradient checkpointing, CTC decoding, SpecAugment, BF16 on CUDA, accumulation, AdamW, warmup/cosine schedule, clipping, dev WER/CER, and sample output. Default is 512 width, 8 blocks, 8 heads, FF multiplier 4. The user's completed model at the 3,000-piece tokenizer size has 57,558,457 parameters. Its 126-row dev evaluation now runs and reports poor WER 0.9989/CER 0.9919 with mostly blank outputs.

Stages consume only unconsumed shards, warm-start later stages from the prior weights with a fresh optimizer and lower LR, and mix replay references. Checkpoints include model/optimizer/scheduler/scaler, RNG states, data cursor, and partial gradient-accumulation state. `--max-minutes` and Ctrl-C finish the current shuffle pool, then save, so the saved cursor never jumps over untrained rows. Checkpoints retain the latest three plus separate best-by-WER weights. Run records include configs, hardware, shard hashes/OOV, metrics, evaluations, plots, and report files.

## Phase 4 - cleanup

Implemented a hand-written encoder-decoder transformer using RMSNorm, RoPE, SwiGLU, and manual attention; synthetic filler/repetition/self-correction/noise corruption with clean pairs; sentence-level deterministic held-out split; rule baseline; held-out metrics; and 30 real evaluation pairs. It reuses the frozen ASR tokenizer for compatible Hindi/Latin/byte coverage. At vocabulary size 3,000, default dimensions yield 29,883,904 parameters. Cleanup training supports staged warm starts, resumable checkpoints, time limits, Ctrl-C-safe saves, punctuation reporting, and clean-input over-edit measurement. Real performance cannot be assessed until a full corpus is trained. If transcript punctuation is sparse, the run report directs the user to add punctuated text under `data/text/`.

## Phase 5 - dictation app

Implemented configurable hold-to-record hotkey, 300 ms tap rejection, energy trim, local checkpoint loading/warmup, personal spelling dictionary, guarded cleanup fallback to original ASR text, clipboard injection/restoration, per-stage timing output, and SQLite dictation logs. `run` exits clearly without a trained ASR checkpoint. Hardware permissions/audio device behavior still need a real Windows laptop check.

## Phase 6 - tests and docs

Tests cover fake parquet scan, text normalization, embedded WAV decode/resample, duration batching, dev-speaker exclusion, tokenizer freeze, small model forwards, cleanup guard fallback, VAD, clipboard restore, cleanup real-pair count, generic trainer resume equivalence, retention, ASR pool-order resume, greedy CTC decoding, schedule estimation, repeat CLI parsing, global-best retention, and tiny ASR overfit. Current test result: 23 passed with two PyArrow deprecation warnings.

The first full real-data ASR stage completed at 14,708 steps with 57,558,457 parameters, but initially had no dev score or best checkpoint. `eval asr` now falls back to the latest completed checkpoint and saves a recovered best model after successful evaluation. The first evaluation exposed a CTC batch-shape bug that skipped all 126 rows; it is fixed and covered by a regression test.

The user then successfully evaluated all 126 rows: WER 0.9989, CER 0.9919, with mostly blank outputs. Inspection of the run record showed that the learning-rate schedule ended at 0.0004008 from a 0.0005 peak because its step estimate did not divide by gradient accumulation. The estimate is fixed for future runs. Added `train asr --repeat-last-stage` to create a new stage over the previous stage's verified shards, initialized from the latest weights with a fresh optimizer/lower LR; replay mixing is disabled on this repeat. A repeat can resume with `--repeat-last-stage --resume`.

`README.md` has been rewritten as the quickstart, workflow, and command reference. Added root `explanation.md` with the data path, algorithms, model details, stage/checkpoint behavior, limitations, and current real-data status.

After reviewing the real ASR metrics, found schedule estimation counted batches but did not divide by gradient accumulation. At step 14,708 the old run was still at LR 0.0004008 from a 0.0005 peak, so cosine decay was incomplete. Fixed the estimate and added `train asr --repeat-last-stage`, which checks recorded shard hashes and starts a separate stage from the latest weights with a fresh optimizer and lower peak LR. Repeated stages disable replay mixing and can be resumed with `--repeat-last-stage --resume`. The root best ASR checkpoint is only updated when dev WER improves. Latest suite: 23 passed, two PyArrow deprecation warnings. No second multi-hour training run has been started.

## Decisions and known limits

- Generated fixtures contain synthetic one-second audio and Hindi text for plumbing only; they cannot demonstrate ASR quality.
- Replay/dev audio stays inside the source parquet. Manifests contain references and transcript text only, following the project data-safety rules.
- Use the PyTorch models implemented here only. There are no pretrained weights or hosted model calls.
- The required actual 8 GB training, real hardware, and user-data smoke runs remain unverified. The bounded sampler is deterministic but can use extra host RAM while decoding a pool of up to 128 clips.
