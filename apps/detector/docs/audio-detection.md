# Audio Detection (English Speech)

Status: implemented. Date: 2026-11-03. Code: `ai_detector/detectors/audio.py`. Policy: `ai_detector/fusion.yaml`.

## 1. Pipeline

1. **C2PA** is read from the original file (`img.c2pa`, shared with images).
2. **Decode** with ffmpeg to 16 kHz mono float. Phone-band audio (no energy above 4 kHz) is `out_of_domain`.
3. **AudioSeal** (`aud.audioseal`, watermark). It finds only Meta's AudioSeal mark. A hit makes the verdict `declared_ai`; no hit says nothing.
4. **Learned detectors** score 4 s windows with a 2 s hop. The clip score is the median window logit. The per-window `timeline` is kept, and the UI lists windows with p(AI) ≥ 0.8.
5. **Transcript** (Muse Voice Transcribe), only with consent. It is report context and is never scored.
6. **Report and UI.** `muse/client.py` turns the evidence into plain-language reasons: the deciding score, which seconds sound synthetic, the watermark and C2PA results, and the context scores labelled as unused. The Audio tab analyses an upload or a microphone recording as soon as it finishes. It shows the card and a per-window timeline. Warm latency is about 5 s for a 10 s clip on an M1 Max; the first run loads about 5 GB of weights.

| Evidence id | Model | Weight | Role |
| --- | --- | --- | --- |
| `aud.w2vbert` | Frozen w2v-BERT 2.0 (Meta, MIT) + our linear probe `en-L18-C0.01-cal1` (v6, recalibrated) | **1.0** | Decides, via the audio stacker's bands |
| `aud.dfarena` | DF Arena 500M (**non-commercial**) | 0 | Evidence only |
| `aud.mambo` | XLSR-MamBo (MIT) | 0 | Evidence only |

Weight 0 means the score is shown but cannot vote, and cannot force `inconclusive` through the disagreement rule. Without the probe file, `aud.w2vbert` is `skipped` and audio verdicts are `inconclusive`.

## 2. Why this setup

| Step | Finding | Decision |
| --- | --- | --- |
| Arena benchmarks | W2V2-AASIST (the original plan): 38.6% EER on modern TTS (CD-ADD) | Not used |
| Open TTS vs audiobooks | DF Arena near perfect; MamBo flagged 17% of humans (37% after MP3) | MamBo weight 0 |
| Real voices + Common Voice humans | DF Arena flagged **37% of Common Voice humans** and missed half of ElevenLabs v4 | Train our own probe |
| Probe v1 (ElevenLabs only) | 0% human FPR, but missed unseen generators (Piper 0%, Multilingual v2 0%) | Train on more families |
| Probe v2 (+ MMS, Piper, macOS voices) | 80/80 on unseen Gemini; 1/90 humans flagged | Shipped until v6 |
| Probe v3 (+ OpenAI gpt-4o-mini-tts) | Better on OpenAI docs samples (4/6 vs 1/6), worse on Gemini (63/80) and LibriSpeech FPR (6% vs 2%) | Not shipped; kept in `data/probe_train/` |
| Probe v4 (+ 40 API-generated ElevenLabs clips, vendor-level CV, multi-layer candidates) | Same layer as v3 (L6); LibriSpeech FPR 3/52 vs v2's 1/52; no gain on new ElevenLabs (both 48/48) | Not shipped: the pre-stated rule needed no worse human FPR |
| Probe v5 (selection at the shipped threshold, logit 0, FPR <= 3% per human source) | No candidate passed: every one flagged about 5% of held-out LibriSpeech speakers. Only 19 audiobook speakers in training | Data problem, not model choice |
| **Probe v6** (+ 300 LibriSpeech clips from 77 more speakers, same rule) | Out-of-fold LibriSpeech FPR 5% to 1.8%; picks layer 18. Test: **0/90 humans flagged**, AUROC 1.000, EER 1.0%; Gemini 69/80 at logit 0 | **Shipped** (user's call: zero human false alarms over Gemini recall) |
| **Learned mixture** (2026-11-04): logistic stacker over probe, DF Arena and MamBo, leave-one-vendor-out on 610 training clips | Worst-vendor recall at 1% human FPR: probe 0.89, probe+DF Arena 0.88, +MamBo 0.86, all three 0.83, DF Arena+MamBo 0.16; MamBo's weight ~0. The probe's old cut-off was too cautious | **Shipped:** probe alone, recalibrated (`logit*1.3455+3.2610` folded into the weights; bands 0.4193 / 0.8954 in `fusion.yaml` `stackers.audio`) |

## 3. Training the probe

`scripts/train_audio_probe.py`, features from `audio.embed_w2vbert` (the same function the detector uses).

| Class | Source | Clips | Licence |
| --- | --- | --- | --- |
| AI | ElevenLabs v4, LAION _train_ split, 24 voices | 406 | CC BY 4.0 |
| AI | MMS-TTS, 5 Piper voices, 10 macOS voices, LibriSpeech train texts | 240 | see NOTICE |
| AI (v3+) | gpt-4o-mini-tts, 6 emotions, prompts disjoint from test | 420 | MIT |
| AI (v4+) | ElevenLabs API, flash_v2_5 + v4_turbo, LibriSpeech train texts (`scripts/generate_elevenlabs.py`) | 40 | our generations |
| Human | Common Voice 17 English _train_, one clip per speaker | 400 | CC0 |
| Human | LibriSpeech train-clean-100, shard 0 | 250 | CC BY 4.0 |
| Human (v6) | LibriSpeech train-clean-100 shards 4/8/12 + train-other-500 shard 0, 77 more speakers (`--stage fetch-humans`) | 300 | CC BY 4.0 |

- Every clip is used twice: as published and after an MP3 round trip (24–128 kbps). The probe cannot separate classes by codec.
- Up to 3 windows per clip. Logistic regression with class balancing and standardised features.
- **Selection (v6):** leave one _vendor_ out. Humans are split by speaker. The rule is measured at logit 0, the shipped threshold: among candidates with out-of-fold FPR <= 3% on each human source, take the best worst-vendor recall. Candidates: all 24 single layers, layers 6+12+18 concatenated, and the mean of layers 1-24, each at C = 0.001, 0.01 and 0.1. Earlier versions selected at a 2% pooled-human threshold that did not match the shipped one. That is why v3 and v4 picked layer 6.
- An out-of-fold Platt fit was tried and dropped: it moved the threshold until 17% of LibriSpeech humans were flagged.
- Shipped v6 cross-validation (layer 18, C=0.01), at logit 0:
  - human FPR: Common Voice 0.00, LibriSpeech 0.02;
  - held-out recall: ElevenLabs 0.79, OpenAI 1.00, MMS 0.79, Piper 0.75, macOS 0.82.

## 4. Test results

All test clips are disjoint from training: different splits, speakers, prompts, or voices. The cells give clips flagged at p(AI) ≥ 0.5, out of the group size.

| Test group | n | DF Arena | MamBo | Probe v1 | Probe v2 | Probe v3 | Probe v4 | **Probe v6 (shipped)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Common Voice humans (home mics) | 38 | 14 ❌ | 22 ❌ | 0 | 0 | 0 | 0 | **0** |
| LibriSpeech humans | 52 | 1 | 9 | 0 | 1 | 3 | 3 | **0** |
| ElevenLabs v4 acting (same voices as training, new clips) | 26 | 13 | 22 | 26 | 26 | 26 | 26 | **26** |
| ElevenLabs premade previews | 21 | 20 | 21 | 20 | 21 | 21 | 21 | **21** |
| ElevenLabs Multilingual v2 (LAION set) | 24 | 8 | 24 | 0 | 24 | 24 | 24 | **24** |
| ElevenLabs API, generated 2026-11-03: v4, v4_turbo, flash_v2_5, turbo_v2_5 | 32 | 29 | 32 | – | 32 | – | 32 | **32** |
| ElevenLabs API: **eleven_v3, multilingual_v2 (never trained on)** | 16 | 14 | 16 | – | 16 | – | 16 | **16** |
| OpenAI gpt-4o-mini-tts, held-out prompts | 78 | 78 | 73 | – | 78 | 78 | 78 | **78** |
| OpenAI docs samples (tts-1) | 6 | 6 | 4 | 3 | 1 ❌ | 4 | 4 | **2** ❌ |
| **Gemini 2.5 Pro TTS (never trained on)** | 80 | 78 | 79 | – | 80 | 63 | 64 | **69** |
| **Kokoro (never trained on)** | 13 | 13 | 11 | 11 | 13 | 13 | 13 | **13** |
| Piper / MMS / macOS (test voices) | 39 | 39 | 34 | 15 | 39 | 39 | 39 | **39** |

The two Common Voice clips that the phone-band gate catches are excluded from the Common Voice row.

On all 393 real-voice clips, probe v6 scored AUROC 1.000 and EER 1.0%, with every AI clip ranked above every human clip. DF Arena scored 0.929 and 16.7%. Probe v2 on the first 189 clips scored 0.997 and 3.1%. The LibriSpeech clip `121-127105-0002` fooled every earlier detector; v6 scores it 0.17. The same sentence read by ElevenLabs turbo_v2_5 scores 0.997, and read by Piper 0.994.

The shipped probe-only verdicts (v6) on the same groups:

| Group                  | likely AI | inconclusive | no AI signal |
| ---------------------- | --------- | ------------ | ------------ |
| Common Voice humans    | 0         | 2            | 38           |
| LibriSpeech humans     | 0         | 0            | 52           |
| All ElevenLabs English | 119       | 0            | 0            |
| OpenAI gpt-4o-mini-tts | 78        | 0            | 0            |
| Gemini 2.5 Pro TTS     | 58        | 21           | 1            |
| OpenAI docs samples    | 0         | 3            | 3 ❌         |
| Open TTS               | 52        | 0            | 0            |

Compressed audio: the 32 kbps MP3 copies were measured for DF Arena and MamBo only (Section 2). The probe trains on MP3 round trips, but it has not been re-measured on `data/eval_audio_mp3`.

### Recalibrated probe (shipped), same 422 test clips, current fusion vs stacker

| Group | before: likely / inconclusive / no signal | after |
| --- | --- | --- |
| Common Voice humans (38) | 0 / 0 / 38 | 0 / 0 / 38 |
| LibriSpeech humans (52) | 0 / 0 / 52 | 0 / 4 / 48 |
| All ElevenLabs (119) | 118 / 1 / 0 | 119 / 0 / 0 |
| OpenAI gpt-4o-mini-tts (78) | 78 / 0 / 0 | 78 / 0 / 0 |
| OpenAI docs samples, tts-1 (6) | 0 / 3 / 3 | 3 / 3 / 0 |
| Gemini 2.5 Pro TTS (80) | 51 / 24 / 5 | 78 / 2 / 0 |
| Open TTS incl. Kokoro (52) | 52 / 0 / 0 | 52 / 0 / 0 |
| A user's ElevenLabs clone of their own voice (1, not in training or selection) | inconclusive | likely AI, 0.96 |

## 5. Limits (say these out loud in the demo)

- **English only.** No other language is calibrated. Non-English ElevenLabs clips score well, but there are no non-English human clips to measure false positives.
- **OpenAI's own docs samples (tts-1):** the shipped probe flags 2 of 6 and calls none "likely AI". gpt-4o-mini-tts is caught 78/78.
- **Gemini (unseen vendor):** 58/80 are "likely AI" and 21 "inconclusive". The scores rank them above all humans, so a lower threshold would catch more. Set that threshold on fresh data, not these test sets.
- **These test sets have been seen many times.** Each probe's selection used training-side CV only, but choosing between v2 and v6 used test results. Gemini is no longer a clean unseen vendor. The next measurement needs a vendor nobody has looked at (for example Cartesia, PlayHT or Azure).
- **Voice clones of real people:** only one clip tested (caught after recalibration). No clone is in training.
- **Not tested:** reverberation, speech over music, phone calls (gated as out of domain), voice conversion, cloning of a real person, partial edits.
- p(AI) is a class-balanced logistic output, not a calibrated probability.
- All numbers come from a Mac (MPS for w2v-BERT features). The Modal image does not install the audio models yet.

## 6. Reproduce

The commands are in the README, under "Speech (English)". Per-clip scores and summaries are in `data/eval_audio_results/` (git-ignored with the audio). Rebuild the shipped probe (v6): `train_audio_probe.py --stage fetch`, then `--stage fetch-humans`, then `--stage features`, then `--stage fit`. You also need the open-TTS, OpenAI and ElevenLabs API training clips in `data/probe_train/ai/`; see the README. Window sampling is random, so a refit can differ slightly from the shipped weights.

## 7. Next

1. Hold out a commercial vendor nobody has examined. Set the shipped threshold on it, and report Gemini / OpenAI recall there.
2. More human diversity: accents, podcasts and conversational speech. Each new human source becomes its own FPR constraint in selection.
3. Measure the probe on `data/eval_audio_mp3` and on reverberant and noisy copies.
4. Install the audio dependencies and weights in `modal_app.py` for the GPU demo.
