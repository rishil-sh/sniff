# AI content detector

Open-source, evidence-first verdicts for AI-generated **images, video, speech, and text**. Fusion returns **likely AI**, **inconclusive**, **no AI signal found**, or **declared AI** when provenance or a watermark says so. Muse Wonder explains the verdict in plain words; our code decides it.

**WhatsApp reply.** Every verdict's `report.answer` is a ready-to-send WhatsApp message (`*bold*`, `_italic_`, emoji), and `report.lean` is `ai`, `human` or `unclear`:

```
🤖 *Written by AI*

We found the writing follows the smooth, regular patterns of AI-made text from start to finish.

_Automated check: please don't use it as the only evidence against anyone._
```

Code writes the headline, so its confidence matches the measured verdict: on held-out text, 96.5% of `likely_ai` and 97.8% of `no_ai_signal` calls were right. Scores below 0.25 count as human: that edge was raised by hand from the fitted 0.032, which cut "can't say for sure" from 9.4% to 2.8% of texts while AI texts called human rose from 1.4% to 3.0%. With consent (`send_to_muse: true`), `muse-wonder-1.0` writes the one-line _why_ from the verdict JSON, minus GLTR's token list, so the text itself isn't sent; otherwise a template line is used. If Muse contradicts a decided verdict, its line is replaced. An _inconclusive_ verdict (scores 0.25–0.62) gets a "Best guess" headline, never a confident one. Too-short or non-English input gets a "Couldn't check this" message without calling Muse. With `reasoning_effort=low`, Wonder answered in 4–13 s; `AI_DETECTOR_REPORT_MODEL=muse-spark-1.3` is faster and cheaper.

Images run **C2PA/IPTC**, **generator metadata** (AUTOMATIC1111/ComfyUI/InvokeAI/NovelAI tags, Software/CreatorTool), the **Stable Diffusion/SDXL invisible watermark**, **Community Forensics**, and the **Cozzolino CLIP** linear probe locally. Text is decided by two supervised classifiers (**Fakespot RoBERTa**, **Desklib DeBERTa-v3**); **Binoculars**, **Fast-DetectGPT** and **GLTR** token colouring on a Llama base/Instruct pair are shown as evidence. No Meta Model API key is required for a verdict. The first analysis downloads model weights into `data/models/` (git-ignored; override with `AI_DETECTOR_CACHE`) (Community Forensics ~87 MB, CLIP backbone ~1 GB, text classifiers ~2 GB, audio models ~3 GB, Llama-3.2-3B pair ~13 GB). `uv run ai-detector-download` fetches them all up front; `AI_DETECTOR_TEXT_PAIR=0` skips the Llama pair.

Built for the Meta Global AI Developer Hackathon (Indie track, "Build for Your World").

## 1. Accounts (all optional)

The detector runs without any keys.

- **Meta Model API**, [dev.meta.ai](https://dev.meta.ai): only for Muse explanations and transcription when called with
  `send_to_muse: true`. Put the key in `.env` as `MODEL_API_KEY`. (In Sniff, the server's Mastra agent writes replies
  instead.)
- **Hugging Face**, [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens): only for gated models,
  e.g. the original [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) pair. Put it in `.env`
  as `HF_TOKEN`. The default models are all ungated.
- **Modal**, [modal.com](https://modal.com): only for `modal_app.py` deployments (`uv run modal setup`).

Copy `.env.example` to `.env` and fill the values. Do not commit `.env`.

## 2. Local setup

Requires **Python 3.11** and **uv**. ffmpeg is bundled through `imageio-ffmpeg`. From the monorepo root, `pnpm bootstrap` does all of this; standalone:

```bash
cp .env.example .env            # optional settings
uv sync --extra dev
uv run pytest
uv run ai-detector-download     # optional: fetch every model now
uv run ai-detector              # API on http://127.0.0.1:7860
```

Optional third image detector, **SPAI** (CVPR 2025, Apache-2.0, any resolution). It installs into its own venv under `data/models/spai` (~1 GB weights) and is reported as `skipped` until then:

```bash
./scripts/setup_spai.sh
```

**Text detector.** Two supervised classifiers decide: `fakespot-ai/roberta-base-ai-text-detection-v1` (Apache-2.0) and `desklib/ai-text-detector-v1.01` (MIT), combined by a logistic stacker fit on held-out data (see below and the measured numbers in `ai_detector/fusion.yaml`). If the stacker can't run, they vote with equal weight and disagreement means _inconclusive_. The zero-shot pair (Binoculars + Fast-DetectGPT, run locally because the Meta Model API returns no `logprobs`) is reported but has weight 0. On 2026-11-03 it flagged 0 of 5 modern chat-model essays that both classifiers caught (details in `ai_detector/fusion.yaml`).

The default pair is the ungated bf16 mirror of Llama-3.2-3B (base + Instruct), ~13 GB of RAM in bf16. The original 8B pair needs ~32 GB and gated access (`HF_TOKEN`):

```bash
AI_DETECTOR_TEXT_OBSERVER=meta-llama/Llama-3.1-8B
AI_DETECTOR_TEXT_PERFORMER=meta-llama/Llama-3.1-8B-Instruct
```

`AI_DETECTOR_TEXT_PAIR=0` skips the pair entirely; it has weight 0, so verdicts don't change.

Before any detector runs, the router undoes character attacks: it strips invisible format characters (zero-width, bidi), applies NFKC, maps Cyrillic/Greek look-alikes inside Latin words, and collapses whitespace. Any hidden or look-alike characters are reported as `txt.tamper` evidence, which never votes. Fakespot also gets its own `clean_text` preprocessing. Both classifiers read the whole text in overlapping windows (stride = half a window) and score the mean window logit; per-window logits are in `details.window_logits`. The zero-shot pair scores up to 8 × 512-token windows. Text that doesn't look like English is out of domain.

When `fusion.yaml` has a `stackers: text:` block, a logistic stacker over the raw detector scores replaces the weighted mean and disagreement veto. Its bands are set at a fixed false-positive rate on held-out human text. It runs only when every input is present and from the model version it was fit on; otherwise fusion falls back to the weighted rule.

Measure the detectors and fit Platt calibration on your own labelled images or `.txt` files (`DIR/ai/…`, `DIR/real/…`):

```bash
uv run python scripts/eval_images.py data/eval --csv eval.csv            # add --no-domain-gate to test the palette/transparency/size gate
```

For text, build the eval set first (PAN'25 is research-only and git-ignored; never publish it), then score it once and refit from the CSV as often as you like. Every number the script prints comes from a held-out half; Platt, the stacker and its bands are fit on the other half:

```bash
uv run python scripts/fetch_text_eval.py                                  # data/eval_text: 1,244 human, 839 AI texts
uv run python scripts/eval_images.py data/eval_text --csv data/eval_text/scores-3b.csv
uv run python scripts/eval_images.py --from-csv data/eval_text/scores-3b.csv --stack txt.fakespot,txt.desklib,txt.binoculars,txt.fastdetect
```

### Speech (English)

Open the **Audio** tab, then upload a clip or record from the microphone. It is analysed as soon as the upload or recording finishes. The card shows:

- the verdict, with plain-language reasons and caveats;
- the deciding score and which seconds sound synthetic;
- a per-window timeline chart.

The first audio analysis downloads and loads about 5 GB of weights. After that, a 10 s clip takes about 5 s on an M1 Max.

Audio runs C2PA, the **AudioSeal** watermark check, and three learned detectors over 4 s windows.

- Only the **w2v-BERT 2.0 probe** (`aud.w2vbert`, `ai_detector/weights/w2vbert_probe.npz`) votes.
- **DF Arena 500M** and **XLSR-MamBo** are shown for context with weight 0.

Results and limits are in [docs/audio-detection.md](docs/audio-detection.md).

ffmpeg comes bundled through `imageio-ffmpeg`. A system `ffmpeg`, or one named by `FFMPEG`, is used first when present.

The probe's training and eval scripts stay on the `audio-detection` branch; main ships only the runtime and the 14 KB probe weights.

API server only; the UI is sniff's web app (`apps/web`, http://localhost:3001).  
Health: http://127.0.0.1:7860/health  
JSON API: `POST /v1/analyze` (multipart file) or `POST /v1/analyze/text`

```bash
curl -s http://127.0.0.1:7860/v1/analyze/text \
  -H 'content-type: application/json' \
  -d '{"text":"'"$(python3 -c 'print(" token"*200)')"'"}'
```

## 3. Deploy

Docker: `docker compose up` from the monorepo root runs this detector with the rest of Sniff (models download into a
volume, not the image). Modal (CPU-only API, no ML models yet): `uv run modal deploy modal_app.py`.

Set `AI_DETECTOR_SKIP_ML=1` to skip the learned image models (tests do this via mocks).

## Verdict JSON

```json
{
  "request_id": "9f2c0001",
  "modality": "image",
  "verdict": "inconclusive",
  "p_ai": null,
  "decided_by": "fusion",
  "evidence": [],
  "report": { "model": "stub", "summary": "…", "caveats": [] }
}
```

Bands: `declared_ai | declared_capture | likely_ai | inconclusive | no_ai_signal`.  
Limits: images 20 MB, mp4 ≤ 60 s, speech ≤ 10 min / 32 MB, English text ≥ 150 tokens.

## License

MIT. Third-party notices live in [NOTICE](NOTICE). The audio detector DF Arena 500M is **non-commercial** and is downloaded at runtime; drop `aud.dfarena` for commercial use. Llama, when enabled: **Built with Llama**.
