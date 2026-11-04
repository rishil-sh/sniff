# Improving Detection Accuracy (Deepfakes and AI-Morphed Images)

Status: plan, not implemented. Date: 2026-11-02.

## 1. Where we are

Image evidence today: C2PA, generator metadata, invisible watermark, and three learned detectors (`img.commfor`, `img.clip`, `img.spai`) fused by hand-set weights in `ai_detector/fusion.yaml`.

Findings from the lens icon test (AI clay render, no metadata):

| Image                       | commfor | clip | spai | Fused | Verdict      |
| --------------------------- | ------- | ---- | ---- | ----- | ------------ |
| Original art                | 0.02    | 0.83 | 0.81 | 0.50  | inconclusive |
| Icon (ring + border, 512px) | 0.035   | 0.80 | 0.00 | 0.23  | inconclusive |

- SPAI was crashing on every image (missing `clip` stub, checkpoint load rejected). Fixed in `spai_runner.py`.
- Commfor is confidently wrong on stylized art. Disagreement above `max_disagreement` (0.6) returns inconclusive.
- SPAI (spectral) loses its signal after resize or compositing.
- All learned scores are raw sigmoids (`platt: {}`), so they are not comparable across detectors.

Core limitation for the stated priority: all three detectors are trained for whole-image generation. Face swaps, face morphs and local AI edits are mostly real pixels, so they score as real.

## 2. Priority: deepfakes and AI-morphed images

Definitions used here:

- **Deepfake:** face swap, face reenactment, or synthetic face in a real scene.
- **Morph:** two identities blended into one face (ID fraud), or a face altered by AI.
- **AI edit:** inpainting or generative fill on part of a real photo.

### Step 1. Eval data first (nothing else can be validated without it)

Buckets, a few hundred images each to start:

| Bucket | Sources |
| --- | --- |
| Face swaps | FaceForensics++, Celeb-DF, DF40 |
| Face morphs | FRLL-Morphs and other public morph sets |
| AI edits | Self-made: Stable Diffusion inpainting on real photos |
| Fully generated | Existing eval set |
| Real | Matched real photos for every bucket, including stylized and JPEG-compressed |

Rules:

- Test cross-dataset: fit on one set, test on another. Forgery detectors collapse on unseen generators.
- Hold out a test split that is never used for fitting or tuning.
- Several datasets need a signed agreement; check access and license before relying on them.

Metric: TPR at 1% FPR and AUROC per bucket (`scripts/auroc.py`, `scripts/tpr_at_fpr.py`). Report per bucket, never one pooled number.

### Step 2. Face stage in the pipeline

- Detect faces (RetinaFace or insightface), crop each with a margin.
- Score each crop with the existing detectors, plus the full image.
- Per-image result is the max over faces, and the evidence records which face triggered it.
- This gives a cheap baseline and shows how much the current detectors already help.

### Step 3. Face-specific models

- **Deepfake detector:** one from the DeepfakeBench family (for example a CLIP-based one such as Effort or UCF). Pick by cross-dataset TPR at 1% FPR, not in-dataset accuracy.
- **Morph detector:** a single-image morph attack detector, since morphs have their own artifacts (blending seams, texture smoothing).
- Check each model's license; many are research-only.
- New evidence ids: `img.face_forgery`, `img.morph`.

### Step 4. Manipulation localization (edits and inpainting)

- Use a TruFor-style model that outputs a heatmap of edited regions.
- Whole-image detectors score a lightly edited photo as real because most of it is real; localization catches the edited part.
- Store the heatmap as evidence so the report can show where the edit is.
- New evidence id: `img.localize`.

## 3. General accuracy improvements

1. **Provenance first.** C2PA, IPTC, generator metadata and watermarks are near-perfect when present. Absence proves nothing, so they only decide when they fire.
2. **Calibrate per detector and per bucket.** Run `scripts/fit_platt.py` and write fits into `platt:` in `fusion.yaml`. Raw sigmoids are not comparable across detectors.
3. **Replace hand-set fusion with a small learned stacker.** Logistic regression over detector scores plus cheap features (size, alpha channel, JPEG-ness, palette size, face present). It learns when each detector is unreliable (for example commfor on stylized art) and removes the blunt `max_disagreement` rule that turns two-vs-one splits into non-answers.
4. **Test-time robustness.** Score the image as-is plus center crops and tiles. For composites (rings, borders, UI chrome), also score the inner region. SPAI went from 0.81 to 0.00 mostly from framing.
5. **Diverse detectors.** Keep at least one semantic (CLIP), one spectral (SPAI) and one broad-trained (commfor), plus the face and localization models above. New generators defeat old detectors, so re-evaluate every few months.
6. **Honest inconclusive band.** Use it only where calibrated detectors still disagree. Tune the thresholds for a fixed 1% FPR, since false accusations cost more than misses.
7. **Regression tests.** Keep a small fixed set of known cases (including the lens icon) and fail CI if their verdicts regress.

## 4. Implementation order

1. Build the bucketed eval set (Step 1).
2. Add the face crop stage and measure current detectors per bucket (Step 2).
3. Add one face-forgery model and one morph model; measure cross-dataset (Step 3).
4. Add localization (Step 4).
5. Calibrate with Platt, then train the stacker on the combined evidence.
6. Add regression tests and re-run the report.

Video deepfakes (temporal consistency, lip-sync) are out of scope until the image path meets its targets.

## 5. Risks and open questions

- Dataset access and licenses are unverified.
- Model licenses (research-only) may block commercial use.
- Face crops raise privacy handling concerns; do not send faces to external APIs without consent (the existing Muse consent flow applies).
- Accuracy targets are not set yet; agree on a TPR at 1% FPR target per bucket before starting.
