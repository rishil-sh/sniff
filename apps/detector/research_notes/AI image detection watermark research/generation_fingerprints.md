# Generation architectures and forensic fingerprints of AI images

Scope note: built from ~15 search/fetch calls. Primary papers were sometimes only reachable as abstracts or search snippets. Items marked "unverified" should be checked before use in a report.

## Which generator families exist and what architectural traces each leaves

### Takeaway

Latent diffusion (SD), rectified-flow transformers (SD3, FLUX) and many commercial systems all decode through a learned autoencoder (VAE), so decoder/upsampling traces are the most shared fingerprint. Autoregressive models add a discrete-tokenizer trace. GAN spectral peaks from upsampling are the oldest and most architecture-specific. Evidence for the newest closed models (GPT-image, Gemini image, Midjourney internals) is thin.

### Cited Findings

- Latent diffusion models run denoising in the latent space of a pre-trained autoencoder; generated images are reconstructed by that AE more accurately than real images, which AEROBLADE (CVPR 2024) uses for training-free detection of Stable Diffusion and Midjourney images, scored with LPIPS — [AEROBLADE, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.html)
- SD3 pairs a rectified-flow objective with an MMDiT backbone, and its autoencoder moved from 4 to 16 latent channels; FLUX.1 [dev] is a 12B rectified-flow transformer with a 16-channel VAE — [SD3 / rectified flow notes](https://awesome.papernotes.org/en/era5_genai_explosion/2024_stable_diffusion3/), [Qubrid FLUX listing](https://qubrid.com/models/flux-dev) (secondary sources; SD3 original paper is arXiv 2403.03206)
- Qwen-Image is listed as a 20B MMDiT flow-matching model with a 16-channel VAE, but only from secondary sources — unverified (technical report not retrieved).
- OpenAI's GPT-4o image generation is described as autoregressive and natively embedded in ChatGPT (unlike DALL-E), per secondary reporting; the academic GPT-ImgEval paper notes that whether the decoder head is AR or diffusion is not settled, and one blog claims a diffusion decoder — [InfoQ](https://www.infoq.com/news/2025/04/gpt-4o-images), [GPT-ImgEval](https://arxiv.org/html/2504.02782v2). GPT-4o images carry C2PA tags (InfoQ).
- Visual AR models (VAR, LlamaGen, Infinity named in D3QE references) encode images into discrete VQ tokens. Due to exposure bias, AR sampling can produce token sequences the tokenizer never saw, which decode to structural artifacts — [D3QE, ICCV 2025](https://arxiv.org/pdf/2510.05891), [search summary of ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_D3QE_Learning_Discrete_Distribution_Discrepancy-aware_Quantization_Error_for_Autoregressive-Generated_Image_ICCV_2025_paper.pdf)
- D3QE: real data has a long-tail codebook usage, fakes show concentrated usage; it fuses VQVAE quantization-error features with CLIP semantic features, evaluated on ARForensics (7 visual AR models) — [D3QE abstract](https://arxiv.org/abs/2510.05891). Numbers not retrieved.
- GAN upsampling operations create spectral artifacts (Frank et al., ICML 2020), enough for a linear classifier to separate GAN images in frequency space; later work shows the discriminator also shapes spectral discrepancies — [Frequency Bias of Generative Models, NeurIPS 2021](https://ar5iv.labs.arxiv.org/html/2111.02447), [liner review of Frank et al.](https://liner.com/review/leveraging-frequency-analysis-for-deep-fake-image-recognition)
- Diffusion models also show frequency bias (cannot reproduce all high frequencies) and the up-sampling layer is a key module — [Intriguing properties of synthetic images](https://arxiv.org/pdf/2304.06408)
- CVPR 2024 work argues the up-sampling operator creates generalized forgery artifacts beyond frequency peaks, captured by local Neighboring Pixel Relationships — [Tan et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.pdf)

### Inferences

- Because SD, SD3, FLUX and likely Qwen-Image share the "VAE decoder" stage, VAE-decoder traces are a family-level signal, but the 16-channel VAEs differ from the SD 4-channel AE, so an AEROBLADE-style check using SD AEs would likely miss them unless the matching AE is used (not tested in any source I found).
- Autoregressive/native-multimodal models whose pixels come from a diffusion or other detokenizer head would carry that head's traces, not a VQ trace; GPT-image's exact decoder is unverified.
- Midjourney is proprietary; AEROBLADE reports it detectable, implying an LDM-style AE, but no architecture source was found.

### Gaps

- No primary technical report found for Gemini image, Janus-Pro, Emu3, Midjourney, Qwen-Image architectures (searches returned nothing for these).
- No paper found specifically studying flow-matching-specific artifacts (search for "FLUX SD3 flow matching detection" returned only general detectors).

## Fingerprint types and detection methods

### Takeaway

Methods fall into: (a) spectral/upsampling cues, (b) reconstruction-error methods (DIRE, LaRE2, FIRE, AEROBLADE), (c) noise-residual/coding-cost statistics (ZED, Noiseprint-based), (d) foundation-model features (CLIP, DINOv2) with generator-agnostic training (UnivFD, B-Free, Community Forensics).

### Cited Findings

- DIRE (ICCV 2023) detects via the error between an image and its diffusion reconstruction; critiques: inversion unreliable, single-feature, costly, limited cross-model generalization — [FIRE paper discussion](https://arxiv.org/html/2412.07140v1)
- LaRE2 computes reconstruction error in latent space with single-step denoising (cheaper) plus error-guided feature refinement — [LaRE2](https://arxiv.org/pdf/2403.17465)
- FIRE: diffusion models struggle to reconstruct mid-band frequencies of real images; frequency-guided reconstruction error is claimed to generalize to unseen diffusion models and resist perturbations (venue conflicting across sources; check arXiv) — [FIRE](https://arxiv.org/html/2412.07140v3)
- ZED (ECCV 2024): training on real images only; scores how far an image's lossless-coding cost departs from what a coder trained on real images predicts; claims >3% average accuracy gain over prior art — [ZED](https://arxiv.org/abs/2409.15875). Aggregator figures show large variance by generator (84 Midjourney, 44 Wukong) — [wizwand](https://www.wizwand.com/paper/68fa74e61e209131cea6487c), secondary, unverified.
- Noise-residual clustering (2026): Noiseprint++ residuals + frozen ViT + K-Means, training-free, 82.2% average accuracy over four benchmarks; robustness not reported in abstract — [arXiv 2607.10695](https://arxiv.org/abs/2607.10695)
- UnivFD (Ojha, CVPR 2023): linear classifier on frozen CLIP ViT features to avoid learning generator-specific fingerprints — [summary](https://liner.com/review/towards-universal-fake-image-detectors-that-generalize-across-generative-models)
- Community Forensics (2024): 5.4M images (2.7M generated from thousands of HF diffusion models + real); performance improves with number of generator models even when architectures are similar — [Community Forensics](https://arxiv.org/pdf/2411.04125)
- B-Free (CVPR 2025): DINOv2+reg ViT fine-tuned on self-conditioned SD reconstructions of real images (content-aligned pairs) plus inpainting/augmentation, large crops, no resizing; reports bAcc Midjourney 99.6, SDXL 99.8, DALL-E 3 98.2, Firefly 99.7, FLUX 92.3, SD3.5 98.9, +20.7% bAcc over second best — [B-Free](https://arxiv.org/html/2412.17671v2)
- Raising the Bar (Cozzolino, CVPRW 2024): CLIP-based detector with few training samples, tested incl. DALL-E 3 and Midjourney; CLIP-based detectors largely insensitive to compression and resizing, while Ojha lost up to ~10 AUC under compression — [paper](https://arxiv.org/pdf/2312.00195), [supplement](https://openaccess.thecvf.com/content/CVPR2024W/WMF/supplemental/Cozzolino_Raising_the_Bar_CVPRW_2024_supplemental.pdf) (via search snippet)
- MiraGe (2025): trained only on SD v1.4, reports >93% accuracy on FLUX.1-dev and SD3.5 — [MiraGe](https://arxiv.org/pdf/2508.01525) (abstract-level claim)
- AIDE (ICLR 2025): combines artifact and noise-pattern experts; paper's Chameleon set of human-fooling images makes 9 detectors misclassify almost all fakes as real — [Sanity Check](https://arxiv.org/html/2406.19435v3)

### Inferences

- Generator-agnostic features (CLIP/DINOv2) plus content-aligned training data are the strongest-evidenced route to cross-architecture generalization; pure spectral cues are the weakest.
- A practical detector should combine a foundation-feature classifier with reconstruction/coding-cost cues and not rely on any single fingerprint.

### Gaps

- No unified benchmark found comparing AEROBLADE/ZED/FIRE on FLUX, SD3, and AR models.
- DIRE's own limitations section and AEROBLADE robustness numbers not retrieved (CVPR page returned 403).

## Robustness: what survives JPEG, resizing, social media

### Takeaway

Low-level fingerprints (high-frequency spectra, subtle residuals) are fragile to JPEG/resizing; semantic foundation-model features and bias-free training survive better. Real-world (in-the-wild) data causes the largest drops.

### Cited Findings

- AIGIBench: across 11 detectors, accuracy high in controlled settings drops significantly on real-world data (23 fake subsets; social media and AI-art platform samples); most detectors suffer catastrophic drops under JPEG or Gaussian noise; common augmentations give limited benefit — [AIGIBench](https://arxiv.org/abs/2505.12335), snippet at search result for same paper
- B-Free: viral-image test, most competitors fall below 70% bAcc after one day, B-Free stays ~92% over 100 days; simulated upload (resize 0.7-1, JPEG 70-100) leaves B-Free at 86.3 to 86.4 while DMID drops 96.1 to 81.3; authors say resizing can wash out low-level clues — [B-Free](https://arxiv.org/html/2412.17671v2)
- Format bias: fakes often saved as PNG, reals as JPEG, so detectors can learn compression instead of forensics; content bias flips verdicts depending on the real-image source — [B-Free](https://arxiv.org/html/2412.17671v2)
- Corvi detector barely affected by JPEG but lost accuracy under strong resizing; Wang et al. near chance on DALL-E 3 / Midjourney even before degradation — [Cozzolino supplement](https://openaccess.thecvf.com/content/CVPR2024W/WMF/supplemental/Cozzolino_Raising_the_Bar_CVPRW_2024_supplemental.pdf) (snippet)
- TextRich (2026): 12,095 GPT-Image-2 text-rich images (posters, charts, receipts, tables, UI); five zero-shot detectors vary strongly by domain; the strongest is ineffective on structured categories and highly sensitive to JPEG — [TextRich](https://arxiv.org/abs/2606.19259)
- SIDBench provides a framework to study transformations such as JPEG on detectors — [SIDBench](https://arxiv.org/abs/2404.18552v1)
- FIRE and D3QE claim robustness to perturbations (abstract-level only; numbers not retrieved).

### Inferences

- Persistent: semantic/CLIP-DINO features, large-crop high-level statistics, and properly debiased training. Fragile: GAN-style spectral peaks, fine noise residuals, PNG-vs-JPEG cues, and reconstruction error at default settings (FIRE exists specifically to improve this).
- Evaluate any detector on a JPEG(70-100) + resize(0.5-1) + re-upload pipeline, with format-matched real/fake data.

### Gaps

- No benchmark found passing Midjourney/DALL-E/GPT-image images through real social-media pipelines; evidence is from simulated degradations. GlobalForge (2026) claims sharp drops through real channels, only abstract seen, unverified.

## Why detectors trained on older generators fail on new ones

### Takeaway

Detectors tend to learn generator- or dataset-specific shortcuts (upsampling spectra, format, content) rather than a general "synthetic" property; new architectures change the decoder and tokenizer, so shortcuts vanish while generation quality rises.

### Cited Findings

- Traditional classifiers learn low-level, family-specific fingerprints, motivating UnivFD — [UnivFD summary](https://liner.com/review/towards-universal-fake-image-detectors-that-generalize-across-generative-models)
- Dataset biases (format, content, LAION resampling) inflate benchmark scores — [B-Free](https://arxiv.org/html/2412.17671v2)
- AR-generated images hide artifacts in the discrete latent space, not pixels, which challenges pixel-based detectors — [D3QE](https://arxiv.org/pdf/2510.05891) (via search summary)
- Detectors fail on DALL-E 3 and in-the-wild content, misclassifying nearly all samples from some sources — [AIGIBench via search](https://arxiv.org/abs/2505.12335)
- Training on more generators improves generalization even with similar architectures — [Community Forensics](https://arxiv.org/pdf/2411.04125)
- Frequency-only artifacts may not suffice for generalization — [Tan et al. CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.pdf)
- Conflict: MiraGe and B-Free report high accuracy on FLUX/SD3.5 from SD-only or SD-reconstruction training, whereas AIGIBench/Chameleon report catastrophic failure on harder or in-the-wild sources; difference is probably test-set curation and degradation, not a contradiction.

### Inferences

- Failure drivers: (1) decoder/tokenizer change removes learned artifacts, (2) higher fidelity reduces residual traces, (3) dataset shortcuts, (4) post-processing, (5) in-the-wild distribution shift (text-heavy, edited, upscaled outputs).
- Detector design implication: keep a model-update pipeline with fresh outputs from new closed models (GPT-image, Gemini, Midjourney), use content-aligned real/fake pairs, report per-domain accuracy, and pair detection with provenance (C2PA, watermarks), since pixel forensics alone is unlikely to keep pace.

### Gaps

- No independent measurement of detectors on GPT-image-1, Nano Banana/Gemini image, or Seedream found (only vendor claims); no attribution (which-model) fingerprint papers retrieved in this pass.
