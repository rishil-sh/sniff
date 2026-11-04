# Robustness, attacks and recommended architecture for a multi-generator AI image detector

Method note: about 14 searches and fetches. Several search summaries were secondary, and I flag these. Search summaries are not full-paper reads. Items marked "unverified" should not be quoted as fact.

## 1. Watermark removal and spoofing, and SynthID/C2PA stripping in practice

### Takeaway
Invisible watermarks (pixel-level and semantic) are removable by regeneration or diffusion editing, and semantic watermarks can also be forged. They are useful as a positive signal, but their absence proves nothing. C2PA metadata is mostly stripped by platforms, so it only helps when the file arrives untouched.

### Cited Findings
- UnMarker (Kassis & Hengartner, IEEE S&P 2025) is a universal, black-box, data-free and query-free attack. It disrupts spectral amplitudes. It was tested against 7 schemes and cut the best detection rate on semantic watermarks to 43%. The authors conclude that defensive watermarking is not viable against deepfakes. These are the authors' own claims, with no independent replication found. [arXiv](https://arxiv.org/html/2405.08363v2); [Waterloo PDF](https://cs.uwaterloo.ca/~uhengart/publications/oakland25.pdf)
- WAVES (ICML 2024) is a standardized stress-test benchmark. It covers distortions plus diffusive, adversarial and embedding-based attacks. It reports TPR at 0.1% FPR together with image quality loss. Schemes built on public VAEs can have their watermarks removed with minimal manipulation. [PMLR](https://proceedings.mlr.press/v235/an24a.html); [arXiv](https://arxiv.org/html/2401.08573v3)
- Diffusion-based image editing breaks robust watermarks (arXiv 2510.05978, Oct 2025). Targets: HiDDeN, StegaStamp, TrustMark, VINE, and a spread-spectrum scheme. Attacks: JPEG-50, noise, FGSM, SD1.5 img2img regenerating 30-40% of content, and a guided white-box variant. Unguided diffusion cut exact-match detection to near zero for the learned schemes, at about 29-32 dB PSNR. SynthID is not tested. Caution: the fetched table did not fully agree with the paper's text, so use the qualitative conclusion only. [arXiv](https://arxiv.org/html/2510.05978v1)
- Forging: Müller et al. (CVPR 2025) show black-box forgery of semantic watermarks. "Imprint" pushes an arbitrary image's latent toward a watermarked image's latent using an unrelated model. "Reprompt" inverts a watermarked image and regenerates it with a new prompt. Only one reference watermarked image is needed. Reported: near-perfect detection and attribution for Gaussian Shading, and over 84% detection for Tree-Ring on most models. The same attacks can also remove the watermark. [CVPR](https://openaccess.thecvf.com/content/CVPR2025/html/Muller_Black-Box_Forgery_Attacks_on_Semantic_Watermarks_for_Diffusion_Models_CVPR_2025_paper.html)
- Plug-and-Plant (arXiv 2506.06018) is an optimization-free universal forgery using regenerative diffusion. [arXiv](https://arxiv.org/html/2506.06018v1)
- SynthID-Image (Google, arXiv 2510.09263) has watermarked more than 10 billion images and video frames. The verifier is limited to trusted testers. The paper lists regeneration attacks in its threat model and claims state-of-the-art robustness to common perturbations versus other post-hoc methods. The abstract gives no attack numbers. [arXiv](https://arxiv.org/abs/2510.09263)
- Community "SynthID-Bypass" claims that a low-strength diffusion re-render defeats SynthID. This is unverified and not independently confirmed by Google or academics. [SourcePulse listing](https://www.sourcepulse.org/projects/23071281)
- C2PA in practice: a reported Washington Post test uploaded one AI video with Content Credentials to 8 platforms. Seven stripped the credentials and one buried the disclosure. This is a secondary source, with platform names and method unverified. [blog report](https://www.top10films.co.uk/89040-content-authenticity-in-2025-where-c2pa-content-credentials-really-stand/)
- Platforms often read C2PA to apply their own label and then strip the file's metadata. Some platforms, such as TikTok, now label on read. This is also from non-primary sources. [makeinfluence](https://www.makeinfluence.com/en/academy/content-credentials-c2pa-how-platforms-label-ai-assisted-content)
- Adobe/CAI "durable credentials" pair metadata with an invisible watermark (TrustMark) and fingerprinting so the credential can be recovered after stripping. This is a vendor claim. [CAI](https://contentauthenticity.org/blog/three-pillars-of-provenance)
- Figures such as "23% of platforms strip C2PA, down from 61%" and "80% of images lose metadata" could not be traced to a primary source. Do not use them.

### Inferences
- A missing watermark or C2PA manifest is weak evidence, so the system must treat "no provenance" as "unknown", never as "real".
- A present, valid, signed C2PA manifest is strong positive evidence of origin. A detected generator watermark is strong evidence but forgeable for semantic schemes. If we ever claim attribution, require the watermark detector to be the vendor's own and treat third-party latent-style schemes as weaker.
- Regeneration at low strength is the cheapest attack. It also leaves a "diffusion-edited" trace that a passive detector may still catch. This is the main argument for layering passive over provenance.

### Gaps
- No peer-reviewed measurement of SynthID removal. No peer-reviewed per-platform C2PA survival table was found.
- I did not find the UnMarker result against the production watermarks of commercial generators.

## 2. Social-media pipelines (resize, JPEG, screenshots, WhatsApp/Instagram/X)

### Takeaway
Detectors trained on clean lab data degrade sharply after recompression and resizing, typically biasing toward "real". Training on in-the-wild data, augmentation and a modern backbone recover much of it. Metadata is mostly gone after upload.

### Cited Findings
- AIGIBench (NeurIPS 2025 D&B): 11 detectors, 23 generators, plus social-media data. Average performance drop exceeds 40%. Under JPEG, fake accuracy of nearly all detectors falls to about 0% while real accuracy stays about 100%, a severe bias toward "real". Standard augmentation gave limited benefit and preprocessing effects were nonlinear. [arXiv 2505.12335](https://arxiv.org/pdf/2505.12335); [NeurIPS](https://neurips.cc/virtual/2025/poster/121688)
- TrueFake (2025): mainstream detectors lose 32-57% average accuracy after social-network propagation. Secondary summary, not verified at paper level. [listing](https://newx.sg/paper/detail/5324f73e-f823-11f0-b84c-00163e10baa7)
- "AI-Generated Image Detection in the Wild: What Truly Matters?" (arXiv 2507.10236): the ITW-SM benchmark has 10k balanced images from Facebook, Instagram, LinkedIn and X. DINOv2-L/14 gave the best average AUC (94.90), and CLIP variants did worse. In-the-wild training data lifted DMID from 79.96 to 91.55 AUC on ITW-SM. Texture-based cropping beat center crop (DMID 78.78 to 89.46). Augmentation helped most models. The paper reports an average 26.87% AUC gain after its modifications. It does not study ensembling. [arXiv](https://arxiv.org/html/2507.10236v1)
- Reposting produces multiple copies of differing quality, so the verdict can differ by copy. QuAD (arXiv 2604.15027) retrieves near-duplicates to address this. [arXiv](https://arxiv.org/html/2604.15027v1)
- MIRROR stress-tests JPEG quality 100 to 40 and resize 0.5x to 2.0x as a social-media model. [arXiv 2602.02222](https://arxiv.org/pdf/2602.02222)
- A camera-metadata self-supervised detector (arXiv 2512.05651) claims robustness to benign post-processing. [arXiv](https://arxiv.org/pdf/2512.05651)
- Metadata: blogs say Instagram, X and Facebook strip EXIF, and WhatsApp strips GPS but may keep some camera info. Consumer-grade sources, verify empirically per platform. [dev.to](https://dev.to/hassann/why-ai-image-detection-fails-and-what-to-use-instead-4dp8)

### Inferences
- Detector outputs should be conditioned on estimated compression history (JPEG quality, double-compression, resolution). A heavily recompressed, small image should move to abstain or widen its uncertainty rather than score "real".
- Evaluate with per-platform pipelines (upload and re-download through real platforms), not only synthetic JPEG sweeps.

### Gaps
- No controlled, published per-platform (WhatsApp vs Instagram vs X) quantification of detector TPR or metadata survival was found. Screenshots specifically were not covered by any source I retrieved.

## 3. Adversarial evasion, edited and inpainted images, and false positives on real photos

### Takeaway
Passive detectors can be evaded cheaply, including by transferable black-box attacks and prompt-level attacks. Partial edits and upscaling are blind spots. Real phone photos are a serious false-positive risk, and threshold choice dominates.

### Cited Findings
- RAID (arXiv 2506.03988): 72k adversarial examples crafted against an ensemble of 7 detectors using 4 text-to-image models. They transfer well, and ensemble-crafted attacks are comparable to white-box. Authors conclude SOTA detectors are easily fooled. [arXiv](https://arxiv.org/html/2506.03988v3)
- "Take Fake as Real" (R2BA, arXiv 2412.06727): black-box evasion using blur, JPEG, noise and light spots. [arXiv](https://arxiv.org/html/2412.06727v2)
- "Fooling the Watchers" (arXiv 2505.23192): prompt-level attack that makes the generator itself produce detector-evading images. Reported first place in a Tencent 2025 competition. [arXiv](https://arxiv.org/html/2505.23192v1)
- BFGSM (IEEE, Dec 2024): FGSM adapted to binary detectors. Secondary summary only.
- Partial edits: Stay-Positive (arXiv 2502.07778) finds that Corvi and Rajan detectors improve as more of the image is inpainted. This shows that real-image features hurt them on partially inpainted images. The authors' method targets this. [arXiv](https://arxiv.org/pdf/2502.07778)
- Copyleaks notes that inpainting regenerates nearby texture, so localization spreads beyond the edited element. Vendor statement. [Copyleaks](https://copyleaks.com/ai-image-detector)
- LAION-Mobile (arXiv 2609.11134, ~1M smartphone photos): thresholds set on GAN-era data gave under 11% FPR, but refit on modern content several detectors flagged 17-91% of real photos. No detector was both above chance on modern AI content and at usable FPR. The dataset only covers 2018-2020 phones, so current flagships are an open gap. I read the abstract only. A search snippet quoted "0.2% vs 39.1%", which the abstract fetch did not confirm. [arXiv](https://arxiv.org/abs/2609.11134)
- NewsGuard audit (May 2026, as reported): authentic news photos flagged as AI 13.3% on average across 5 tools, worst 40%. Secondhand and unverified. [PhotoWorkout](https://www.photoworkout.com/?p=222434)
- Upscaling (Real-ESRGAN and similar) is claimed to add synthetic-looking patterns or mask artifacts. This comes from blogs only, with no rigorous study found. [Fast.io](https://www.fast.io/resources/ai-image-detector-tools-2026.md)

### Inferences
- Treat any single learned detector as attackable. Fusing diverse detectors does not stop transfer attacks crafted against an ensemble (RAID), so keep at least one non-differentiable or independent signal (watermark, metadata, retrieval/near-duplicate).
- Add a "processing class" side-head (AI-upscaled, heavily edited, mobile computational photo, inpainted) and route these to abstain or "edited" labels rather than "AI-generated".
- Report FPR on a held-out real-photo set that includes modern phones with HDR and night mode, since existing public sets underrepresent them.

### Gaps
- No rigorous measurement of AI-upscaled real photos or modern-phone HDR FPR. No primary data on localized inpainting detection rates for current editors.

## 4. Calibration, abstention, low-FPR thresholds, fusion, evaluation

### Takeaway
Operate at a fixed low FPR set on in-domain real data, recalibrate per domain, and abstain when inputs look out of distribution. Published evidence on abstention for AI-image detection is thin.

### Cited Findings
- "Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated" (arXiv 2602.01973): fake logits shift globally under distribution shift, so the default threshold is wrong. A single learnable logit bias, fit on a few target samples with a frozen backbone, gives gains such as +16.16% for RINE on GenImage. [arXiv](https://arxiv.org/html/2602.01973v1)
- "Finding AI-Generated Faces in the Wild": a threshold at 0.5% FPR gives about 98% TPR in-distribution, dropping to 84.5% on an unseen synthesis engine at the same FPR. [arXiv](https://arxiv.org/pdf/2311.08577)
- WAVES uses TPR at 0.1% FPR as its headline metric. [PMLR](https://proceedings.mlr.press/v235/an24a.html)
- LAION-Mobile shows FPR swinging from under 11% to 17-91% purely from the threshold calibration set. [arXiv](https://arxiv.org/abs/2609.11134)
- Open-set deepfake work (contrastive, "unknown" class; arXiv 2503.08055) exists. Standard conformal prediction rarely abstains on unseen classes, so Fern and Dietterich suggest anomaly-detection nonconformity scores. No paper found applying conformal abstention to AI-image detection. [arXiv 2503.08055](https://arxiv.org/pdf/2503.08055)
- Ensembles: HEDGE (arXiv 2604.03555) is a heterogeneous ensemble for in-the-wild detection with JPEG and resize robustness metrics. I only saw the abstract-level snippet. [arXiv](https://arxiv.org/pdf/2604.03555)
- Continual updating: E3 uses expert embedders per generator plus a fusion network (CVPRW 2024). DECODE reports 99.36% average accuracy and 0.39% forgetting over 19 domains, and 95.36% on 11 unseen generators. Also arXiv 2601.05580 and LiteUpdate (2511.07192). All are authors' own numbers. [E3](https://arxiv.org/abs/2404.08814v1); [2601.05580](https://arxiv.org/pdf/2601.05580); [2511.07192](https://www.arxiv.org/pdf/2511.07192)

### Inferences
- Metrics to report: AUC is not enough. Report TPR at 0.1% and 1% FPR per generator and per degradation, plus FPR on real sets by source (camera, phone model, platform), and the abstention rate with coverage-vs-risk curves.
- Calibration: recalibrate with per-domain Platt or temperature scaling and a bias term fit on in-domain real images. Choose thresholds from real-image quantiles on matched processing, not from GAN-era or lab data.
- Fusion: stack calibrated scores in logit space (logistic regression or a small GBM) with quality/compression features as inputs. Learned scores near 0.5 should not vote, as the repo already does in fusion v0.3. Abstain when detectors disagree strongly or the input is OOD (tiny, palette, transparent, extreme recompression, which the repo already treats as out of domain).
- Use a rolling, dated evaluation set of newly released generators to detect drift.

### Gaps
- No standard published protocol for abstention in AIGI detection. TPR@1%FPR specifically was not found as a convention beyond WAVES' 0.1% and the 0.5% faces work.

## 5. Recommended layered architecture

### Takeaway
Provenance first (positive evidence only), then a diverse calibrated passive ensemble with quality-aware fusion, then a tri-state output (declared AI, likely real, inconclusive) with continual updating.

### Cited Findings
These are mostly inferred. Cited support appears in sections 1-4. The sources that most directly support the design: SynthID-Image for large-scale watermark deployment ([arXiv](https://arxiv.org/abs/2510.09263)), CAI's metadata plus watermark plus fingerprint "three pillars" ([CAI](https://contentauthenticity.org/blog/three-pillars-of-provenance)), AIGIBench for degradation failures ([arXiv](https://arxiv.org/pdf/2505.12335)), and the calibration paper ([arXiv](https://arxiv.org/html/2602.01973v1)).

### Inferences
1. Layer 0, input triage: estimate JPEG history, resolution, palette or transparency, screenshot and upscale hints. Record these as features and OOD flags.
2. Layer 1, provenance: validate the C2PA manifest (signature chain, trusted issuers, hard-binding hash). Run vendor watermark detectors where available (SynthID verifier access is limited, so this may not be possible). Valid "AI-generated" claims give a declared-AI verdict. A valid camera or capture claim lowers the prior but is not proof, because the content can be edited or spoofed. No manifest means no information.
3. Layer 2, passive ensemble: diverse detectors, such as a DINOv2/CLIP-feature detector, a low-level artifact detector, and a spectral-based one (SPAI-style). Use texture-aware multi-crop inference. Train on in-the-wild and recompressed data with augmentation.
4. Layer 3, calibrated fusion with abstention: logit-space stacking, per-domain bias recalibration, thresholds at fixed low FPR measured on matched real data, and tri-state output. Abstain on OOD, disagreement, heavy recompression, or detected upscaling and inpainting. Never output "real" as certain.
5. Layer 4, optional retrieval: near-duplicate or reverse-image lookup to find a less-degraded original (as in QuAD).
6. Continual updating: a monthly cadence for new generators. Keep a frozen held-out evaluation set and a rolling one. Use adapter-based updates with replay to avoid forgetting, and re-fit the fusion and calibration layers after every update.
7. Adversarial hygiene: keep some detector weights private, use randomized preprocessing or crops, and rate-limit queries. Red-team with RAID-style ensemble transfer attacks and low-strength regeneration.

### Gaps
- No published end-to-end evaluation of this exact layered stack. Cost and latency trade-offs were not researched.
