# AI-generated image detection: repos, models, datasets, benchmarks, commercial detectors, provenance tools (status as of 2026-11-02)

Method note: GitHub stars, licenses and push dates come from the GitHub API (gh) on 2026-11-02. HF license and last-modified data come from the HF API on the same date. Several READMEs were read via fetch and were truncated or silent on some fields. Those fields are listed under Gaps rather than guessed.

## 1. Open-source detector repos and pretrained models (2024-2026)

### Takeaway
Permissive, usable-today options are SPAI (Apache-2.0, weights released), GRIP-UNINA's CLIP-based detector (Apache-2.0), UniversalFakeDetect (MIT) and AIDE (MIT code). The strongest recent academic detectors have non-commercial or unclear terms (B-Free, AIGIBench checkpoints, Community Forensics eval data). No source I found shows any open detector being validated on GPT-image, Gemini/Nano Banana or Qwen-Image outputs.

### Cited Findings
- SPAI (CVPR 2025), spectral masked-learning, flags AI images as out-of-distribution from real-image spectral statistics, handles any resolution. Apache-2.0, README says code and weights are Apache-2. Checkpoint is on Google Drive. Inference needs under 8 GB GPU. Stars 91, last push 2025-08-01. README reports no numbers. — [GitHub](https://github.com/mever-team/spai)
- GRIP-UNINA ClipBased-SyntheticImageDetection ("Raising the Bar of AI-generated Image Detection with CLIP"). Apache-2.0, 92 stars, last push 2024-11-24. Weights fetched through git lfs; the weights' own license was not stated in the sources. — [GitHub](https://github.com/grip-unina/ClipBased-SyntheticImageDetection)
- GRIP-UNINA DMimageDetection (latent diffusion detector). Apache-2.0, 241 stars, last push 2024-05-30. SPAI reuses its training/validation data. — [GitHub](https://github.com/grip-unina/DMimageDetection)
- B-Free (CVPR 2025). ViT with 4 registers, DINOv2-pretrained, trained on SD2.1 self-conditioned reconstructions plus inpainting; about 51.5K real COCO plus 309K SD2.1 fakes. License is custom GRIP-UNINA, "informational and nonprofit purposes" only (GitHub shows NOASSERTION). 68 stars, last push 2026-01-14. Authors released an extended Synthbuster with 1K FLUX and 1K SD3.5 images. — [GitHub](https://github.com/grip-unina/B-Free)
- AIDE (ICLR 2025, accepted 2025-01-23), hybrid features/experts. MIT license, 344 stars, last push 2025-06-04. Checkpoints on a Google Drive "Model Zoo". Its Chameleon dataset is not downloadable directly (email request) and is academic-only, no commercial use. — [GitHub](https://github.com/shilinyan99/AIDE)
- UniversalFakeDetect (CVPR 2023, CLIP ViT-L/14 linear probe). MIT, 379 stars, last push 2026-08-19. The repo is older, but still maintained. — [GitHub](https://github.com/WisconsinAIVision/UniversalFakeDetect)
- Effort-AIGI-Detection (ICML 2025 Oral, orthogonal subspace decomposition). No license declared, 240 stars, last push 2025-07-14. — [GitHub](https://github.com/YZY-stack/Effort-AIGI-Detection)
- NPR-DeepfakeDetection (CVPR 2024). No license declared, 316 stars, last push 2025-05-01. — [GitHub](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
- DRCT (ICML 2024). No license declared, 168 stars, last push 2025-04-05. — [GitHub](https://github.com/beibuwandeluori/DRCT)
- DIRE: archived, 399 stars, last push 2024-09-26. Do not use for new work. — [GitHub](https://github.com/ZhendongWang6/DIRE)
- Community Forensics classifiers on HF: `OwensLab/commfor-model-384` and `commfor-model-224`, trained on the 4,803-generator dataset (CVPR 2025). Code repo JeongsooP/Community-Forensics is MIT, 62 stars, pushed 2026-01-06. — [HF dataset card](https://huggingface.co/datasets/OwensLab/CommunityForensics); [GitHub](https://github.com/JeongsooP/Community-Forensics)
- AIGCDetectBenchmark (Ekko-zn): a test harness for many detectors. No license declared, 436 stars, last push 2026-03-09. — [GitHub](https://github.com/Ekko-zn/AIGCDetectBenchmark)
- DeepfakeBench (SCLBD): 1,134 stars, last push 2025-08-20, license NOASSERTION. It is mainly face-deepfake oriented (my inference from the name). — [GitHub](https://github.com/SCLBD/DeepfakeBench)
- AIGIBench retrained-detector checkpoints (Setting-II, ProGAN+SD1.4 training) on HF `HorizonTEL/AIGIBench`; repo license file not identified (NOASSERTION), data cc-by-nc-sa-4.0. 60 stars, pushed 2026-10-28. Best retrained results: DGS-Net Acc 82.3 / AP 87.8; Effort AP 87.2; FerretNet Acc 79.4; SAFE Acc 78.1. CNNDetection real acc 98.2 but fake acc only 11.6. — [GitHub](https://github.com/HorizonTEL/AIGIBench)
- Popular HF off-the-shelf classifiers (ViT/SigLIP fine-tunes on small datasets, mostly pre-FLUX era). By downloads: `umm-maybe/AI-image-detector` 68.5K, CC-BY-4.0; `Ateeqq/ai-vs-human-image-detector` 12K, Apache-2.0; `abacusai/Smaug-AI-Image-Detector` 7.7K, Apache-2.0; `haywoodsloan/ai-image-detector-deploy` 6.9K, Apache-2.0; `dima806/ai_vs_real_image_detection` 6.5K, Apache-2.0; `Bombek1/ai-image-detector-siglip-dinov2` 820, MIT; `Smogy/SMOGY-Ai-images-detector` 2.5K, **CC-BY-NC-4.0**. — [HF API listing](https://huggingface.co/api/models?search=ai%20image%20detector&sort=downloads&direction=-1&limit=12)
- 2026 AI-GenBench fine-tuned DINOv3 checkpoint exists (license "other", 0 stars). — [GitHub](https://github.com/SerafinoPandolfini/AI-GenBench-DINOv3-Checkpoint)

### Inferences
- For a permissive production stack, SPAI + GRIP-UNINA CLIP detector (both Apache-2.0) are the cleanest, but both were trained on older generators (latent-diffusion era). Re-validate on FLUX and later.
- HF community classifiers are trained on small datasets, and generalization is unknown. Do not trust them for new generators without testing.
- "No license declared" on GitHub means all rights reserved by default. Treat as non-usable commercially until clarified.

### Gaps
- Reported cross-benchmark numbers for SPAI, GRIP CLIP and Community Forensics were not retrieved (READMEs lack them; papers not read).
- No open detector's accuracy on GPT-image, Nano Banana, Qwen-Image or Midjourney v7 found.
- Weights license for GRIP CLIP detector and AIDE not verified.

## 2. Datasets and benchmarks

### Takeaway
Newest-generator coverage is best in OpenFake (FLUX.1/FLUX.2, Imagen 4, HiDream; CC-BY-NC) and AIGIBench (FLUX1-dev, Midjourney-V6; CC-BY-NC-SA). Community Forensics is the largest and most diverse (4,803 generators). No dataset surveyed was confirmed to include GPT-image, Gemini/Nano Banana or Qwen-Image. Most carry non-commercial or research-only terms.

### Cited Findings
- OpenFake (arXiv 2509.09495): about 4M images, roughly 3M real with captions and nearly 1M synthetic. HF `ComplexDataLab/OpenFake`: core subset 2.45M rows (train 2.3M / val 58.5K / test 91.4K), plus a 36.1K reddit subset. Generators visible in rows: FLUX.1-dev/schnell, FLUX.2-dev, FLUX-1.1-pro, Imagen 4.0, SD1.5/2.1/3.5, SDXL finetunes, Pony, HiDream-I1, Mystic. Midjourney, GPT-image, Gemini and Qwen as generators not seen in the part of the card read. Real images from LAION and Pexels. **License cc-by-nc-4.0.** Updated 2026-09-29. Includes a crowdsourced adversarial platform to keep adding hard examples. — [HF](https://huggingface.co/datasets/ComplexDataLab/OpenFake); [arXiv](https://arxiv.org/abs/2509.09495)
- Community Forensics (CVPR 2025): about 2.76M images, about 1.08 TB, from 4,803 generators. Subsets: systematic 1.92M (HF latent diffusion models), manual 774K, commercial 14.9K. Real: LAION 40.3%, ImageNet 40.3%, CelebA 7.2%, COCO 4.4%, etc. Main set cc-by-4.0 "for research only"; generated images inherit source-model licenses (mostly CreativeML OpenRAIL-M). HF shows Eval and Small variants as cc-by-nc-sa-4.0. Specific commercial models not named on the card. — [HF](https://huggingface.co/datasets/OwensLab/CommunityForensics); [paper](https://arxiv.org/pdf/2411.04125)
- AIGIBench (NeurIPS 2025): 25 test subsets, 23 sources, includes FLUX1-dev, Midjourney-V6, SD-XL, SD3, DALL-E 3, Imagen-3; also CommunityAI (6,000) and SocialRF (3,000) crawled from the wild, face-swap sets. Training: 144K images per setting. HF data cc-by-nc-sa-4.0. — [GitHub](https://github.com/HorizonTEL/AIGIBench); [summary](https://en.papernotes.org/NeurIPS2025/image_generation/is_artificial_intelligence_generated_image_detection_a_solved_problem/)
- AI-GenBench (MI-BioLab, 2025 ongoing): temporal benchmark, union of many existing datasets, covering generators 2017-2024. Code BSD-3-Clause, dataset has no single license, per-source terms in `dataset_creation/README.md`. 37 stars, repo pushed 2026-02-02. A search summary says it includes FLUX and DALL-E 3; not confirmed on the README. — [GitHub](https://github.com/MI-BioLab/AI-GenBench); [search summary source](https://arxiv.org/pdf/2504.20865v1)
- GenImage (2023): over 1M real/fake pairs; Midjourney, SD, ADM, GLIDE, Wukong, VQDM, BigGAN. Predates FLUX and newer models. HF mirrors vary: cc-by-nc-sa-4.0 (nebula), apache-2.0 (ENSTA-U2IS mirror; verify against original). GitHub repos show NOASSERTION. — [GitHub](https://github.com/GenImage-Dataset/GenImage); [HF search](https://huggingface.co/api/datasets?search=GenImage)
- Chameleon (AIDE paper): 9 off-the-shelf detectors mostly classified its AI images as real. Request-only via email; academic use only. — [GitHub](https://github.com/shilinyan99/AIDE)
- WildRF (LaDeDa paper, arXiv 2406.09398): images from Reddit, X, Facebook. Protocol: train on Reddit, test on unseen Twitter and Facebook. A 2026 paper describes 2,150 real/2,150 AI from Reddit, 340/340 Twitter, 160/160 Facebook. LaDeDa reports 93.7% mAP. — [GitHub](https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld); [paper](https://arxiv.org/pdf/2406.09398)
- Synthbuster: repo qbammey/synthbuster, no license shown, last push 2025-09-14. Extended with FLUX/SD3.5 by B-Free authors. — [GitHub](https://github.com/qbammey/synthbuster)
- Semi-Truths (NeurIPS D&B track; partial-edit images): code MIT, HF data license tag "cc". Updated Dec 2024. — [GitHub](https://github.com/J-Kruk/SemiTruths); [HF](https://huggingface.co/datasets/semi-truths/Semi-Truths)
- ImagiNet: about 200K examples across photos, paintings, faces, uncategorized; open-source and proprietary generators; own synthetic and DALL-E 3 images CC0. — [arXiv](https://arxiv.org/pdf/2407.20020)
- VCT2 (AACL 2025): about 166K images (140K synthetic) from SD variants, DALL-E 3, Midjourney 6; 17 detectors score about 58% on COCO_AI and Twitter_AI subsets. — [search summary source](https://arxiv.org/pdf/2504.20865v1)
- RealHD (ACM MM 2025): over 730K images including SD 1.5 to 3.5 and FLUX. — same search summary above
- So-Fake-Set / So-Fake-OOD (social-media forgery with explanations): HF `saberzl/So-Fake-OOD` cc-by-nc-4.0. — [arXiv](https://arxiv.org/html/2505.18660v1)
- FakeClue (explainable detection data): Apache-2.0 on HF mirrors. — [HF](https://huggingface.co/datasets/lingcco/FakeClue)

### Inferences
- If you need GPT-image / Nano Banana / Qwen-Image / Midjourney v7 test data, you will likely have to generate it yourself or use OpenFake updates; none of the surveyed sets confirms these.
- Almost all datasets are non-commercial or research-only; use them for evaluation and be careful training a commercial model on them.

### Gaps
- FakeBench: no repo or dataset size found (HF hits were unrelated music sets).
- Exact generator lists for AI-GenBench, ImagiNet commercial subset and Community Forensics commercial subset not retrieved.
- Semi-Truths and Synthbuster sizes not retrieved.
- Whether OpenFake later versions added Midjourney/GPT-image: the card was truncated after 100K characters.

## 3. Commercial and API detectors and independent evaluations

### Takeaway
Vendor claims run 98-99%, independent tests land lower (roughly 75-90%) and degrade on newer generators, compression and perturbation. Hive has the best peer-reviewed record; Illuminarty is weaker; rigorous independent papers for AI or Not and Reality Defender were not found.

### Cited Findings
- "Organic or Diffused" (arXiv 2402.03214): Hive ranked first of five detectors, about 98% accuracy, 0% FPR on human art, 3.17% FNR; Illuminarty worse and rejected images over 3MB. — [arXiv](https://arxiv.org/pdf/2402.03214)
- ImageDetectBench (arXiv 2411.13553): on 200 images Hive accuracy 1.0 unperturbed, 0.88 with Gaussian noise; Illuminarty 0.8 and 0.57. Watermark-based detection was far less sensitive to distribution shift than passive detectors. Small sample. — [arXiv](https://arxiv.org/pdf/2411.13553)
- NewsGuard audit, May 2026 (45 images, via secondary table): Sightengine 0% real photos wrongly flagged, 33% of substantially altered images caught; Hive 0% and 73.33%. Another summary says detectors often called real content fake, which contradicts this. I did not open NewsGuard's own report. — [secondary roundup](https://checkthat.ai/answers/what-are-the-best-ai-image-detector-tools); [other summary](https://www.tecnoandroid.it/news/deepfake-e-ai-i-software-per-riconoscerli-continuano-a-sbagliare-1895219/)
- DetectArena (self-described independent, about 1,000 images, 11 services including Hive, Sightengine, AI or Not): some FPR over 20%; ensembles beat any single service. Blog-level evidence. — [DEV Community](https://dev.to/aidetectarena/i-checked-1000-ai-and-real-images-with-top-ai-image-detectors-you-will-be-surprised-2jd8)
- Sightengine: vendor claim 98.3% on 80K images; one aggregator reports 98% on Ideogram v3 vs 75% on Hunyuan Image 3.0. Low-reliability blog numbers. — [roundup](https://fast.io/resources/ai-image-detector-tools-2026/)
- AI or Not: about 84-88% in independent benchmarks per roundups vs about 98.9% vendor claim. Low-reliability aggregators. — [AIMultiple](https://aimultiple.com/ai-image-detector)
- CBC three-image test: AI or Not and Hive correct on all three, Illuminarty wrong on all. Anecdotal. — [CBC](https://amp.cbc.ca/news/canada/ai-detector-tools-test-1.7482913)

### Inferences
- Treat vendor accuracy as upper bounds. Plan on an ensemble plus a human or provenance fallback.

### Gaps
- No peer-reviewed independent evaluation of AI or Not or Reality Defender. Reality Defender numbers (about 90%) are from vendor-linked pages and are not cited here.
- No pricing or API terms gathered.
- Per-generator accuracy on GPT-image, Nano Banana, Midjourney v7 not found for any vendor.

## 4. Provenance, C2PA and watermark verification tools

### Takeaway
C2PA verification is well served by Adobe/CAI open-source tooling (Apache-2.0 or MIT). Invisible-watermark detection is vendor-gated: SynthID image detection runs only inside Google products, so you cannot verify it yourself. Open watermark systems (Video Seal, TrustMark) only detect watermarks you embedded yourself.

### Cited Findings
- c2pa-rs (Rust SDK), contentauth org. 438 stars, pushed 2026-11-02, license shown as "other" by GitHub; docs for c2patool say Apache-2.0 / MIT dual. — [GitHub](https://github.com/contentauth/c2pa-rs)
- c2patool CLI: Apache-2.0 and MIT dual, 160 stars, pushed 2026-10-30. Reads manifests of image, audio, video files. — [GitHub](https://github.com/contentauth/c2patool)
- c2pa-python: Apache-2.0, 109 stars, pushed 2026-10-29. — [GitHub](https://github.com/contentauth/c2pa-python)
- c2pa-js: MIT, 40 stars, pushed 2026-11-02. c2pa-node archived (MIT, last push 2025-09-19), c2pa-node-v2 also archived (2026-06-19). verify-site: Apache-2.0. — [c2pa-js](https://github.com/contentauth/c2pa-js); [verify-site](https://github.com/contentauth/verify-site)
- Google Gemini app verification (blog dated 2025-11-20): checks SynthID plus Gemini reasoning, only for Google-made images. Nano Banana Pro images carry C2PA metadata; C2PA verification of non-Google content was announced as planned. The SynthID Detector portal was limited to testers, with no public API mentioned. — [Google blog](https://blog.google/innovation-and-ai/products/ai-image-verification-gemini-app/)
- A secondary report says Google extended SynthID verification to Lens, AI Mode and Circle in 2026, and that OpenAI will add SynthID to ChatGPT/API images. Unconfirmed by primary sources I read. — [search summary](https://chromeunboxed.com/you-can-now-use-the-gemini-app-to-spot-ai-generated-images)
- google-deepmind/synthid-text: Apache-2.0, 1,225 stars, pushed 2026-10-26. This is text only; there is no official SynthID image detector repo. — [GitHub](https://github.com/google-deepmind/synthid-text)
- Meta Video Seal (image and video watermarking): MIT, 817 stars, pushed 2026-07-02. AudioSeal MIT, 802 stars. — [GitHub](https://github.com/facebookresearch/videoseal)
- Adobe TrustMark: 180 stars, pushed 2026-10-16, license NOASSERTION on GitHub (not verified). Stable Signature (Meta): NOASSERTION, 529 stars. invisible-watermark: MIT, 1,992 stars, last push 2023-09-23. — [TrustMark](https://github.com/adobe/trustmark); [Stable Signature](https://github.com/facebookresearch/stable_signature); [invisible-watermark](https://github.com/ShieldMnt/invisible-watermark)
- Unofficial reverse-engineering and removal tools for SynthID exist (reverse-SynthID, 5,042 stars, license "other"; others claim removal). Unverified third-party claims, and a sign that SynthID is not tamper-proof. — [GitHub](https://github.com/aloshdenny/reverse-SynthID)

### Inferences
- Pipeline suggestion: read C2PA with c2pa-python or c2patool first (cheap, high-precision when present, but metadata is often stripped), then an ML detector as a probabilistic signal, and Gemini or Google's tools for SynthID checks on Google images.

### Gaps
- Midjourney, GPT-image and Qwen-Image C2PA/watermark status not verified from primary sources.
- Licenses for TrustMark and Stable Signature weights not read.
- Current Content Credentials trust-list/conformance details were not researched.
