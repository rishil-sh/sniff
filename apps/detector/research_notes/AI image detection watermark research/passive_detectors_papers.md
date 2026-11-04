# Passive AI-generated-image detection: 2024-2026 literature

## Key methods and what they do

### Takeaway
The field splits into (a) CLIP/foundation-model feature probes (UnivFD, FusionDetect, C2P-CLIP, DRCT), (b) low-level/spectral artifact detectors (NPR, LaDeDa, SPAI, AIDE's low-level branch), (c) data-centric training (B-Free, Community Forensics, GAMMA), (d) training-free (ZED, RIGID), and (e) MLLM explainers (AIGI-Holmes). Data design is repeatedly shown to matter more than architecture.

### Cited Findings
- SPAI (CVPR 2025): learns the spectral distribution of real images via masked frequency reconstruction (ViT-B/16, frozen backbone pretrained with masked frequency modeling), scores fakes as out-of-distribution by Spectral Reconstruction Similarity; Spectral Context Attention handles any resolution without resize/crop. Trained on ~180k real + 180k fake 256x256 images from a single latent diffusion model (Corvi et al.). — [arXiv 2411.19417](https://arxiv.org/html/2411.19417v2); [CVPR page](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html)
- B-Free (CVPR 2025, Guillaro et al.): training set from MS-COCO, 51,517 real and 309,102 "self-conditioned" fakes made by SD 2.1 inpainting with an empty mask (same content as the real), plus inpainted-object augmentations; DINOv2-with-registers backbone fine-tuned end to end on 504x504 crops with no resizing. — [arXiv 2412.17671](https://arxiv.org/html/2412.17671v2)
- Community Forensics (CVPR 2025, Park and Owens): 2.7M images from 4,803 generators (thousands of community diffusion models plus hand-picked open/commercial ones); performance rises with number and diversity of generators, even with similar architectures. Caveat: mostly diffusion models. — [arXiv 2411.04125](https://arxiv.org/pdf/2411.04125v1.pdf); [summary](https://www.themoonlight.io/en/review/community-forensics-using-thousands-of-generators-to-train-fake-image-detectors)
- AIDE (ICLR 2025, "Sanity Check"): hybrid of a low-level branch (high/low-frequency patch artifacts) and a CLIP semantic branch; introduces the Chameleon dataset of fakes that passed a human Turing test. Reported +3.5% / +4.6% accuracy over SOTA on AIGCDetectBenchmark / GenImage; 9 off-the-shelf detectors almost all call Chameleon fakes real; AIDE itself also struggles on Chameleon (exact numbers not retrieved). — [ICLR page](https://iclr.cc/virtual/2025/poster/29835); [arXiv 2406.19435](https://arxiv.org/html/2406.19435v2)
- LaDeDa / WildRF (Cavia et al., Hebrew U): scores 9x9 patches then pools to an image score to focus on local artifacts; Tiny-LaDeDa is a distilled 4-conv-layer student (375x fewer FLOPs). Reports 93.7% mAP on WildRF (social-media-sourced benchmark) and notes a large gap to perfect real-world detection remains. Predates 2025 generators (June 2024). — [arXiv 2406.09398](https://arxiv.org/html/2406.09398v1); [summary](https://www.themoonlight.io/review/real-time-deepfake-detection-in-the-real-world)
- FusionDetect (arXiv 2510.05740): concatenates frozen CLIP ViT-L/14 and DINOv2 ViT-L/14 features into a 4-layer MLP; trained on only 60k images with SD1.4/SD2.1 fakes and 10% JPEG/blur augmentation. — [arXiv 2510.05740](https://arxiv.org/html/2510.05740v1)
- GAMMA (arXiv 2509.10250): inpainting-based and semantics-preserving training with pixel-level source-attribution segmentation heads; claims +5.8% on GenImage and "strong robustness" on GPT-4o (no figure in abstract). — [arXiv 2509.10250](https://arxiv.org/abs/2509.10250v2)
- AIGI-Holmes (arXiv 2507.02664): MLLM detector trained on Holmes-Set (SFT explanations plus DPO preference data, built with a "Multi-Expert Jury"), three-stage pipeline (visual expert pretraining, SFT, DPO) and collaborative decoding combining the visual expert with MLLM reasoning; evaluated on unseen generators incl. FLUX, SD3.5, PixArt and autoregressive models (numbers not retrieved). — [arXiv 2507.02664](https://arxiv.org/abs/2507.02664); [search snippet listing test generators](https://arxiv.org/pdf/2507.02664)
- Training-free: ZED (ECCV 2024, Cozzolino et al.) is a zero-shot, entropy/surprise-based detector with no fake training data, claiming >3% average accuracy over SOTA; RIGID compares representation similarity of an image vs. its noise-perturbed copy, claiming >25% average over the previous best training-free method. — [ZED arXiv 2409.15875](https://arxiv.org/abs/2409.15875); [RIGID summary](https://fugumt.com/fugumt/paper_check/2405.20112v1_enmode)

### Inferences
- B-Free, GAMMA and Community Forensics all attack dataset bias (content/format/generator-count) rather than architecture; this is the dominant 2025 trend.

### Gaps
- Not retrieved with sources: UnivFD, NPR, DRCT, C2P-CLIP, FatFormer original-paper details and numbers (only their numbers in third-party benchmarks below); other 2025-26 MLLM detectors' numbers; HFI/manifold-style training-free methods (search found none).

## Reported results on new generators (FLUX, SD3/3.5, Midjourney, GPT-image, Gemini image)

### Takeaway
Numbers are not comparable across papers (different metrics, thresholds, training data). Authors' own results are high (B-Free 92-99% on FLUX/SD3.5/MJ); independent zero-shot evaluation of released checkpoints is poor (18-31% average accuracy on newest commercial generators).

### Cited Findings
- B-Free balanced accuracy (own paper, Table 5): FLUX 92.3, SD3.5 98.9, Midjourney 99.6, DALL-E 3 98.2, DALL-E 2 95.6; average 96.3, about 20.7 points above second-best. Averaged over GenImage (unbiased), FakeInversion, SynthWildX: 90.4. — [arXiv 2412.17671](https://arxiv.org/html/2412.17671v2)
- SPAI AUC (own paper, avg over 5 real sources): SD1.4 99.6, SDXL 97.4, Firefly 96.0, MJv5 94.5, DALLE3 90.2, MJv6.1 84.0, **Flux 83.0, SD3 75.9**; average 91.0 over 13 generators. Baseline avg AUC: RINE 85.5, DMID 83.5, PatchCraft 80.4, DeFake 70.1, UnivFD 67.3, LGrad 65.9, CNNDetect 55.5, **NPR 45.4**. — [arXiv 2411.19417](https://arxiv.org/html/2411.19417v2)
- FusionDetect on its OmniGen benchmark (12 generators): mean accuracy 97.38%, weakest generator Midjourney v7 at 87.5%. Caveat: authors' own benchmark and partly retrained baselines. — [arXiv 2510.05740](https://arxiv.org/html/2510.05740v1)
- Independent zero-shot benchmark of 16 detectors / 23 checkpoints on 12 datasets (incl. GPT-4o via Echo-4o and Nano-Banana via Nano-consistent-150k): per-generator mean accuracy across detectors Flux Dev 21%, Firefly v4 18%, Midjourney v7 24%, Imagen 4 19%, DALL-E 3 31% (vs ProGAN 87%, StyleGAN2 82%, SD1.4 73%). Hardest datasets: Community Forensics test 37.6%, MNW 41.7%, Nano-Banana 44.6%. No per-detector GPT-4o numbers in the retrieved portion. — [arXiv 2602.07814](https://arxiv.org/html/2602.07814)
- Same benchmark, mean accuracy: Community-Forensics 0.78, DRCT CLIP-ViT 0.72, SAFE (FatFormer) 0.69, PatchCraft 0.68, DRCT ConvNeXt 0.51, FreDect 0.49, UnivFD 0.41, CNNSpot 0.38. The abstract says 75.0 for Community-Forensics but Table 3 says 78.0 (internal inconsistency). — [arXiv 2602.07814](https://arxiv.org/html/2602.07814)
- VCT2 benchmark (166k images; SD, SDXL, SD3 Medium, SD3.5 Large, DALL-E 3, MJ6): DALL-E 3 and SD3.5 gave lowest detection accuracy (no FLUX or GPT-4o). — [arXiv 2411.16754](https://arxiv.org/html/2411.16754v2) (via search summary only)
- A 2026 benchmark dataset for GPT Image 2 and Nano Banana 2 exists (arXiv 2608.01258) but the PDF was too large to fetch; results not read. — [arXiv 2608.01258](https://arxiv.org/pdf/2608.01258)

### Inferences
- Author-reported numbers are inflated relative to independent ones, due to matched post-processing, test-set curation, and tuned baselines. Treat any single number as an upper bound.
- Both SPAI and independent benchmarks put FLUX/SD3/MJ v6+/v7 as the weak spots; no source I retrieved gives clean GPT-image/Gemini-image per-detector numbers.

### Gaps
- Per-detector results on GPT-image-1/GPT-4o and Gemini/Nano-Banana images: not found in retrievable text. AIDE Chameleon numbers: not retrieved.

## Robustness: JPEG, resizing, social media

### Takeaway
Robustness is trained in (augmentation, no-resize, content-matched fakes), not architectural. Most spectral/low-level methods are fragile; B-Free and FusionDetect report strong robustness but on their own protocols.

### Cited Findings
- B-Free: balanced accuracy stays above 80% in the hardest JPEG/resize/blur case on Synthbuster; FakeInversion with simulated social-network upload: AUC 99.3 to 98.5, bAcc about 86%. On 1,400 viral images tracked 100 days, most competitors fall below 70% accuracy after day one (DeFake about 75%), B-Free about 92%. — [arXiv 2412.17671](https://arxiv.org/html/2412.17671v2)
- FusionDetect: 79-81% accuracy across JPEG QF 95-50 and blur sigma 1-3; SSP falls from 76.86 to 52.77 at QF 95, NPR from 74.33 to 52.30. — [arXiv 2510.05740](https://arxiv.org/html/2510.05740v1)
- SPAI: outperforms the two best competitors under JPEG/WebP (q 85/70/50), blur, noise, resize (85/70/50%); exact values only in a figure. Fails on derivative images (screenshots, memes, photos of screens/prints). — [arXiv 2411.19417](https://arxiv.org/html/2411.19417v2)
- AIGIBench (23 fake subsets incl. social-media and AI-art-platform real data, 11 detectors): detectors accurate in controlled settings drop significantly on real-world data; common augmentations give limited benefit; test-time preprocessing effects are nuanced. — [arXiv 2505.12335](https://arxiv.org/abs/2505.12335)
- The 2602.07814 zero-shot benchmark did NOT evaluate compression/social-media robustness (listed as future work). — [arXiv 2602.07814](https://arxiv.org/html/2602.07814)

### Inferences
- NPR's 45.4 AUC in SPAI's table (below chance-ish) is consistent with NPR's upsampling-artifact cue being destroyed by resizing/new decoders; this is an inference, not stated by sources.

### Gaps
- No independent, standardized JPEG/resize results for the newest generators.

## What generalizes best vs overfits (consensus)

### Takeaway
No universal winner. Ensembles/large-diverse-data detectors (Community-Forensics) and content-matched, no-resize fine-tuned foundation models (B-Free) generalize best; frequency/low-level and GAN-era detectors overfit.

### Cited Findings
- No universal winner; rankings vary hugely across datasets (Spearman 0.01-0.87); Community-Forensics best on average and most stable (partly because it ensembles five models); training-data alignment matters more than architecture (AIDE and DRCT variants with the same architecture differ substantially); frequency-domain detectors perform poorly zero-shot. — [arXiv 2602.07814](https://arxiv.org/html/2602.07814)
- Generator count/diversity drives generalization. — [Community Forensics](https://arxiv.org/pdf/2411.04125v1.pdf)
- Detectors over-associate content seen in training fakes (asymmetric bias) and show limited cross-model generalization (arXiv 2604.12353, 2603.00717); specialist detectors near-perfect on curated sets degrade in the wild (arXiv 2602.01738, ACM MM '26) — snippet-level only. — [2604.12353](https://arxiv.org/html/2604.12353v1); [2603.00717](https://arxiv.org/html/2603.00717); [2602.01738](https://arxiv.org/html/2602.01738)
- FusionDetect argues generalization has two axes (unseen generators and unseen semantic domains), supported mainly by a qualitative t-SNE; the analysis flagged caveats (Chameleon-style prompts in training, mixed retrained/official baselines). — [arXiv 2510.05740](https://arxiv.org/html/2510.05740v1)
- Chameleon: human-indistinguishable fakes are mostly misclassified as real by existing detectors. — [ICLR 2025](https://iclr.cc/virtual/2025/poster/29835)

### Inferences
- Practical implication: a production system should not rely on one detector; ensemble plus calibrated thresholds, and expect precision/recall collapse on each new generator generation.

### Gaps
- Overfit evidence for UnivFD/NPR/C2P-CLIP from their own papers not retrieved.

## Open problems

### Takeaway
Newest commercial generators, real-world degradation, calibration, adversarial robustness and derivative content (screenshots) remain unsolved.

### Cited Findings
- Commercial generators (Flux Dev, Firefly v4, MJ v7, Imagen 4) defeat most open detectors (18-31% mean accuracy). — [arXiv 2602.07814](https://arxiv.org/html/2602.07814)
- SPAI authors: method does not solve detection; screenshots/memes/photos of screens break the spectral signal; future work is combining spectral with semantic context. — [arXiv 2411.19417](https://arxiv.org/html/2411.19417v2)
- B-Free: trained only on SD2.1 reconstructions, may fail on future generators with very different synthesis; vulnerable to adversarial attack. — [arXiv 2412.17671](https://arxiv.org/html/2412.17671v2)
- LaDeDa authors: large gap to perfect real-world detection remains. — [arXiv 2406.09398](https://arxiv.org/html/2406.09398v1)
- Standardized evaluation is lacking: threshold-0.5 accuracy vs AUC differ by method and dataset; benchmarks are inconsistent (e.g. 2602.07814 internal number mismatches). — [arXiv 2602.07814](https://arxiv.org/html/2602.07814)

### Inferences
- Threshold calibration per generator/post-processing and open-set/OOD scoring are likely the most useful engineering levers; autoregressive/multimodal-LLM generators (GPT-image, Gemini) are under-evaluated.

### Gaps
- Searches ran with a ~15 call budget; did not cover FatFormer, C2P-CLIP, DRCT primary papers, other 2026 MLLM/VLM detectors, or surveys. Search-engine summaries (VCT2, 2603.00717, 2604.12353, 2602.01738) were not read in full.
