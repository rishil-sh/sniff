# Raising the accuracy of the open-source AI text detector

For this pipeline, the next accuracy gains come from measuring properly and fixing the plumbing, not from adding another detector. The current numbers cannot support any false-positive claim. Zero false positives on 65 human texts only puts the 95% upper bound on the false-positive rate (FPR) at about 5.6%. Five caught essays leave the true-positive rate (TPR) anywhere between 57% and 100%. Both voting classifiers were also trained on RAID, so the RAID abstracts in the eval set are very likely in-sample for them. **The first change is therefore a held-out eval set built around this tool's users**: English-learner and native student essays, formal writing, modern chat-model text, and paraphrased and Unicode-attacked variants, reported as TPR at 1% FPR with bootstrap confidence intervals.

Two near-free fixes come next:
- **Unicode normalisation.** On RAID's own leaderboard data, homoglyphs cut Binoculars from 79% to 36% TPR at 5% FPR, and zero-width spaces cut desklib from 95% to 88%.
- **Fakespot's `clean_text` preprocessing**, which the model card asks for and `text.py` skips.

After that:
- **Calibrate on held-out data.** Calibrate the two classifiers, set thresholds at a fixed FPR, and replace the hand-weighted mean and disagreement rule with a small logistic stacker that also takes the zero-shot scores. Fakespot's own report found that combining RoBERTa with Binoculars raised worst-case out-of-domain AUROC from 0.31 to 0.95.
- **Test stronger open classifiers.** MELD, DeBERTa-ConPara and Tabularis ModernBERT (all MIT) beat desklib on RAID and on author-run cross-dataset tests. None has independent results on 2026 generators, so treat them as candidates to measure, not upgrades to assume.
- **Longer term, fine-tune our own detector.** The strongest published recipe is a QLoRA-fine-tuned decoder trained on diverse, fresh, adversarially augmented data. PAN 2025's winner reached ROC-AUC 0.995 at 0.6% FPR. On the same test, a Binoculars Llama-3.1-8B pair reached 0.827 at 26% FPR.

Muse can't replace the Llama pair for logit scoring. We tested this on 2026-11-03: every Muse text model on the Model API rejects `logprobs`, and the API has no completions/echo endpoint for scoring the prompt's own tokens. It is, however, the cheapest source of fresh generator data and paraphrase attacks.

Every number below says who measured it. Most are author-reported, a few come from independent leaderboards, and claims we couldn't trace to a primary source are marked **unverified**. Status as of 2026-11-03.

## What today's numbers can and cannot show

The pipeline as read from the repo:
- **Voting classifiers.** `txt.fakespot` (RoBERTa-base, 512 tokens) and `txt.desklib` (DeBERTa-v3-large, 768 tokens) vote with weight 0.5 each on raw sigmoids. Neither is calibrated.
- **Bands and disagreement.** `fusion.py` uses `chance_band` 0.15, `max_disagreement` 0.6 and bands 0.20/0.80.
- **Zero-shot pair.** Binoculars and Fast-DetectGPT come from one forward pass each through a Llama base/Instruct pair on the first 512 tokens, and have weight 0.
- **Eval script.** `scripts/eval_images.py` fits Platt scaling on the same rows it reports metrics on and gives point estimates only.

Five problems follow from the sources:

- **The eval set is contaminated for both voters.** The desklib model card lists `liamdugan/raid` as its training dataset ([desklib card](https://huggingface.co/desklib/ai-text-detector-v1.01)). Fakespot's training-data list includes RAID and `falcon-refinedweb` web text ([ApolloDFT data README](https://github.com/FakespotAILabs/ApolloDFT/blob/main/data/README.md)). RAID publishes labels only for its train and extra splits; test labels are hidden ([RAID repo](https://github.com/liamdugan/raid)). *Inference:* the 40+40 labelled RAID abstracts are almost certainly training-distribution data for both classifiers, so "0/65 human false positives" is optimistic.
- **The sample sizes are too small to bound anything.** Our own calculation, using the Wilson interval ([binomial CI](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval)):

  | Observation | 95% confidence interval |
  |---|---|
  | 0 false positives in 65 human texts | FPR up to 5.6% |
  | 5/5 essays caught | TPR 57–100% |
  | 0/5 essays caught (zero-shot pair) | TPR 0–43% |

  So the essay results neither prove the classifiers good nor the zero-shot pair useless.
- **A fixed 0.5 cut is not a principled threshold.** On RAID's test set, desklib v1.01 needs a score of 0.17 (Reddit) to 0.77 (abstracts) to hold 5% FPR, and 0.72 to 0.98 to hold 1% FPR, depending on the domain. These are per-domain thresholds from the leaderboard's own results file ([RAID results data](https://raid-bench.xyz/data/submissions/desklib-ai-text-detector-v1.01.json)).
- **Fakespot is run without its preprocessing.** The card says "for better performance, you should apply the `clean_text` function". That function strips Markdown, code and HTML and flattens newlines ([Fakespot card](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1); [utils.py](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/blob/main/utils.py)). `score_classifiers` passes raw text. The gain is unmeasured but the cost is about one line.
- **Long documents are mostly unread.** Both classifiers truncate at 512 or 768 tokens and the zero-shot pair at 512, so most of a long document never reaches any detector.

## 1. Zero-shot and logprob methods

Several methods can reuse the observer/performer logits we already compute. The question is which ones beat analytic Fast-DetectGPT and Binoculars on modern chat text. The honest answer is that newer methods mostly report small gains on older generators, and **the choice of scoring model matters more than the formula**.

| Method | Venue / year | Reported result (benchmark, metric, FPR) | Licence | Compute | Fits our pipeline? | Effort |
|---|---|---|---|---|---|---|
| Binoculars | ICML 2024 | Authors: over 90% of ChatGPT text caught at 0.01% FPR with a Falcon-7B pair ([paper](https://arxiv.org/abs/2401.12070)). Independent: RAID 79.0% TPR at 5% FPR, unattacked ([RAID data](https://raid-bench.xyz/data/submissions/binoculars.json)). PAN 2025, Llama-3.1-8B pair: ROC-AUC 0.827, **FPR 26.3%** at a validation-tuned threshold ([PAN 2025 overview](https://ceur-ws.org/Vol-4038/paper_277.pdf)) | BSD-3 | 2 forward passes | Already implemented | — |
| Fast-DetectGPT (analytic) | ICLR 2024 | Authors: average AUROC 0.9887 white-box; 0.9615 on ChatGPT and 0.9061 on GPT-4 with a GPT-J/Neo-2.7 pair; about 87% recall at 1% FPR on ChatGPT ([paper](https://arxiv.org/abs/2310.05130)). Independent: RAID 78.4% at 5% FPR, unattacked (our aggregation of [leaderboard rows](https://raid-bench.xyz/data/submissions/fastdetectgpt.json)) | MIT | Same passes | Already implemented | — |
| AdaDetectGPT | NeurIPS 2025 | Authors, gemma-2-9b pair, AUROC: 0.9589 vs Fast-DetectGPT 0.9571 on GPT-4o text and 0.9478 vs 0.9439 on Claude-3.5-Haiku text (XSum/Writing/Yelp/Essay) ([paper](https://arxiv.org/abs/2510.01268)) | **No licence** on repo ([GitHub](https://github.com/Mamba413/AdaDetectGPT)) | Same passes plus a small learned witness function | Yes, needs labelled data | Medium |
| Lastde / Lastde++ | arXiv 2024 | Authors: black-box average AUROC 91.47 (Lastde++) vs Fast-DetectGPT 85.68 with a GPT-J proxy; white-box 98.85 vs 97.82 ([paper](https://arxiv.org/abs/2410.06072)) | **No licence** ([GitHub](https://github.com/TrustMedia-zju/Lastde_Detector)) | One pass plus time-series statistics | Yes, technically | Low, but the licence blocks reuse |
| DetectLLM (LRR / NPR) | Findings EMNLP 2023 | In Lastde's tables: LRR 80.45 and NPR 72.75 black-box AUROC, both below Fast-DetectGPT ([Lastde](https://arxiv.org/abs/2410.06072)) | No licence ([GitHub](https://github.com/mbzuai-nlp/DetectLLM)) | LRR is free from our pass; NPR needs perturbations | LRR only, as a stacking feature | Low |
| ImBD (imitate before detect) | AAAI 2025 | Authors: +13% AUC on text revised by open models, +5% on GPT-3.5-revised, +19% on GPT-4o-revised text vs prior SOTA ([paper](https://arxiv.org/abs/2412.10432)) | Apache-2.0 ([GitHub](https://github.com/Jiaqi-Chen-00/ImBD)) | Fine-tunes the scoring model with SPO on about 1,000 samples (about 5 minutes) | Yes, as a tuned performer | Medium |
| DetectAnyLLM (DDL) | arXiv 2025 | Authors on MIRAGE (17 LLMs including GPT-4o, Claude-3.7, Gemini-2.0, DeepSeek-R1, Llama-3.1): AUROC 0.95 vs Fast-DetectGPT 0.78 on generated text and 0.93 vs 0.57 on polished text ([paper](https://arxiv.org/abs/2509.14268)) | **Non-commercial** "Pi-Lab License" ([GitHub](https://github.com/fjc2005/DetectAnyLLM)) | LoRA r=8 on the scoring model, 500 training pairs | The idea fits; the code licence doesn't | Medium |
| Glimpse | ICLR 2025 | Authors: about 0.95 average AUROC using GPT-3.5 as scorer ([paper](https://arxiv.org/abs/2412.11506)) | MIT | Needs proprietary-API top-k logprobs **on the prompt** | No (see the Muse section) | — |
| DALD | arXiv 2024 | Abstract claims SOTA black-box detection, no numbers on the page ([paper](https://arxiv.org/abs/2406.05232)) | Code not found | Fine-tunes a surrogate on public GPT-4/Claude outputs | Idea only | Medium |
| TOCSIN | EMNLP 2024 | Plug-in "token cohesiveness" score over random token deletions; abstract gives no numbers ([paper](https://arxiv.org/abs/2409.16914)) | MIT ([GitHub](https://github.com/Shixuan-Ma/TOCSIN)) | Several extra passes per text | Possible, unmeasured | Medium |
| BiScope | NeurIPS 2024 | Measured by AdaDetectGPT's authors: AUROC 0.909 on GPT-4o and 0.893 on Claude-3.5 text, below Fast-DetectGPT ([AdaDetectGPT Table 2](https://arxiv.org/abs/2510.01268)) | CC-BY-SA-4.0 ([GitHub](https://github.com/MarkGHX/BiScope)) | Generation plus a classifier | Poor | High |
| DNA-GPT | ICLR 2024 | Authors: with K=10 regenerations by the *same* model, GPT-4 text gets AUROC 99.34 and 91% TPR at 1% FPR on Reddit, but only 30% TPR on XSum ([paper](https://arxiv.org/abs/2305.17359)) | MIT ([GitHub](https://github.com/Xianjun-Yang/DNA-GPT)) | 5–10 generations per text | Only for "was this written by *this* model" | High |
| PHD / intrinsic dimension | NeurIPS 2023 | Among detectors an independent study found reaching **0% TPR at 1% FPR** on some unseen sets ([Tufts et al., NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.271/)) | MIT ([GPTID](https://github.com/ArGintum/GPTID)) | Encoder embeddings | Weak | Low |
| Compression (Zippy, PPMd) | — | Fakespot: Zippy AUROC 0.66–0.77 on RAID ([Apollo report](https://github.com/FakespotAILabs/ApolloDFT/blob/main/apollo-text.md)). PAN 2025: PPMd baseline ROC-AUC 0.644 ([overview](https://ceur-ws.org/Vol-4038/paper_277.pdf)) | MIT ([Zippy](https://github.com/thinkst/zippy)) | CPU | Not worth it | — |
| DivEye | TMLR 2026 | Authors: up to 33.2% over zero-shot detectors, +18.7% as an auxiliary signal ([paper](https://arxiv.org/abs/2509.18880)). RAID: 93.6% at 5% FPR, unattacked | **CC BY-NC-SA 4.0** ([LICENSE](https://github.com/IBM/diveye)) | Surprisal statistics plus a classifier | Licence blocks it | — |
| Multi-view Gaussian processes | arXiv 2026-06 | Stylistic, likelihood and structural views with calibrated abstention; claims gains on DetectRL, RAID and PAN 2025, no numbers on the abstract page ([paper](https://arxiv.org/abs/2606.14060)) | No code found | — | Watch | — |

**Which observer/performer pairs work on modern chat text?** No peer-reviewed comparison of Llama-3.x, Qwen or Gemma pairs on 2025–26 generators turned up. The evidence:

- **Falcon-7B pair.** Binoculars' authors found Falcon-7B / Falcon-7B-Instruct best among the pairs they tried, including several Llama-2 pairs. Using the same model for both roles was "competitive" but worse ([paper §A.5 and Table 3](https://arxiv.org/html/2401.12070v3)).
- **Llama-3-8B pair.** Fast-DetectGPT's README (2026-01-31) says a Llama3-8B / Llama3-8B-Instruct pair "substantially outperforms falcon-7b/falcon-7b-instruct, especially on LRM-generated texts", with no numbers and no peer review ([README](https://github.com/baoguangsheng/fast-detect-gpt)).
- **gemma-2-9b pair.** AdaDetectGPT's authors got Binoculars and Fast-DetectGPT to about 0.95 AUROC on GPT-4o and Claude-3.5-Haiku text ([paper](https://arxiv.org/html/2510.01268v5)). Gemma weights are under Google's Gemma Terms of Use, not an OSI licence; flag before shipping.
- **Llama-2 pairs on 2026 generators.** In the GPTZero paper, Binoculars scored AUC 83.5 on essays from GPT-5.2, Gemini 3 Pro, Claude Sonnet 4.5 and Grok 4 Fast, with 41.9% recall at 1% FPR, and AUC 40.0 on abstracts ([GPTZero, Table 1](https://arxiv.org/html/2602.13042v1)). This is vendor-run (GPTZero sells a competing detector), and the baseline models were reportedly Llama-2 7B (**unverified**; from a search snippet, not the table).
- **Llama-3.1-8B pair, PAN 2025.** ROC-AUC 0.827 on the PAN 2025 test set, with an FPR of 26.3% at the organisers' validation-tuned threshold ([PAN 2025 overview](https://ceur-ws.org/Vol-4038/paper_277.pdf)).

*Inference:* the 3B pair's 0/5 matches this picture of zero-shot scores weakening on modern, heavily post-trained generators. An 8–9B pair should do better but is unlikely to match the classifiers alone. The realistic job for the zero-shot scores is as a feature in a stacker, not a vote. Pairs to measure, by licence:

| Pair | Licence |
|---|---|
| Llama-3.1-8B (already wired up) | Llama community licence |
| Falcon-7B | Apache-2.0 |
| Qwen2.5-7B | Apache-2.0 |
| gemma-2-9b | Gemma Terms of Use; flag |

OLMo-2-7B (Apache-2.0) is also an option, but there is no detection evidence for it (our suggestion). Two implementation notes:
- **`BINOCULARS_PRIOR` is wrong for Llama.** It centres on Falcon's 0.9015 accuracy threshold; Binoculars' code also ships a separate low-FPR threshold of 0.8536 ([detector.py](https://github.com/ahans30/Binoculars/blob/main/binoculars/detector.py)). Both are Falcon-specific, so only a held-out Platt fit is meaningful for Llama.
- **The pair breaks on Unicode attacks.** Homoglyphs drop it from 79% to 36% TPR on RAID (section 6).

## 2. Supervised open-weight classifiers and how they rank

**RAID leaderboard, recomputed from its raw result files.** The figures below are TPR with per-domain thresholds set on RAID's hidden human test texts ([RAID results index](https://raid-bench.xyz/data/submissions/_index.json); [leaderboard](https://raid-bench.xyz/leaderboard)). The leaderboard runs the evaluation itself, which makes this the most independent number in the field. But **most top entries train on RAID's train split, which has the same generators (2023-era GPT-4, ChatGPT, Llama-2-70B-chat, Mistral-7B, MPT, Cohere) and domains as the test split**. So these rows measure in-distribution skill and attack robustness, not generalisation to 2026 models.

| Detector (open weights) | Licence | Base | Unattacked, TPR@5% FPR | All attacks, TPR@5% | All attacks, TPR@1% |
|---|---|---|---|---|---|
| MELD ([HF](https://huggingface.co/anon-review-meld-2026/meld)) | MIT (anonymous-review repo; provenance unclear) | Ettin-1B encoder | 0.998 | 0.998 | 0.992 |
| GPTTrace Ettin-1B ([HF](https://huggingface.co/nakas/gpttrace-v5-1b)) | **CC-BY-SA-4.0** | Ettin-1B | 1.000 | 0.999 | 0.994 |
| Linda-RAID ([HF](https://huggingface.co/Lindarixon/Linda-RAID)) | **CC-BY-NC-4.0** | DeBERTa-v3-large | 0.997 | 0.995 | 0.977 |
| DeBERTa-ConPara ([HF](https://huggingface.co/SES-Lab-OTH/deberta-conpara)) | MIT | DeBERTa-v3-large | 0.990 | 0.990 | 0.966 |
| Tabularis ModernBERT R1.1 ([HF](https://huggingface.co/tabularisai/ai-text-detection)) | MIT | ModernBERT-base (149M) | 0.992 | 0.990 | 0.959 |
| TMR ([HF](https://huggingface.co/Oxidane/tmr-ai-text-detector)) | MIT | RoBERTa-base | 0.997 | 0.958 | 0.902 |
| ModernBERT raid-mage ([HF](https://huggingface.co/GeorgeDrayson/modernbert-ai-detection-raid-mage)) | Apache-2.0 | ModernBERT-base | 0.991 | 0.941 | 0.882 |
| ADAL ([HF](https://huggingface.co/Shushant/ADAL-detector-large)) | Apache-2.0 | — | 0.980 | 0.963 | 0.877 |
| **desklib v1.01 (ours)** | MIT | DeBERTa-v3-large | 0.949 | 0.912 | 0.765 |
| RADAR ([HF](https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B)) | Not stated on HF | RoBERTa-large | 0.656 | 0.639 | 0.431 |
| Binoculars (Falcon) | BSD-3 | — | 0.790 | n/a | n/a |
| fakespot RoBERTa (ours) | Apache-2.0 | RoBERTa-base | not submitted | — | — |

Commercial entries score about the same (Winston 0.999, Grammarly 0.982, GPTZero 0.929 at 1% FPR, all attacks), and their weights aren't available. Two points matter for us:
- **desklib has holes.** It ranks in the bottom half. Its worst RAID generator is Cohere base (0.654 TPR at 5% FPR) and its worst attacks are synonym swap (0.806), paraphrase (0.837) and upper/lower-case swap (0.872), all from our breakdown of its rows.
- **Fakespot has no independent number on RAID.** Its own report gives RAID AUROC 0.99 *in-distribution* but a worst out-of-domain AUROC of **0.31** (books held out). The full Apollo ensemble, RoBERTa plus Binoculars, which Fakespot has not released, kept a worst case of 0.95 ([Apollo report, Tables 1–2](https://github.com/FakespotAILabs/ApolloDFT/blob/main/apollo-text.md)).

**Cross-dataset tests.** The best available cross-dataset comparison of open checkpoints is the DeBERTa-ConPara authors' re-run of every RAID entry that publishes weights, scored by balanced accuracy at one fixed threshold ([ConPara README](https://github.com/SES-Lab-OTH/deberta-conpara)):

| System | HC3-QA, HC3-SI, MAGE (average) | M4 | HC3-SI alone |
|---|---|---|---|
| ConPara | 93.1 | 98.3 | 83.5 |
| MELD | 85.9 | 94.3 | 66.9 |
| ModernBERT raid-mage | 80.5 | 84.1 | — |
| desklib v1.01 | 79.2 | 90.6 | **56.4** |
| TMR | 70.6 | — | — |
| ADAL | 55.3 | — | — |
| RADAR | 54.3 | — | — |

The authors state the bias themselves: HC3, MAGE and M4 are in ConPara's training mix, so those columns are held-out data for ConPara but external data for everyone else. The ConPara card also reports, unusually candidly, **false-positive rates of 13–67% on academic prose** and 4–5× more errors below about 60 words ([ConPara card](https://huggingface.co/SES-Lab-OTH/deberta-conpara); [paper, AACL 2026](https://arxiv.org/abs/2610.00883)). Another independent study ran 15 public detectors over MAGE, RAID, HC3+ and M4GT and found **every model near chance (AUROC at or below 0.60) on at least one set**. On human-only sets, no system exceeded 0.8 accuracy and 0.8 recall at once ([Stowe & Patil, 2026](https://arxiv.org/abs/2604.16607)).

**Shared tasks and competitions**

| Event | What won | Reported result | Relevance |
|---|---|---|---|
| COLING 2025 GenAI Task 3, RAID ([overview](https://arxiv.org/abs/2501.08913)) | 9 teams, 23 systems | Several above 99% TPR at 5% FPR | Fixed domains and generators seen in training, so not a generalisation test |
| COLING 2025 GenAI Task 1 ([overview](https://arxiv.org/abs/2501.11012)) | 36 English teams | No winning scores on the abstract page | — |
| SemEval-2024 Task 8 ([overview](https://arxiv.org/abs/2404.14183)) | "For all subtasks, the best systems used LLMs" | 126 teams on subtask A | Subtask C is boundary detection |
| PAN 2025 Voight-Kampff ([overview](https://ceur-ws.org/Vol-4038/paper_277.pdf); [results](https://pan.webis.de/clef25/pan25-web/generated-content-analysis.html)) | Macko: **Qwen3-14B fine-tuned with QLoRA**, homoglyph-augmented training, model selection on external out-of-distribution data ([mdok](https://arxiv.org/abs/2506.01702)) | ROC-AUC 0.995, FPR 0.006, FNR 0.018. Baselines: TF-IDF SVM 0.963 (FPR 0.106); Binoculars Llama-3.1-8B 0.827 (FPR 0.263) | Test set includes GPT-4o, o1, o3-mini, GPT-4.5, Gemini 2.0 Flash, DeepSeek-R1-distill and Llama-3.3-70B. mdok weights are **CC-BY-NC-ND** and code is **GPL-3.0**, so reuse the recipe, not the artefacts |
| Kaggle "LLM – Detect AI Generated Text" ([1st place code](https://github.com/rbiswasfc/llm-detect-ai); [write-up](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121)) | Mistral-7B QLoRA classifiers plus a DeBERTa-v3-large ranking model plus an embedding kNN, trained on essays from many LLMs fine-tuned on PERSUADE to imitate students | Private-leaderboard score not retrieved (**unverified**) | Shows that in-domain synthetic data is the main lever. Code is MIT |

The pattern across all five events (our synthesis): winners are fine-tuned LLMs or large encoders trained on generator- and domain-diverse data with deliberate obfuscation augmentation. Pure zero-shot baselines finish mid-table or lower.

## 3. Fine-tuning our own detector

- **Decoder LoRA vs. encoder.** The 2025 evidence favours a QLoRA-tuned decoder classifier for robustness: PAN's winner used Qwen3-14B; the Kaggle winner used Mistral-7B ([mdok](https://arxiv.org/abs/2506.01702); [Kaggle 1st](https://github.com/rbiswasfc/llm-detect-ai)). Encoders are close behind on RAID and much cheaper to serve: MELD and GPTTrace use Ettin-1B, ConPara uses DeBERTa-v3-large, and Tabularis uses a 149M ModernBERT trained for 2.5 hours on 4×H100 ([Tabularis card](https://huggingface.co/tabularisai/ai-text-detection)). *Inference:* on an L40S, QLoRA on Llama-3.1-8B or Qwen3-8B (Apache-2.0) is feasible. A DeBERTa-v3-large run is the cheaper first experiment.
- **Scoring-model tuning instead of a classifier head.** ImBD (Apache-2.0, about 1,000 samples) and DetectAnyLLM's direct discrepancy learning (500 pairs, LoRA r=8; non-commercial code) tune the Fast-DetectGPT scoring model itself ([ImBD](https://arxiv.org/abs/2412.10432); [DetectAnyLLM](https://arxiv.org/abs/2509.14268)). That would turn our weight-0 performer into a trained detector with very little data. All results are author-reported.
- **Contrastive and retrieval methods.** DeTeCtive (NeurIPS 2024) learns a multi-level contrastive embedding and supports training-free incremental adaptation: new generators are added to a kNN database without retraining. In Stowe & Patil's independent test it had the best aggregate F1/AUROC among public models ([DeTeCtive](https://arxiv.org/abs/2410.20964); [independent eval](https://arxiv.org/abs/2604.16607)). The repo has **no licence** ([GitHub](https://github.com/heyongxin233/DeTeCtive)). Its successor DETree (NeurIPS 2025) is MIT ([GitHub](https://github.com/heyongxin233/DETree)).
- **Training data and licences.**

  | Dataset | Licence |
  |---|---|
  | RAID ([HF](https://huggingface.co/datasets/liamdugan/raid)) | MIT |
  | MAGE ([HF](https://huggingface.co/datasets/yaful/MAGE)) | Apache-2.0 |
  | HC3 ([HF](https://huggingface.co/datasets/Hello-SimpleAI/HC3)) | CC-BY-SA-4.0 |
  | M4GT-Bench ([paper](https://arxiv.org/abs/2402.11175)) | CC BY 4.0 per the paper; no licence file in the repo |
  | PERSUADE 2.0 ([GitHub](https://github.com/scrosseye/persuade_corpus_2.0)) | CC BY 4.0; over 25,000 essays, grades 6–12 |

  Two training-time lessons from the sources:
  - **Don't normalise the training corpus.** ConPara found that normalising Unicode in training data collapses 35.4% of RAID rows into duplicates and deletes the adversarial supervision. Normalise only at inference ([ConPara README](https://github.com/SES-Lab-OTH/deberta-conpara)).
  - **Mine hard negatives.** Pangram's "hard negative mining with synthetic mirrors" is reported to cut false positives by orders of magnitude on high-data domains (vendor report, no numbers on the page) ([Pangram](https://arxiv.org/abs/2402.14873)).
- **Generalisation to unseen generators.** M4GT-Bench concludes good detection "requires training data from the same domain and generators" ([M4GT](https://arxiv.org/abs/2402.11175)). MAGE's best detector caught 86.54% of out-of-domain text from an unseen LLM ([MAGE](https://arxiv.org/abs/2305.13242)). Pu et al. found detectors trained on medium-size models transfer to larger ones ([Pu et al.](https://arxiv.org/abs/2310.05165)). *Inference:* we need a rolling supply of fresh generations; see the Muse section.

## 4. Ensembling and calibration

- **Stacking.** The current rule is a 0.5/0.5 mean, plus inconclusive when the spread exceeds 0.6. A logistic regression over the classifier logits, the zero-shot scores, token count and normalisation flags would replace it. That stacker learns the weights, uses the disagreement as information rather than a veto, and outputs a probability one can threshold. Evidence that heterogeneous combinations help:
  - **Apollo:** worst-case out-of-domain AUROC 0.31 for RoBERTa alone vs 0.95 for RoBERTa + Binoculars ([Apollo](https://github.com/FakespotAILabs/ApolloDFT/blob/main/apollo-text.md)).
  - **DivEye:** +18.7% as an auxiliary feature ([DivEye](https://arxiv.org/abs/2509.18880)).
  - **PAN 2025:** the third-placed system ensembled a fine-tuned Qwen with a ModernBERT ([overview](https://ceur-ws.org/Vol-4038/paper_277.pdf)).
  - **Complementary failures on RAID:** Binoculars fails on homoglyphs (0.36) where desklib holds (0.997), and desklib loses more to paraphrase (0.837) than Binoculars does (0.821). These are our breakdowns of the leaderboard rows.
- **Platt vs isotonic vs temperature.** scikit-learn's guidance: sigmoid (Platt) "is most effective for small sample sizes". Isotonic "is more prone to overfitting, especially on small datasets" and only matches or beats sigmoid "when there is enough data (greater than ~1000 samples)". The calibrator must be fit on data independent of the model's training data ([sklearn calibration](https://scikit-learn.org/stable/modules/calibration.html)). Temperature scaling is the one-parameter multiclass case ([Guo et al. 2017](https://arxiv.org/abs/1706.04599)). For us that means Platt per detector, or a logistic stacker (which is Platt on several inputs), fit on a held-out split. The current in-sample fit to 73 RAID rows reports AUROC 1.0, which only shows the fit memorised its own data.
- **Thresholding at a fixed FPR.** Set the "likely AI" cut at the quantile of held-out human scores that gives the target FPR, separately per text type where possible. RAID does exactly this per domain, and desklib's required cut ranges from 0.72 to 0.98 for 1% FPR (see "What today's numbers can and cannot show").
- **Conformal guarantees.** Split-conformal calibration on human texts gives FPR ≤ α under exchangeability ([Angelopoulos & Bates](https://arxiv.org/abs/2107.07511)). Zhu et al. (ACL 2025) found that naive conformal prediction "constrains FPRs" but "leads to a significant reduction in detection performance", and propose a multiscale variant ([MCP](https://arxiv.org/abs/2505.05084)). *Inference:* the guarantee holds only for text distributed like the calibration set. That argues for calibrating per input type (student essay, formal or technical prose, web text) and abstaining outside them.
- **Abstention.** PAN scores a 0.5 output as "undecidable" and rewards it via C@1 ([PAN task](https://pan.webis.de/clef25/pan25-web/generated-content-analysis.html)). We already abstain. The change is to derive the inconclusive band from calibrated probabilities and measured FPR rather than fixed 0.2/0.8 cuts on uncalibrated sigmoids.

## 5. Long and mixed texts

- **Windowing.** No 2024–26 paper found compares window aggregation rules for these detectors. The supporting evidence:
  - Zero-shot accuracy rises with length ([Fast-DetectGPT §3.4](https://arxiv.org/abs/2310.05130); [DetectRL](https://arxiv.org/abs/2410.23746)).
  - Fakespot says accuracy improves with longer inputs ([card](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1)).
  - DNA-GPT gives a sliding-window variant for texts that start AI-written and continue human-written ([paper §4](https://arxiv.org/abs/2305.17359)).

  *Inference:* score overlapping windows of 512 tokens with stride 256 through every detector. Fuse the mean logit for the document verdict and show per-window scores as a heat map. Calibrate the aggregate on long held-out documents, because taking the maximum over many windows inflates FPR.
- **Mixed authorship.** It is hard for everyone:
  - Mixing human text into LLM text cut zero-shot AUROC by 18.17% in DetectRL ([DetectRL](https://arxiv.org/abs/2410.23746)).
  - PAN 2025's six-class human–AI collaboration subtask was won at F1 0.65, and systems "struggled significantly" with machine-written, human-edited text ([overview](https://ceur-ws.org/Vol-4038/paper_277.pdf)).
  - Sentence-level methods exist: SeqXGPT uses log-probability "waves" from white-box LMs (EMNLP 2023, Apache-2.0) ([paper](https://arxiv.org/abs/2310.08903); [GitHub](https://github.com/Jihuai-wpy/SeqXGPT)), and boundary detection is SemEval-2024 Task 8 subtask C and M4GT task 3 ([SemEval](https://arxiv.org/abs/2404.14183); [M4GT](https://arxiv.org/abs/2402.11175)).
  - A 2026 benchmark with full edit histories found detectors "often only perform well for specific notions" of AI-generated text ([AITDNA](https://arxiv.org/abs/2606.04906)).

  *Inference:* show window-level evidence and say "mixed or edited text is not reliably detected" rather than building a segment classifier now.

## 6. Robustness

**Paraphrase.**
- DIPPER cut DetectGPT from 70.3% to 4.6% detection at 1% FPR ([Krishna et al., NeurIPS 2023](https://arxiv.org/abs/2303.13408)).
- On RAID's paraphrase subset, desklib holds 0.837 and Binoculars 0.821 TPR at 5% FPR (RAID data). The strongest RAID-trained entries hold 0.98–0.99, but those were trained on RAID's own paraphrase attack.

**Humanizer tools.**
- Pangram's audit of 19 humanizers found many detectors fail on humanized text (vendor paper, GenAIDetect 2025) ([DAMAGE](https://arxiv.org/abs/2501.03437)).
- GPTZero reports 93.5% recall on paraphrased and "bypasser" text vs Originality 57.3% and Pangram 49.7% (vendor-run, competing products) ([GPTZero §5.3](https://arxiv.org/html/2602.13042v1)).
- In ELOQUENT 2025, back-translation and homoglyph obfuscations fooled PAN systems the most (inverse C@1 0.618 and 0.436, averaged over all classifiers) ([overview, Table 2](https://ceur-ws.org/Vol-4038/paper_277.pdf)).

**Homoglyph, zero-width and whitespace attacks.**
- SilverSpeak drove the MCC of seven detectors, including Binoculars and Fast-DetectGPT, to about 0 ([Creo & Pudasaini, GenAIDetect 2025](https://arxiv.org/abs/2406.11239)).
- On RAID, from our breakdown: homoglyphs cut Binoculars to 0.361 and Fast-DetectGPT to 0.587, and zero-width spaces cut desklib to 0.875.

**Defences, in order of cost.**
1. **Inference-time normalisation.** NFKC, strip zero-width and bidi characters, map confusables to ASCII, collapse whitespace. Tabularis projects +7.5 points TPR at 1% FPR from NFKC alone (author projection, **unverified**) ([card](https://huggingface.co/tabularisai/ai-text-detection)). ConPara calls inference-time normalisation "an effective defence" ([README](https://github.com/SES-Lab-OTH/deberta-conpara)). The presence of such characters in submitted prose is itself worth reporting as evidence of tampering (our suggestion).
2. **Adversarial training augmentation.** PAN's winner homoglyph-attacked part of its training data ([overview](https://ceur-ws.org/Vol-4038/paper_277.pdf)).
3. **RADAR-style adversarial training.** It underperforms on RAID (0.639) ([RADAR](https://arxiv.org/abs/2307.03838)).
4. **Retrieval.** Detects 80–97% of paraphrased generations at 1% FPR over 15M stored generations ([Krishna et al.](https://arxiv.org/abs/2303.13408)), but only for text the operator itself generated. That doesn't apply to us.

## 7. Fairness and false positives

**The evidence.**
- Seven detectors flagged 61.22% of 91 TOEFL essays as AI on average. 18 were flagged by all seven and 89 by at least one, while 88 US 8th-grade essays from ASAP were classified nearly perfectly. Having ChatGPT "enhance the word choices" cut the FPR to 11.77% ([Liang et al., Patterns 2023](https://arxiv.org/abs/2304.02819); data MIT on [HF](https://huggingface.co/datasets/WxWx/ChatGPT-Detector-Bias)).
- An EMNLP 2026 study of 135,389 non-native manuscripts paired with their professionally edited versions found false-positive rates from 0% to 100% across 13 detectors. Editing style, not authorship, was a key confound ([Park et al.](https://arxiv.org/abs/2608.26710)).
- Counter-evidence is author-reported. Binoculars' authors report 99.67% accuracy on EssayForum essays by non-native writers ([paper §5.2](https://arxiv.org/html/2401.12070v3)). An ETS study found no bias for detectors trained on its own GRE data (talk transcript only, **unverified**) ([ETS](https://www.ets.org/aera-ncme-conference/transcript/yang-jiang.html)).
- Independent numbers say formal and academic prose is the risk: ConPara's 13–67% FPR on academic subcorpora, and the 26.3% FPR of the Binoculars Llama pair on PAN 2025.

**Short texts.** ConPara: errors are 4–5× more frequent below about 60 words. DetectRL: zero-shot accuracy rises with length ([ConPara](https://huggingface.co/SES-Lab-OTH/deberta-conpara); [DetectRL](https://arxiv.org/abs/2410.23746)). Our 150-token floor is about 110 words, which is reasonable; we should report FPR by length bucket rather than raise it blindly.

**Mitigations.**
1. Measure FPR per subgroup and show it next to the verdict.
2. Calibrate thresholds on human text of the same type.
3. Mine hard negatives during fine-tuning (Pangram).
4. Keep the inconclusive band.
5. Never pitch the tool for discipline. The Ghostbuster authors "strongly discourage incorporation of Ghostbuster into any systems that automatically penalize students" and list non-native writing as a known failure case ([Ghostbuster](https://github.com/vivek3141/ghostbuster)).

**Human-text sets for FPR.**

| Set | Contents | Licence |
|---|---|---|
| ELLIPSE ([GitHub](https://github.com/scrosseye/ELLIPSE-Corpus)) | About 6,500 essays by English-language learners, grades 8–12, with demographics | CC BY 4.0 |
| PERSUADE 2.0 ([GitHub](https://github.com/scrosseye/persuade_corpus_2.0)) | Over 25,000 essays, grades 6–12. It is the Kaggle competition's base, so check that none of the classifiers trained on it | CC BY 4.0 |
| Liang TOEFL + ASAP essays | 91 + 88 essays | MIT |
| ASAP (Kaggle) | Student essays | Competition terms, not checked |
| TOEFL11 | Learner essays | LDC-licensed, paid; not checked |

Add post-2023 human text from the deployment domain.

## 8. Text watermarks

Not applicable as a general detector.
- **SynthID-Text** is in Transformers v4.46+ (`SynthIDTextWatermarkingConfig`). Detection needs the private watermarking keys and a Bayesian detector trained for that configuration on about 10k or more examples. Google says the keys "should be stored securely and privately", and confidence drops after thorough rewriting or translation ([HF blog](https://huggingface.co/blog/synthid-text); [Nature 2024](https://doi.org/10.1038/s41586-024-08025-4)).
- **Kirchenbauer-style green-list watermarks** likewise need the hashing key to detect ([Kirchenbauer et al.](https://arxiv.org/abs/2301.10226)).
- **Black-box watermark research** asks whether a *deployed model* watermarks, by querying it. It can't tell whether a given text carries a mark ([Gloaguen et al., ICLR 2025](https://arxiv.org/abs/2405.20777)).
- **Paraphrasing attacks.** Recursive paraphrasing defeats watermarks, and watermarks can be spoofed ([Sadasivan et al.](https://arxiv.org/abs/2303.11156)).

*Inference:* the useful action is to link out to vendor verifiers, as the image report recommends. PAN 2026 runs a separate text-watermarking task ([PAN 2026](https://arxiv.org/abs/2602.09147)).

## 9. Evaluation methodology

- **Datasets with recent generators.**

  | Dataset | Generators | Caveat |
  |---|---|---|
  | PAN 2025 training data ([Zenodo](https://zenodo.org/records/14962653)) | GPT-4o, o1, o3-mini, GPT-4.5, Gemini 1.5/2.0, DeepSeek-R1-distill, Llama-3.1/3.3, Falcon3, Ministral | Test set kept confidential |
  | MIRAGE ([DetectAnyLLM](https://arxiv.org/abs/2509.14268)) | GPT-4o, o3-mini, Claude-3.5/3.7, Gemini-2.0, DeepSeek-V3/R1, Qwen-2.5, Llama-3.1, Grok-2; about 93K pairs | Data licence not stated; code non-commercial |
  | GPTZero's benchmark ([paper](https://arxiv.org/abs/2602.13042)) | GPT-5.2, Gemini 3 Pro, Claude Sonnet 4.5, Grok 4 Fast | Release status not found |
  | WasItAIGenerated 2026-09 ([HF](https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09)) | 2,572 rows; 66 generator labels and 4 humanizers | Vendor-made, per-row licences, many human rows not redistributed; generator labels **unverified** |

  Nothing open covers Llama 4, Qwen3 or the newest Claude and GPT models with matched human text. Generate our own (Muse section).
- **Metrics.**
  - Report TPR at 1% and 5% FPR with 95% bootstrap CIs. Resample human and AI rows separately and re-derive the threshold in each resample, because threshold noise dominates at small human counts (our recommendation).
  - Also report AUROC, expected calibration error, the inconclusive rate, and FPR per subgroup (ELL, native student, formal, short, long).
  - Tufts et al. show high AUROC can coexist with **0%** TPR at 1% FPR ([NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.271/)). Stowe & Patil show F1 moves with class balance while AUROC and TPR@FPR stay stable ([paper](https://arxiv.org/abs/2604.16607)).
- **How big.** By the rule of three, zero false positives in n human texts bounds FPR below about 3/n at 95% confidence ([rule of three](https://en.wikipedia.org/wiki/Rule_of_three_(statistics))). Our Wilson calculations:

  | Human texts | 95% FPR bound with zero false positives | 95% CI if 1% are flagged |
  |---|---|---|
  | 300 | Below about 1.3% | — |
  | 1,000 | Below about 0.4% | 0.54–1.83% |

  On the AI side, 500 texts with TPR 0.80 give an interval of about ±3.5 points.

  *Recommendation:* at least 1,000 human texts overall and 300 or more per subgroup whose FPR we quote, plus at least 500 AI texts across 5 or more generator families and at least 200 attacked variants. Hold out a test split never used for fitting, and keep a frozen copy for regression. The build guide's 300+300+100 plan ([§10](../ai-detector-meta-hackathon.md)) is enough for a 1% FPR estimate only with wide error bars.

## 10. Can Muse replace Llama?

**Verdict.** Muse cannot replace the Llama pair's logit-based scoring today:
- **Spark and Wonder** reject `logprobs` on the Model API, and reasoning can't be turned off (tested by us 2026-11-03).
- **There is no completions/echo endpoint**, so the API can't score a given text's tokens.
- **Glimmer**, the only Muse model we can run locally, has no base checkpoint.
- **Spark Open** isn't on the Model API and is far too large for our hardware.

The pair has fusion weight 0, so dropping it costs no current accuracy. It would remove the GLTR view and the zero-shot features that the stacking evidence says help in the worst case (section 4). Muse newly enables three things: cheap fresh generator data for training and calibration, paraphrase and "humanize" attacks for robustness tests, and opt-in black-box checks for whether a text was written by Muse.

- **Muse Model API.** Muse Spark is a reasoning model. `logprobs` is "not supported" and returns 400, `top_logprobs` requires it, `n` must be 1, `stop` and `logit_bias` return 400, and no legacy completions or echo endpoint is documented ([chat-completions docs](https://dev.meta.ai/docs/protocols/chat-completions)). `muse-wonder-1.0` costs $4/$20 per 1M input/output tokens with a 1M context, and its page says nothing about logprobs ([Wonder page](https://dev.meta.ai/models/muse-wonder); [model card](https://research.meta.ai/static/muse-wonder-model-card/)).

  **Tested by us 2026-11-03** with the project's Model API key:

  | Probe | Result |
  |---|---|
  | `GET /v1/models` | Lists `muse-wonder-1.0`, `muse-spark-1.3` (and `-contributor`), `muse-spark-1.2` (and `-contributor`), `muse-spark-1.1`, `muse-voice-transcribe-1.0`, `sam-3.1` and `muse-image-1.0` |
  | `logprobs=true` on `muse-spark-1.3` and `muse-wonder-1.0` | HTTP 400: "`logprobs` is not supported with reasoning" |
  | Same, with `reasoning_effort` at `minimal` or `low` | 400: "`logprobs` cannot be true when `reasoning_effort` is `minimal`" |
  | `reasoning_effort=none` | Rejected; only `minimal` to `max` are accepted, so reasoning can't be turned off |
  | `POST /v1/completions` (echo / prompt logprobs) | 404 on both models; the endpoint doesn't exist |

  So no Muse text model on the API returns logprobs at all. Even generated-token logprobs would not have been enough: scoring a user's text needs the logprobs of the *prompt* tokens. Glimpse got those through OpenAI's legacy Completion API, and its README says the demo is now "limited to davinci and babbage models" because that API was discontinued for GPT-3.5 and GPT-4 ([Glimpse README](https://github.com/baoguangsheng/glimpse); [Glimpse, ICLR 2025](https://arxiv.org/abs/2412.11506)). **Glimpse is therefore not possible through the Muse API.** Asking the model to repeat the text and reading the copy's logprobs would measure copying, not authorship (our reasoning). Through the API, only black-box methods and data generation remain. Logit methods need local weights (Llama, or Glimmer as a single-model Fast-DetectGPT) or a third-party host of Spark Open that returns prompt logprobs.
- **Muse Glimmer-30B (Apache-2.0, open weights).** The current Hugging Face listing still has no base checkpoint. The org has only `Muse-Glimmer-30B`, `-GGUF`, `-ExecuTorch-PTE` and `-assistant` ([HF](https://huggingface.co/meta-models/Muse-Glimmer-30B)). The `-assistant` model is a DFlash block-diffusion drafter conditioned on hidden features from layers {1, 13, 25, 37, 49} of the main model, not a standalone LM ([drafter card](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant)), so it can't act as an observer. Options:
  - **Single-model Fast-DetectGPT, with Glimmer as both sampling and scoring model.** It runs in one forward pass. In the paper, same-model sampling and scoring averaged 0.9925 AUROC vs 0.9945 with GPT-J sampling ([Fast-DetectGPT Table 8](https://arxiv.org/html/2310.05130v3)), so losing the pair costs little in that setup. Binoculars' authors found same-model Binoculars "competitive" but below a base/instruct pair ([§A.5](https://arxiv.org/html/2401.12070v3)).
  - **Fit on our hardware.** The card targets 64 GB VRAM at full precision and 24–32 GB at about 4-bit ([card](https://huggingface.co/meta-models/Muse-Glimmer-30B)). *Inference:* 8-bit (about 30 GB) fits the L40S, and the 17 GB K-quant fits the 32 GB Mac, if the runtime returns full-vocabulary logits. Quantised scores need their own calibration.
  - **No published detection results exist for Glimmer.** *Inference:* because it is distilled from Muse Spark, it should be strongest on Muse-generated text and is unproven on GPT, Claude or Gemini text. The pair is worth an A/B against Llama-3.1-8B on the new eval set, not a blind swap.
  - **A Binoculars pair is impractical.** Glimmer's vocabulary is 202,048 tokens with BOS id 200000, the same size as Llama 4 Scout's ([Glimmer config](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/config.json); [Scout config mirror](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E/blob/main/config.json)). That suggests a shared tokenizer, but the token mappings are **unverified**, and Scout's roughly 109B parameters won't fit a 48 GB card (our estimate).
- **Muse Spark Open.** It is run "yourself or through a hosted provider, not on Model API". BF16 and NVFP4 checkpoints exist, and it is "derived from Muse Spark ... with additional safety post-training". No base checkpoint is mentioned ([overview](https://dev.meta.ai/docs/overview); [model page](https://dev.meta.ai/models/muse-spark-open/); [HF collection](https://huggingface.co/collections/meta-models/muse-spark-open)). Our 2026-11-03 probe confirms it is absent from the Model API: `/v1/models` doesn't list it, and the guessed IDs `muse-spark-open`, `muse-spark-open-1.0` and `Muse-Spark-Open-816B-A42B` return 404 "model not found". The name implies 816B total and 42B active parameters. *Inference:* even NVFP4 weights are about 400 GB, so self-hosting needs a multi-GPU node, far beyond the $100 Modal credit for an always-on demo. Provider support for **prompt-token** logprobs:

  | Route | Prompt-token logprobs? | Cap |
  |---|---|---|
  | Self-hosted vLLM | Yes. `prompt_logprobs=-1` returns all vocabulary logprobs ([vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html)) | Full vocabulary; the only route that can |
  | Together | Its completions API documents `echo` "with logprobs to return prompt logprobs" ([Together](https://docs.together.ai/reference/completions-1)). The Spark Open page shows only a chat example and a "Launching soon" banner ([model page](https://www.together.ai/models/muse-spark-open)) | Top-20 |
  | Fireworks | Completions has `echo` / `echo_last`, "useful for obtaining logprobs of the prompt suffix" ([Fireworks](https://docs.fireworks.ai/api-reference/post-completions)) | Top-5 by default |
  | OpenRouter | Documents output-token logprobs only, nothing on echo ([OpenRouter](https://openrouter.ai/docs/api-reference/parameters)) | `top_logprobs` ≤ 20 |
  | Groq | No: `logprobs` and `top_logprobs` return 400 ([Groq](https://console.groq.com/docs/openai)) | — |

  Whether any third-party provider enables echo for *this* model is **untested**; our probe covered only Meta's own API. Even then:
  - **Top-5 or top-20 only.** You get a Glimpse-style approximation of the full distribution, from a single post-trained model with no pair.
  - **Credits don't apply.** Third-party aggregators list $1.25/$4.25 per 1M tokens (**unverified** beyond Together's page), and Meta Model API credits don't pay for these providers.
  - **Privacy.** User text goes to yet another processor.
- **Black-box methods that need only generated text (opt-in).**

  | Method | Reported result | Calls per document | Cost per document, Spark / Wonder (our estimate, before reasoning tokens) |
  |---|---|---|---|
  | Raidar rewrite distance ([ICLR 2024](https://arxiv.org/abs/2401.12970)) | GPT-3.5-turbo rewriter plus a logistic regression trained on edit-distance features. In-domain F1: essays 64.8, news 60.3, code 95.4. Out-of-domain F1: 47–59. Claude student essays 57.8 F1. Evasive prompts drop single-prompt models to 25–34 F1. Author-reported, needs training data; the repo has no licence | 1–3 | About $0.003 / $0.015 |
  | DNA-GPT black-box ([paper](https://arxiv.org/abs/2305.17359)) | Same-model regeneration, K=10, γ=0.5: GPT-4 Reddit AUROC 99.3 with 91% TPR at 1% FPR; XSum 30% TPR. It only tests whether *the regenerating model* wrote the text, so on Muse it means "was this Muse?" | K=10 (n=1 means 10 calls) | About $0.017 / $0.074 |
  | LLM-as-judge | Prompted frontier models scored **53–73%** accuracy (GPT-5.2 0.664, Gemini-3-Pro 0.731, Claude Sonnet 4.5 0.711, Qwen3-Max 0.530) vs 0.953 for a 1.5B fine-tuned detector ([READER, 2026](https://arxiv.org/html/2605.25281v2)). ChatGPT labelled only 2 of 2,000 ChatGPT news texts as AI; GPT-4 misclassified over 95% of human TuringBench texts as AI ([Bhattacharjee & Liu](https://arxiv.org/abs/2308.01284)) | 1 | About $0.001–0.005 |

  Prices are $1.25/$4.25 per 1M for Spark ([build guide §4](../ai-detector-meta-hackathon.md)) and $4/$20 for Wonder. The estimates assume a 600-token essay; reasoning-token billing is unknown and could double them. *Verdict:* LLM-as-judge shouldn't vote; Muse Spark is already the report writer, not the judge. Raidar and DNA-GPT could be weight-0 evidence for "Muse-written" claims only.
- **Muse as a data generator: the best use of the credits.** About 1,000 essays of 600 tokens cost about $2.80 on Spark or about $12.80 on Wonder (our estimate). The evidence that in-generator and in-domain data helps:
  - Fakespot added GPT-4o, Llama-3.x, Qwen-2.5 and others to its training set ([data README](https://github.com/FakespotAILabs/ApolloDFT/blob/main/data/README.md)).
  - The Kaggle winner generated in-domain student essays ([code](https://github.com/rbiswasfc/llm-detect-ai)).
  - M4GT says detection "requires training data from the same domain and generators" ([M4GT](https://arxiv.org/abs/2402.11175)).
  - desklib's RAID TPR ranges from 0.654 to 0.998 across generators (our breakdown).

  Use Muse Spark, Wonder and local Glimmer plus open models (Llama-3.x, Qwen3, OLMo) on PERSUADE- and ELLIPSE-matched prompts. Then have Muse paraphrase and "humanize" a slice for attacked variants. Keep each generator family out of training in turn, to test unseen-model generalisation, and don't let Muse text dominate (our recommendation). Meta's API terms on using outputs to train a classifier were **not checked**; read them first.
- **Privacy.** Every black-box method sends the full user text to Meta, and provider-hosted Spark Open sends it to a third party. That conflicts with the repo's no-third-party-API stance for user text, so treat it as opt-in only, behind the existing `send_to_muse` consent flag, and never as a default verdict input. Generating training and eval data involves no user text and has no such issue.

## Limits the tool must state

- **Theoretical ceiling.** Sadasivan et al. bound the AUROC of any detector by the total-variation distance between human and AI text distributions (roughly 1/2 + TV − TV²/2). Recursive paraphrasing pushes TV down ([TMLR](https://arxiv.org/abs/2303.11156)). Chakraborty et al. counter that detection stays possible unless the distributions are identical, but the number of samples needed grows as AI text approaches human quality ([paper](https://arxiv.org/abs/2304.04736)). Single short documents are therefore the hard case by construction.
- **Known failure modes from the sources above.** Non-native and professionally edited prose, academic writing, short texts, paraphrase and humanizers, Unicode tricks, mixed or edited documents, new generators, and memorised famous texts: Binoculars put the US Constitution on the machine side ([paper §5.3](https://arxiv.org/html/2401.12070v3)).
- **Authors' warnings.** Binoculars' authors "strongly caution against using Binoculars (or any detector) without human supervision" ([README](https://github.com/ahans30/Binoculars)). Ghostbuster's authors discourage automatic penalties for students ([README](https://github.com/vivek3141/ghostbuster)). ConPara says a flag on a student essay "is not evidence of misconduct" ([card](https://huggingface.co/SES-Lab-OTH/deberta-conpara)). OpenAI withdrew its own classifier for low accuracy ([build guide ref 48](../ai-detector-meta-hackathon.md)).

## Ranked implementation shortlist

> **Decision, 2026-11-03:** every Muse option is dropped (judge, rewrite distance, Muse-generated data, Glimmer). The eval set's AI texts come from the PAN'25 validation split instead (`scripts/fetch_text_eval.py`), and paraphrase attacks from PAN's own paraphrased subsets. Items 1–4 and 7 are implemented; 5, 6 and 8 wait on the eval results.

1. **Build the eval set first.**
   - *What:* held-out, uncontaminated, with subgroups and bootstrap CIs. Human texts: ELLIPSE, PERSUADE 2.0, the Liang TOEFL and ASAP essays, formal/technical prose and post-2023 web text. AI texts: Muse Spark, Wonder and Glimmer, Llama-3.x, Qwen3 and others on matched prompts. Attacked texts: Muse paraphrase, homoglyph, zero-width and whitespace variants.
   - *Why first:* every later item needs it. Current numbers bound FPR only below about 5.6% and are probably in-sample for both classifiers.
   - *Cost:* 1–2 days plus under $20 of Muse credit.
   - *Files:* `scripts/eval_images.py`. Add a `--split` held-out mode, bootstrap CIs, per-subfolder subgroup tables, and threshold-at-FPR output, and stop fitting Platt on the reported rows.
   - *Measure:* TPR at 1% and 5% FPR with 95% CIs per subgroup, plus FPR on ELL and native essays.
2. **Normalise input text and apply Fakespot's `clean_text`.**
   - *What:* NFKC, strip zero-width and bidi characters, map confusables, collapse whitespace. Record the counts of removed or mapped characters as evidence.
   - *Evidence:* the RAID per-attack drops (Binoculars homoglyph 0.79 → 0.36; desklib zero-width 0.95 → 0.88), SilverSpeak, ConPara, and Fakespot's card.
   - *Cost:* hours. *Files:* `router.py` (normalise once), `text.py` (`clean_text` before the Fakespot tokenizer).
   - *Measure:* clean vs attacked eval variants. The gap should close.
3. **Calibrate and threshold at a fixed FPR.**
   - *What:* fit Platt per detector on the held-out split, including `txt.fakespot` and `txt.desklib` (isotonic once there are over 1,000 examples per class). Derive "likely AI" from the human-score quantile at 1% FPR, per text type, and set the inconclusive band from calibrated probabilities.
   - *Evidence:* sklearn guidance, and RAID's 0.72–0.98 per-domain thresholds for desklib.
   - *Cost:* about a day after item 1. *Files:* `scripts/eval_images.py`, `fusion.yaml` (`platt:`, bands), `fusion.py` (per-type thresholds if adopted).
   - *Measure:* expected calibration error, realised FPR on held-out humans within the CI of the target.
4. **Logistic stacker replacing the weighted mean and disagreement veto.**
   - *Inputs:* both classifier logits, Binoculars, Fast-DetectGPT, LRR from the same pass, token count and normalisation flags.
   - *Evidence:* Apollo's worst-case 0.31 → 0.95, DivEye +18.7% as an auxiliary signal, and complementary failures on RAID attacks.
   - *Cost:* 1–2 days. *Files:* `fusion.py` (stacker branch), `fusion.yaml` (coefficients), `scripts/eval_images.py` (cross-validated fit and dump).
   - *Measure:* TPR at 1% FPR vs the current rule on the held-out set, plus the inconclusive rate.
5. **Trial one stronger open classifier.**
   - *Candidates:* MELD (MIT; RAID 0.992 at 1% FPR; ships a 1% FPR threshold; anonymous-review provenance), DeBERTa-ConPara (MIT; best cross-dataset; custom loader), Tabularis ModernBERT (MIT; 149M; fast).
   - *Evidence:* RAID leaderboard (in-distribution) and ConPara's cross-dataset table (author-run). Nothing on 2026 generators.
   - *Cost:* about a day per model. *Files:* `text.py` (`CLASSIFIERS` and loaders), `fusion.yaml`, `NOTICE`.
   - *Measure:* subgroup TPR and FPR on the new eval set. Keep a model only if it adds TPR at fixed FPR in the stacker.
6. **Re-test the zero-shot path on the L40S with better pairs before deciding its fate.**
   - *Pairs:* Llama-3.1-8B (current default), gemma-2-9b (licence flag), Qwen2.5-7B, and single-model Glimmer Fast-DetectGPT.
   - *Evidence:* AdaDetectGPT's about 0.95 AUROC with gemma-2-9b on GPT-4o and Claude-3.5 text; PAN's 0.827 and 26% FPR for Llama-3.1-8B; GPTZero's vendor-run collapse on 2026 generators.
   - *Cost:* about a day. *Files:* `text.py` (env vars; drop the Falcon-centred `BINOCULARS_PRIOR` once Platt exists), `fusion.yaml`.
   - *Measure:* the gain each score adds as a stacker feature. Vote weight stays 0 unless it helps at 1% FPR.
7. **Windowing for long documents.**
   - *What:* 512-token windows with stride 256 for all detectors, fused on the mean logit, with a per-window heat map in the UI.
   - *Evidence:* classifiers currently read only the first 512 or 768 tokens; length effects in Fast-DetectGPT and DetectRL; DNA-GPT's window variant. No paper compares aggregation rules.
   - *Cost:* about a day. *Files:* `text.py`, `router.py` (keep the 100k-character cap).
   - *Measure:* long-document and human-then-AI concatenation subsets; FPR of the aggregate.
8. **Fine-tune our own detector.**
   - *What:* QLoRA on Llama-3.1-8B or Qwen3-8B, or a DeBERTa-v3-large / Ettin encoder. Train on RAID, MAGE, HC3 and M4 plus fresh Muse and open-model generations, with homoglyph and paraphrase augmentation on raw (unnormalised) training text. Select models on a held-out generator family.
   - *Evidence:* PAN 2025 winner (0.995 ROC-AUC at 0.6% FPR), Kaggle 1st place, and SemEval's "best systems used LLMs". ImBD or DDL-style performer tuning is the low-data variant.
   - *Cost:* about a week plus L40S time. *Files:* a new training script plus `text.py` and `fusion.yaml`.
   - *Measure:* leave-one-generator-out TPR at 1% FPR and subgroup FPR.
9. ~~**Opt-in Muse black-box evidence**~~ (dropped, see the decision above), weight 0 and behind `send_to_muse`: Raidar-style rewrite distance, or DNA-GPT "was this Muse?".
   - *Evidence:* author-reported F1 of 47–65 on essays and news out-of-domain for Raidar; DNA-GPT works only for the regenerating model.
   - *Cost:* about a day plus about $0.003–0.07 per document. *Files:* `text.py`, `fusion.yaml`.
   - *Measure:* the same eval set, reported separately because it sends user text.

Skipped on the evidence:
- LLM-as-judge verdicts (53–73% accuracy).
- Compression scores (AUROC 0.64–0.77).
- PHD, DetectGPT and NPR (weak or slow).
- RADAR (RAID 0.64).
- Licence-blocked methods: DivEye, DetectAnyLLM, mdok, Linda-RAID, AdaDetectGPT, Lastde, DeTeCtive.
- Watermark detection (needs keys we don't have).
- Spark Open echo scoring (huge model, top-k only, third-party data flow).

## Sources

1. RAID leaderboard and raw result files: https://raid-bench.xyz/leaderboard ; https://raid-bench.xyz/data/submissions/_index.json ; repo (MIT) https://github.com/liamdugan/raid ; paper, Dugan et al., ACL 2024, https://arxiv.org/abs/2405.07940
2. Hans et al., "Spotting LLMs With Binoculars", ICML 2024: https://arxiv.org/abs/2401.12070 ; code (BSD-3) https://github.com/ahans30/Binoculars
3. Bao et al., "Fast-DetectGPT", ICLR 2024: https://arxiv.org/abs/2310.05130 ; README (MIT, note dated 2026-01-31) https://github.com/baoguangsheng/fast-detect-gpt
4. Fakespot RoBERTa card (Apache-2.0): https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1 ; `clean_text`: https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/blob/main/utils.py ; Apollo report: https://github.com/FakespotAILabs/ApolloDFT/blob/main/apollo-text.md ; data list: https://github.com/FakespotAILabs/ApolloDFT/blob/main/data/README.md
5. desklib/ai-text-detector-v1.01 card (MIT): https://huggingface.co/desklib/ai-text-detector-v1.01
6. Mady et al., "DeBERTa-ConPara", AACL-IJCNLP 2026: https://arxiv.org/abs/2610.00883 ; https://github.com/SES-Lab-OTH/deberta-conpara ; https://huggingface.co/SES-Lab-OTH/deberta-conpara
7. MELD (MIT): https://huggingface.co/anon-review-meld-2026/meld
8. Tabularis ModernBERT (MIT): https://huggingface.co/tabularisai/ai-text-detection
9. Other RAID checkpoints: https://huggingface.co/nakas/gpttrace-v5-1b ; https://huggingface.co/Lindarixon/Linda-RAID ; https://huggingface.co/Oxidane/tmr-ai-text-detector ; https://huggingface.co/GeorgeDrayson/modernbert-ai-detection-raid-mage ; https://huggingface.co/Shushant/ADAL-detector-large
10. Adam et al., "GPTZero: Robust Detection of LLM-Generated Texts", 2026 (vendor): https://arxiv.org/abs/2602.13042
11. Zhou et al., "AdaDetectGPT", NeurIPS 2025: https://arxiv.org/abs/2510.01268 ; https://github.com/Mamba413/AdaDetectGPT
12. Xu et al., "Lastde", 2024: https://arxiv.org/abs/2410.06072 ; https://github.com/TrustMedia-zju/Lastde_Detector
13. Chen et al., "Imitate Before Detect (ImBD)", AAAI 2025: https://arxiv.org/abs/2412.10432 ; https://github.com/Jiaqi-Chen-00/ImBD
14. Bao et al., "Glimpse", ICLR 2025: https://arxiv.org/abs/2412.11506 ; https://github.com/baoguangsheng/glimpse
15. Zeng et al., "DALD", 2024: https://arxiv.org/abs/2406.05232
16. Ma & Wang, "TOCSIN", EMNLP 2024: https://arxiv.org/abs/2409.16914 ; https://github.com/Shixuan-Ma/TOCSIN
17. BiScope, NeurIPS 2024: https://github.com/MarkGHX/BiScope
18. Yang et al., "DNA-GPT", ICLR 2024: https://arxiv.org/abs/2305.17359 ; https://github.com/Xianjun-Yang/DNA-GPT
19. Su et al., "DetectLLM", Findings EMNLP 2023: https://arxiv.org/abs/2306.05540 ; https://github.com/mbzuai-nlp/DetectLLM
20. Tulchinskii et al., intrinsic dimension (PHD), NeurIPS 2023: https://arxiv.org/abs/2306.04723 ; https://github.com/ArGintum/GPTID
21. Zippy: https://github.com/thinkst/zippy
22. Basani & Chen, "DivEye", TMLR 2026: https://arxiv.org/abs/2509.18880 ; https://github.com/IBM/diveye
23. Fu et al., "DetectAnyLLM" and MIRAGE, 2025: https://arxiv.org/abs/2509.14268 ; https://github.com/fjc2005/DetectAnyLLM
24. Khan & Andrews, "Multi-View Gaussian Processes", 2026: https://arxiv.org/abs/2606.14060
25. Wu et al., "DetectRL", NeurIPS 2024 D&B: https://arxiv.org/abs/2410.23746
26. Stowe & Patil, "Spotlights and Blindspots", 2026: https://arxiv.org/abs/2604.16607
27. Tufts, Zhao & Li, "A Practical Examination of AI-Generated Text Detectors", Findings NAACL 2025: https://aclanthology.org/2025.findings-naacl.271/
28. Bevendorff et al., PAN/ELOQUENT 2025 Voight-Kampff overview, CEUR Vol-4038: https://ceur-ws.org/Vol-4038/paper_277.pdf ; results: https://pan.webis.de/clef25/pan25-web/generated-content-analysis.html ; data: https://zenodo.org/records/14962653
29. Macko, "mdok of KInIT", PAN 2025: https://arxiv.org/abs/2506.01702 ; weights (CC-BY-NC-ND) https://huggingface.co/DominikMacko/mdok ; code (GPL-3.0) https://github.com/kinit-sk/mdok
30. PAN 2026 overview: https://arxiv.org/abs/2602.09147
31. Dugan et al., COLING 2025 GenAI Detection Task 3: https://arxiv.org/abs/2501.08913
32. Wang et al., COLING 2025 GenAI Detection Task 1: https://arxiv.org/abs/2501.11012
33. Wang et al., SemEval-2024 Task 8: https://arxiv.org/abs/2404.14183
34. Wang et al., "M4GT-Bench", ACL 2024: https://arxiv.org/abs/2402.11175
35. Li et al., "MAGE", ACL 2024: https://arxiv.org/abs/2305.13242 ; data (Apache-2.0) https://huggingface.co/datasets/yaful/MAGE
36. Kaggle LLM-Detect 1st place: https://github.com/rbiswasfc/llm-detect-ai ; https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121
37. He et al., "DeTeCtive", NeurIPS 2024: https://arxiv.org/abs/2410.20964 ; https://github.com/heyongxin233/DeTeCtive ; "DETree", NeurIPS 2025: https://arxiv.org/abs/2510.17489 ; https://github.com/heyongxin233/DETree
38. Emi & Spero, Pangram technical report (vendor): https://arxiv.org/abs/2402.14873
39. Masrour, Emi & Spero, "DAMAGE" (vendor): https://arxiv.org/abs/2501.03437
40. Pu et al., "On the Zero-Shot Generalization of Machine-Generated Text Detectors": https://arxiv.org/abs/2310.05165
41. Wang et al., "SeqXGPT", EMNLP 2023: https://arxiv.org/abs/2310.08903 ; https://github.com/Jihuai-wpy/SeqXGPT
42. Dycke et al., "Your AI Text is not Mine" (AITDNA), 2026: https://arxiv.org/abs/2606.04906
43. Krishna et al., "Paraphrasing evades detectors...", NeurIPS 2023: https://arxiv.org/abs/2303.13408
44. Creo & Pudasaini, "SilverSpeak", GenAIDetect 2025: https://arxiv.org/abs/2406.11239
45. Hu et al., "RADAR", NeurIPS 2023: https://arxiv.org/abs/2307.03838 ; https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B
46. Liang et al., "GPT detectors are biased against non-native English writers", Patterns 2023: https://arxiv.org/abs/2304.02819 ; data (MIT) https://huggingface.co/datasets/WxWx/ChatGPT-Detector-Bias
47. Park, Jeong & Kim, "Style as a Confound", EMNLP 2026: https://arxiv.org/abs/2608.26710
48. ETS talk transcript, Y. Jiang (secondary): https://www.ets.org/aera-ncme-conference/transcript/yang-jiang.html
49. PERSUADE 2.0 (CC BY 4.0): https://github.com/scrosseye/persuade_corpus_2.0 ; ELLIPSE (CC BY 4.0): https://github.com/scrosseye/ELLIPSE-Corpus ; HC3 (CC-BY-SA-4.0): https://huggingface.co/datasets/Hello-SimpleAI/HC3
50. Verma et al., "Ghostbuster", NAACL 2024: https://arxiv.org/abs/2305.15047 ; https://github.com/vivek3141/ghostbuster
51. SynthID Text in Transformers: https://huggingface.co/blog/synthid-text ; Dathathri et al., Nature 634:818–823 (2024): https://doi.org/10.1038/s41586-024-08025-4
52. Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023: https://arxiv.org/abs/2301.10226
53. Gloaguen et al., "Black-Box Detection of Language Model Watermarks", ICLR 2025: https://arxiv.org/abs/2405.20777
54. Sadasivan et al., "Can AI-Generated Text be Reliably Detected?", TMLR: https://arxiv.org/abs/2303.11156
55. Chakraborty et al., "On the Possibilities of AI-Generated Text Detection": https://arxiv.org/abs/2304.04736
56. scikit-learn, Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
57. Guo et al., "On Calibration of Modern Neural Networks", ICML 2017: https://arxiv.org/abs/1706.04599
58. Zhu et al., "Multiscaled Conformal Prediction" (ACL 2025): https://arxiv.org/abs/2505.05084
59. Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction": https://arxiv.org/abs/2107.07511
60. Rule of three: https://en.wikipedia.org/wiki/Rule_of_three_(statistics) ; Wilson interval: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
61. Mao et al., "Raidar", ICLR 2024: https://arxiv.org/abs/2401.12970
62. Bhattacharjee & Liu, "Fighting Fire with Fire", SIGKDD Explorations 2023: https://arxiv.org/abs/2308.01284
63. Su et al., "READER", 2026: https://arxiv.org/abs/2605.25281
64. Meta Model API chat completions: https://dev.meta.ai/docs/protocols/chat-completions ; Muse Wonder: https://dev.meta.ai/models/muse-wonder ; model card: https://research.meta.ai/static/muse-wonder-model-card/ ; overview: https://dev.meta.ai/docs/overview ; live probe by us, 2026-11-03 (`/v1/models`, `logprobs` and `reasoning_effort` on `muse-spark-1.3` and `muse-wonder-1.0`, `/v1/completions`), results in §10
65. Muse Glimmer-30B (Apache-2.0): https://huggingface.co/meta-models/Muse-Glimmer-30B ; drafter: https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant ; https://dev.meta.ai/models/muse-glimmer/
66. Muse Spark Open: https://dev.meta.ai/models/muse-spark-open/ ; https://huggingface.co/collections/meta-models/muse-spark-open
67. Provider docs: Together https://docs.together.ai/reference/completions-1 and https://www.together.ai/models/muse-spark-open ; Fireworks https://docs.fireworks.ai/api-reference/post-completions ; OpenRouter https://openrouter.ai/docs/api-reference/parameters ; Groq https://console.groq.com/docs/openai ; vLLM https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html ; price aggregator (secondary) https://www.getmaxim.ai/bifrost/llm-cost-calculator/provider/fireworks_ai/model/muse-spark-open
68. Llama 4 Scout config (mirror): https://huggingface.co/unsloth/Llama-4-Scout-17B-16E/blob/main/config.json
69. WasItAIGenerated benchmark 2026-09 (vendor): https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09
70. Repo build guide, `ai-detector-meta-hackathon.md` (Muse Spark pricing §4; OpenAI classifier withdrawal ref 48; eval plan §10)
