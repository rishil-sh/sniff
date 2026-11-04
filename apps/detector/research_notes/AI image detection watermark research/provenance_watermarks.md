# Provenance signals and watermarking for AI-generated images (status as of Nov 2026)

Reliability note: many 2026 items below come from secondary sources (law-firm notes, news aggregators, vendor blogs). Primary pages for OpenAI (403) and Meta Seal (404 at the URL tried) could not be fetched. Items marked UNVERIFIED rest on one secondary source.

## 1. Google SynthID for images

### Takeaway

SynthID is a learned, post-hoc invisible watermark applied to Google outputs (and, since May 2026, OpenAI image outputs). Checking it is possible only through vendor-hosted tools: the Gemini app (Nov 2025) and the SynthID Detector portal. No open-source image decoder and no public API were found. Robustness is contested, with credible removal attacks reported.

### Cited Findings

- Gemini app image verification announced 2025-11-20. Users upload an image and ask "Was this created with Google AI?"; Gemini checks for SynthID and adds its own reasoning. Images only at launch, covers only Google-made or Google-edited content, gives no accuracy figures. Google said over 20 billion items watermarked. — [Google blog](https://blog.google/technology/ai/ai-image-verification-gemini-app/)
- Same post: Nano Banana Pro (Gemini 3 Pro Image) outputs in Gemini app, Vertex AI and Google Ads get embedded C2PA metadata starting that week; Google plans to read C2PA from other vendors, with no timeline. — [Google blog](https://blog.google/technology/ai/ai-image-verification-gemini-app/)
- Gemini help page: one file at a time, 100 MB max, approx 20 image checks and 10 video checks per day (per search summary). — [Google support](https://support.google.com/gemini/answer/16722517)
- SynthID Detector portal (synthid.com) reportedly opened to the public 2026-10-07, global, English. Partners listed: OpenAI, NVIDIA, Kakao (Apple "soon"). Reported stats: over 180 billion watermarked images and videos, over 1 million requests per day. Manual upload, one file at a time, no API mentioned. States it is "not a general AI content detector." Formats per Unite.ai (not Google): JPG, PNG, WEBP, HEIC. UNVERIFIED, single secondary source. — [Pillitteri summary](https://pasqualepillitteri.it/en/news/21499/google-synthid-detector-openai-nvidia-kakao)
- SynthID-Image paper (arXiv 2510.09263, submitted 2025-10-10, Gowal et al., Google DeepMind): deep-learning invisible watermark system, over 10 billion images and video frames watermarked at that time, verification service for trusted testers only; external variant SynthID-O compared with other post-hoc methods. Paper argues deterministic behaviour blocks collusion attacks and acknowledges that robustness to many threat models trades off against quality. — [arXiv](https://arxiv.org/abs/2510.09263)
- Removal attacks: MarkNull (arXiv 2608.10166) reports SynthID-Image resists a basic attack but fails against latent-space manipulation. — [MarkNull](https://arxiv.org/pdf/2608.10166) (via search summary, table only partly seen)
- "Reverse-SynthID" community spectral (FFT) attack claims about 91% watermark-energy removal on Nano Banana images with quality retained; Google disputes; not peer reviewed. — [Pillitteri](https://pasqualepillitteri.it/en/news/909/reverse-synthid-broken-google-ai-watermark-spectral-analysis)
- SynthID-Text is separate (open-sourced by Google in 2024; not re-verified here). No open image SynthID decoder found.

### Inferences

- A detector tool cannot call SynthID programmatically today; it can only link users to synthid.com or the Gemini app. A negative result means nothing about non-Google/OpenAI generators.
- Because OpenAI now uses SynthID, one portal covers both Google and OpenAI images.

### Gaps

- No official Google page fetched for the portal launch date or an API roadmap; false-positive rate not published.

## 2. C2PA / Content Credentials

### Takeaway

C2PA is a signed manifest (JUMBF box in the file) with actions and a "digitalSourceType" assertion. Adobe Firefly, OpenAI (conformant generator since May 2026) and Google (Nano Banana Pro, Imagen per secondary source) sign. Midjourney does not (per secondary sources). Manifests are easily stripped by re-encoding, screenshots and most social uploads, so absence proves nothing.

### Cited Findings

- OpenAI (announced about 2026-05-19): images from ChatGPT, Codex and API carry both a C2PA manifest and SynthID; OpenAI is a "C2PA Conforming Generator Product"; public verification preview at openai.com/verify checks for a trusted manifest or an OpenAI SynthID mark; no signal does not prove human-made. — [OpenAI](https://openai.com/index/advancing-content-provenance) (page 403; facts via [TNW](https://thenextweb.com/news/openai-c2pa-synthid-ai-image-detection-watermark) and [search results](https://letsdatascience.com/news/openai-adopts-c2pa-and-synthid-for-image-verification-ed2f7b5f/))
- DALL-E 3 added C2PA in 2023; Adobe Firefly attaches Content Credentials to all outputs since launch (2023); Google Imagen listed as signing; Midjourney does not embed C2PA; Stability hosted platform partial, open weights none. Source is a vendor article with no citations. — [C2PA Viewer](https://c2paviewer.com/articles/ai-tools-c2pa-support) (low reliability)
- One source claims Midjourney V8 (mid-2026) still has no manifest; Claude C2PA claims conflict between sources and are unconfirmed. — [search summary of tech-insider and others](https://machineherald-io.pages.dev/article/2026-03/10-the-ai-content-labeling-gap-regulation-arrives-before-the-infrastructure-is-ready)
- Open-source verifiers: C2PA Rust SDK `c2pa` (docs say v2 claims by default, min Rust 1.88), plus c2patool CLI and c2pa-python bindings built on it. — [CAI Rust SDK docs](https://opensource.contentauthenticity.org/docs/rust-sdk)
- Soft binding (watermark or fingerprint lookup to recover a manifest after stripping) exists in spec 2.x; details only from a weak source. — [MEXC](https://www.mexc.com/news/813273)

### Inferences

- Detector flow: read manifest (c2pa-python), validate the signature chain against the C2PA trust list, then read `c2pa.actions` digitalSourceType. A valid manifest is a strong positive; missing is uninformative.

### Gaps

- Microsoft and Meta signing status not established; current c2pa-python/c2patool versions not confirmed; metadata-stripping behaviour per social platform not sourced (IPTC article below says only that stripping happens).

## 3. IPTC / XMP / EXIF and generator-specific fields

### Takeaway

IPTC `DigitalSourceType` with value `trainedAlgorithmicMedia` ("Created using Generative AI") or `compositeWithTrainedAlgorithmicMedia` ("Edited with Generative AI") is read by Meta, Google and others. Trivially stripped or forged.

### Cited Findings

- Meta said Feb 2024 it reads IPTC DigitalSourceType to label images on Facebook, Instagram, Threads; Meta AI Imagine writes trainedAlgorithmicMedia. — [IPTC](https://iptc.org/news/2024/02)
- Labels clarified Oct 2024 ("Created using Generative AI"); composite value is for inpainting/outpainting edits; Google Search reads it. — [IPTC](https://iptc.org/news/2024/10), [SEJ](https://www.searchenginejournal.com/google-search-now-supports-labeling-ai-generated-or-manipulated-images/521675)
- Meta renamed "Made with AI" to "AI info" on 2024-07-01 after false labels on retouched photos. — [PPC Land](https://ppc.land/iptc-metadata/)
- Stable Diffusion front ends (A1111/ComfyUI) write PNG tEXt chunks such as `parameters` / `prompt`: from my background knowledge, NOT sourced in this session.

### Gaps

- No sourced list of per-generator EXIF/XMP fields; not verified.

## 4. Invisible watermarks (open and closed)

### Takeaway

Only some schemes have public decoders. DwtDct (SD/SDXL default) is readable by anyone; Meta Seal and Tree-Ring/Gaussian Shading/Stable Signature are open research code; SynthID and Amazon Titan are closed. Learned watermarks are generally attackable by regeneration.

### Cited Findings

- invisible-watermark `dwtDct` embeds a fixed message in the SD/SDXL diffusers pipeline (skipped for images under 256 px); decoder with the same settings should read it. — [HF diffusers watermark.py](https://huggingface.co/svjack/diffusers-sdxl-controlnet/blame/3416f50ef53a323a21c9fbb0209d2cf6aa38c1f7/src/diffusers/pipelines/stable_diffusion_xl/watermark.py)
- Amazon Titan Image Generator: invisible watermark on all outputs; detection via Bedrock console and DetectGeneratedContent API returning confidence; proprietary, confidence drops after modification (Apr 2024). — [AWS](https://aws.amazon.com/about-aws/whats-new/2024/04/watermark-detection-amazon-titan-image-generator-bedrock/)
- Meta Seal: open-source framework for audio, image, video, text watermarking, open weights, MIT license. Video Seal released Dec 2024; Pixel Seal is the newer image/video model. — [Meta Seal](https://facebookresearch.github.io/meta-seal), [Pixel Seal paper](https://pierrefdz.github.io/assets/publis/pixelseal/paper.pdf), [Maginative](https://www.maginative.com/article/meta-releases-video-seal-an-open-source-watermarking-tool-for-ai-video/)
- Stable Signature (fine-tuned VAE decoder) is highly vulnerable to regeneration/alternate VAE; Tree-Ring (zero-bit, Fourier-latent) is more robust to regeneration but black-box forgery exists; Gaussian Shading is distribution-preserving and multi-bit. — [Invisible Watermarks: Attacks and Robustness](https://arxiv.org/html/2412.12511v1), [Black-Box Forgery](https://arxiv.org/pdf/2412.03283)
- Tree-Ring, Gaussian Shading and Stable Signature require the generating model or inversion to detect (background knowledge, not re-sourced). They are only useful if the image came from a known deploying pipeline.

### Gaps

- TrustMark (Adobe) details and licensing not found; Microsoft InvisMark mentioned only in one search snippet (github.com/microsoft/InvisMark); no real-world deployment list of who uses which open watermark.

## 5. 2025-2026 regulation

### Takeaway

EU Article 50 marking duties applied from 2026-08-02, with a final Code of Practice (2026-06-10) calling for multi-layer marking. China's rules (since 2025-09-01) mandate implicit metadata labels plus visible labels.

### Cited Findings

- EU final Code of Practice on marking and labelling published 2026-06-10; requires at least two layers of machine-readable marking where needed (e.g., metadata plus watermark); providers should offer detection mechanisms; voluntary; transparency obligations effective 2026-08-02. — [Jones Day](https://www.jonesday.com/en/insights/2026/06/european-commission-publishes-final-code-of-practice-on-marking-and-labelling-aigenerated-content)
- Commission reportedly found the Code an adequate compliance tool on 2026-07-09 (single report; adequacy was still "being assessed" per Jones Day in June). — [Let's Data Science](https://letsdatascience.com/news/eu-endorses-ai-generated-content-transparency-code-f9326b9c)
- China: Measures for Labeling AI-Generated Synthetic Content and mandatory standard GB 45438-2025 effective 2025-09-01. Explicit visible labels plus implicit labels in file metadata (provider, content ID, etc.); watermarks encouraged not required; platforms must check implicit labels. — [Loeb](https://www.loeb.com/en/insights/publications/2025/03/chinas-ai-labeling-measures-and-mandatory-national-standards-take-effect-september-1), [Technology's Legal Edge](https://www.technologyslegaledge.com/2025/03/china-released-new-measures-for-labelling-ai-generated-and-synthetic-content/)

### Inferences

- Expect more Chinese-origin images to carry GB 45438 metadata fields, and EU-market generators to add metadata plus watermark; a detector should read both.

### Gaps

- Exact GB 45438 metadata key names (XMP/EXIF field) not found; official CAC text not read.
