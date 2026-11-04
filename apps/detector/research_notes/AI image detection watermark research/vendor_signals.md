# Vendor provenance/watermark/metadata signals by AI image generator (status as of Nov 2026; research was shallow on several vendors)

## 1. Per-generator signals (visible / invisible watermark / C2PA / EXIF-XMP-PNG markers)

### Takeaway
Well-documented by primary sources: OpenAI (C2PA + SynthID), Google (SynthID; C2PA on Nano Banana Pro), Black Forest Labs (C2PA on API/Playground only), Adobe (C2PA), Microsoft (C2PA). Poorly documented or only community/vendor-of-removal-tool sourced: Grok, Midjourney, Meta, Stable Diffusion API, Qwen/Wan, Seedream. No usable sources found for Ideogram, Leonardo, Recraft.

### Cited Findings
- **OpenAI (ChatGPT, Codex, API):** images include both C2PA metadata and SynthID watermarks (newest help-center version); visible disclosure only if the user prompts for it. — [OpenAI help 8912793](https://help.openai.com/en/articles/8912793-c2pa-in-chatgpt-images) (as summarized by search; page not fetched directly)
- OpenAI C2PA manifest sits in a JUMBF container with `digitalSourceType` = `trainedAlgorithmicMedia` (PixelPeeper, secondary). — [PixelPeeper](https://pixelpeeper.com/detect-ai-images)
- Third-party removal-tool page claims gpt-image-2 = SynthID + content-specific pixel watermark + C2PA, and DALL-E 3 = C2PA + metadata strip. Community/commercial claim, unverified. — [remove-ai-watermarks on PyPI](https://pypi.org/project/remove-ai-watermarks/0.4.1/)
- **Google Gemini / Imagen / Nano Banana:** SynthID invisible watermark on all outputs; Imagen `addWatermark` defaults true and cannot be disabled in the Gemini Developer API; Imagen models deprecated, shutdown as early as June 2026. — [Firebase ImagenGenerationConfig](https://firebase.google.com/docs/reference/js/ai.imagengenerationconfig)
- Nano Banana Pro (Gemini 3 Pro Image) outputs in Gemini app, Vertex AI and Google Ads carry embedded C2PA metadata. — [Neowin on Gemini verification](https://www.neowin.net/news/the-gemini-app-now-tells-you-if-an-image-is-ai-generated-or-not/); Google blog: [How we're bringing AI image verification to the Gemini app](https://blog.google/technology/ai/ai-image-verification-gemini-app/)
- Vertex AI has a Content Credentials doc page (not read in detail). — [Vertex AI Content Credentials](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/content-credentials)
- **Black Forest Labs FLUX:** every image from the BFL API and Developer Playground carries a signed C2PA manifest (AI-generative assertion, claim generator, creation timestamp, model name/version, SHA-256 hash; no prompt or account data). Optional soft-binding invisible watermark for recovering credentials after stripping. Self-hosted/open-weight and third-party inference: BFL does not attach or validate credentials. — [BFL help: C2PA in FLUX-generated images](https://help.bfl.ai/articles/3705384267-c2pa-in-flux-generated-images)
- **Midjourney:** committed to IPTC DigitalSourceType marker (`trainedAlgorithmicMedia`) — [IPTC news](https://iptc.org/news/midjourney-and-shutterstock-ai-sign-up-to-use-of-iptc-digital-source-type-for-generated-ai-content/). A vendor-run guide says (early 2026) no C2PA, generation info goes in EXIF/IPTC — [fast.io guide](https://www.fast.io/resources/ai-generated-image-metadata-detection-tools.md) (low-quality source). PixelPeeper says Discord output "does not include provenance data" — [PixelPeeper](https://pixelpeeper.com/detect-ai-images). These two sources conflict (IPTC tag present vs. none). No source found confirming a Midjourney invisible watermark.
- **Grok / xAI (Aurora):** PixelPeeper claims custom XMP fields with "encrypted prompt data" — [PixelPeeper](https://pixelpeeper.com/detect-ai-images). A watermark-remover seller claims visible corner logo plus xAI C2PA manifest — [gptcleanup](https://www.gptcleanup.com/grok-image-watermark-remover). Both unverified, no xAI documentation found.
- **Stable Diffusion (open code):** reference/diffusers SDXL pipeline applies an invisible-watermark DwtDct bit message (skips images under 256 px); DwtDctSvd is the Stable Diffusion default per research literature; it is provably removable. — [SDXL watermark.py copy](https://huggingface.co/svjack/diffusers-sdxl-controlnet/blame/3416f50ef53a323a21c9fbb0209d2cf6aa38c1f7/src/diffusers/pipelines/stable_diffusion_xl/watermark.py); [arXiv 2306.01953](https://arxiv.org/pdf/2306.01953). PixelPeeper lists Stable Diffusion as embedding no provenance data.
- **Adobe Firefly:** Content Credentials applied automatically when 100% of pixels are Firefly-generated; manifest names the Firefly model and processing steps. — [Adobe Experience League](https://experienceleague.adobe.com/en/docs/advertising/creative/creative-studio/creative-studio-content-credentials) (via search snippet)
- **Microsoft Designer / Bing Image Creator:** Content Credentials (C2PA) attached; Bing described as adding an invisible watermark with creation time; Designer images carry source metadata. — [MediaPost](https://www.mediapost.com/publications/article/390311/microsoft-bing-integrates-content-credentials-for.html); [Microsoft Designer FAQ](https://support.microsoft.com/en-US/designer/frequently-asked-questions-about-microsoft-Designer)
- **Meta AI:** 2024 policy: visible "Imagined with AI" label, invisible watermark, and metadata (C2PA/IPTC) — [The Decoder](https://the-decoder.com/meta-introduces-imagined-with-ai-labels-for-ai-generated-content-on-its-social-media-platforms/). One undated snippet says newer Meta image model "Muse Image" uses an invisible watermark "Content Seal" not compatible with SynthID/C2PA — unverified, source not opened.
- **Qwen-Image / Wan / Tongyi, Seedream/Doubao/Jimeng:** see section 5.

### Inferences
- The industry pattern: closed API/app = C2PA and/or pixel watermark; open weights = nothing (BFL states this explicitly; SD's DwtDct is the one legacy exception and trivially skipped by editing the pipeline).
- A detector can only claim "AI" confidently on positive signals (C2PA with AI assertion, `trainedAlgorithmicMedia`, SynthID hit); absence proves nothing.

### Gaps
- Ideogram, Leonardo, Recraft: no primary or reliable source found on C2PA, IPTC, XMP or watermarking. Leonardo is owned by Canva (acquired July 2024), which may matter. Needs direct vendor-doc check or testing real outputs.
- Midjourney exact fields/chunks; Grok exact XMP field names; Meta current (2026) watermark; Stability API C2PA; Microsoft IPTC/pixel-watermark detail; whether Gemini consumer app (not Pro) outputs carry C2PA.
- No actual inspection of real output files was done (no exiftool/c2patool runs).

## 2. Survival: re-upload, screenshot, JPEG recompression, resize

### Takeaway
Metadata/C2PA does not survive screenshots and most social uploads; pixel watermarks (SynthID, soft-binding) are designed to survive some edits but no vendor guarantees it, and removal attacks exist.

### Cited Findings
- BFL: manifests can be removed; edits after signing break the hard binding; social platforms/CMS may strip metadata; soft binding "improves resilience but cannot guarantee recovery." — [BFL help](https://help.bfl.ai/articles/3705384267-c2pa-in-flux-generated-images)
- OpenAI: negative result not conclusive; image may be OpenAI's if metadata stripped or watermark degraded. SynthID is there because pixels can survive when metadata does not. — [OpenAI help 8912793](https://help.openai.com/en/articles/8912793-c2pa-in-chatgpt-images)
- Community report: a C2PA PNG uploaded into ChatGPT came back with only IHDR/IDAT/IEND chunks (metadata stripped), yet the original verifies on OpenAI's verifier. No official answer. — [OpenAI community thread](https://community.openai.com/t/are-c2pa-metadata-chunks-intentionally-stripped-from-images-uploaded-to-chatgpt/1385097)
- Metadata lost via re-save, screenshot or social download. — [PixelPeeper](https://pixelpeeper.com/detect-ai-images); IPTC notes stripping can be intentional or accidental — [IPTC](https://iptc.org/news/midjourney-and-shutterstock-ai-sign-up-to-use-of-iptc-digital-source-type-for-generated-ai-content/)
- A test reported that Nano Banana Pro still recognized an image as generated after SynthID cleanup attempts (anecdotal, via Neowin summary). — [Neowin](https://www.neowin.net/news/the-gemini-app-now-tells-you-if-an-image-is-ai-generated-or-not/)
- Research: pixel watermarks are bound to pixels so can survive recompression/format conversion; a file can hold a valid C2PA manifest and a conflicting watermark. — [arXiv 2603.02378](https://arxiv.org/html/2603.02378)
- DwtDct invisible watermarks are removable with generative AI regeneration. — [arXiv 2306.01953](https://arxiv.org/pdf/2306.01953)

### Inferences
- Rough robustness order: C2PA/EXIF/XMP (fragile) < DwtDct (fragile to resize/crop, per general knowledge, not tested here) < SynthID/learned watermarks (more robust, not unbreakable).

### Gaps
- No quantitative robustness numbers per vendor (resize factor, JPEG quality). SynthID image robustness specifics not retrieved.

## 3. Public verification

### Takeaway
OpenAI and Google offer vendor-specific checkers; generic C2PA tools verify any C2PA vendor; invisible watermarks other than OpenAI/Google SynthID have no public decoder found.

### Cited Findings
- OpenAI: openai.com/verify checks C2PA and SynthID; a verification API exists for developers; general C2PA checkers (contentcredentials.org/verify) can also read OpenAI manifests. — [OpenAI help 8912793](https://help.openai.com/en/articles/8912793-c2pa-in-chatgpt-images); [Sightengine tutorial](https://sightengine.com/python-c2pa-tutorial-verifying-images-detecting-tampering)
- Google: upload an image to the Gemini app and ask if it was made with Google AI (SynthID check); Google says C2PA verification will be extended. The checker cannot flag images from other vendors. — [Google blog](https://blog.google/technology/ai/ai-image-verification-gemini-app/); [Neowin](https://www.neowin.net/news/the-gemini-app-now-tells-you-if-an-image-is-ai-generated-or-not/)
- BFL: contentcredentials.org/verify, c2patool, c2pa-js/c2pa-rs; result depends on verifier trust list. — [BFL help](https://help.bfl.ai/articles/3705384267-c2pa-in-flux-generated-images)
- Microsoft: provenance viewable on the Content Credentials site. — [Microsoft Designer FAQ](https://support.microsoft.com/en-US/designer/frequently-asked-questions-about-microsoft-Designer)
- Stable Diffusion DwtDct: decoder is the open `invisible-watermark` library, using the known fixed bit message in the SDXL code (inferred from the code in the watermark.py link above).

### Inferences
- A local detector can cheaply cover: C2PA (c2pa-rs), IPTC/XMP `trainedAlgorithmicMedia`, DwtDct decode. SynthID cannot be checked locally; only via Gemini app / OpenAI verify.

### Gaps
- No public decoder found for Meta's watermark, Grok, Qwen, Seedream. OpenAI verify API details/limits not read.

## 4. API vs web app vs open-weight differences

### Takeaway
Hosted surfaces carry provenance; open-weight local runs generally do not. Flag: rule is explicit only for BFL and Google (Imagen watermark non-optional on Gemini API).

### Cited Findings
- BFL: API, Playground, BFL-controlled integrations get C2PA; self-hosted/third-party inference does not. — [BFL help](https://help.bfl.ai/articles/3705384267-c2pa-in-flux-generated-images)
- OpenAI: ChatGPT, Codex and API all carry both signals. — [OpenAI help](https://help.openai.com/en/articles/8912793-c2pa-in-chatgpt-images)
- Google: Gemini Developer API cannot disable SynthID on Imagen; Nano Banana Pro C2PA in app, Vertex, Ads. — links in section 1.
- SD: the DwtDct watermark lives in the reference/diffusers SDXL pipeline code, so it is an app-code feature a local user can bypass (inference from code location).

### Inferences
- Open-weight Qwen-Image, FLUX.1 [dev], SD checkpoints run locally: expect no watermark (not verified for Qwen-Image; no source retrieved).

### Gaps
- Qwen-Image GitHub/diffusers watermark behavior not checked. FLUX open-weight reference code (flux repo) historically included an optional safety/watermark step; not verified here.

## 5. China labeling regime, Qwen/Wan and Seedream

### Takeaway
Since 1 Sept 2025 Chinese providers must embed implicit labels in file metadata; watermarks encouraged. I found no primary documentation of exactly what Qwen/Wan or Seedream write, only regulatory context and an enforcement action against ByteDance.

### Cited Findings
- Measures issued 14 March 2025 with mandatory national standard GB 45438-2025, effective 1 Sept 2025. Implicit labels must be in metadata (content attributes, provider name/code, content ID); digital watermarks encouraged not required; platforms check metadata and add notices, and warn on suspected AI content where metadata is missing. — [Bird & Bird](https://www.twobirds.com/en/insights/2025/new-ai-content-labelling-rules-in-china-what-are-they-and-how-do-they-compare-to-the-eu-ai-act); [Loeb](https://www.loeb.com/en/insights/publications/2025/03/chinas-ai-labeling-measures-and-mandatory-national-standards-take-effect-september-1)
- Douyin published a spec: metadata label with an "AIGC" marker and timestamp, extensible. (Platform spec, not model-level.) — [ChinaLawTranslate](https://www.chinalawtranslate.com/en/抖音关于人工智能生成内容标识的水印与元数据规/)
- April 2026: CAC said ByteDance's Jianying, Maoxiang and Jimeng AI did not adequately implement identification requirements. — [Invezz](https://invezz.com/news/2026/04/28/china-flags-bytedance-platforms-over-ai-content-labelling-violations/)
- A Ksyun (third-party host) Seedream API example has a `watermark` request field (examples set false); default not stated. — [Ksyun docs](https://docs.ksyun.com/documents/45132)
- Tencent Hunyuan (not asked, for contrast) returns an image with a visible text watermark via a footnote parameter. — [CloudBase docs](https://docs.cloudbase.net/en/ai/image-model/custom-watermark)
- Qwen-Image: 20B open model on Hugging Face, GitHub, ModelScope and Qwen Chat; sources reviewed say nothing about watermarks or labels. — [VentureBeat](https://venturebeat.com/ai/qwen-image-is-a-powerful-open-source-new-ai-image-generator-with-support-for-embedded-text-in-english-chinese)

### Inferences
- Chinese hosted services (Qwen Chat, Tongyi Wanxiang, Jimeng/Doubao) very likely write a metadata implicit label (likely in XMP/EXIF, GB 45438 format) and often a visible corner mark; inferred from the legal duty and the CAC enforcement, NOT confirmed by inspecting outputs. Open-weight Qwen-Image/Wan run locally likely carry none.

### Gaps
- Exact field name/location of the GB 45438 implicit label in JPEG/PNG (read the standard text or run exiftool on Jimeng/Qwen Chat outputs).
- Volcengine Ark and Alibaba Model Studio official watermark parameter docs and defaults.
- Whether any Chinese vendor adds a pixel watermark, and any public decoder.
