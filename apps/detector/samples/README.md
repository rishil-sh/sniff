# Samples

One file for each kind of result the detector can give. Drop any of them into the web app's **Check** page (http://localhost:3001), or POST them to `http://127.0.0.1:7860/v1/analyze`.

The table shows what the app returned for each file on 2026-11-04, with Muse off.

| File | Verdict | p(AI) | Message the user sees |
|---|---|---|---|
| `audio/01_ai_elevenlabs_v3.mp3` | likely_ai | 0.94 | 🤖 AI-generated voice |
| `audio/02_ai_elevenlabs_v4_acting.mp3` | likely_ai | 1.00 | 🤖 AI-generated voice |
| `audio/03_ai_openai_borderline.wav` | inconclusive | 0.65 | 🤔 Can't say for sure. Best guess: AI |
| `audio/04_human_audiobook.wav` | no_ai_signal | 0.00 | ✅ No signs of AI |
| `audio/05_human_phone_mic.wav` | no_ai_signal | 0.00 | ✅ No signs of AI |
| `audio/06_ai_audioseal_watermarked.wav` | declared_ai | 1.00 | 🤖 AI-generated |
| `image/01_real_camera_photo.jpg` | no_ai_signal | 0.14 | ✅ No signs of AI |
| `image/02_real_photo_dataset.jpg` | no_ai_signal | 0.09 | ✅ No signs of AI |
| `image/03_ai_gpt_image_with_c2pa.png` | declared_ai | 1.00 | 🤖 AI-generated |
| `image/04_ai_nano_banana_with_c2pa.png` | declared_ai | 1.00 | 🤖 AI-generated |
| `image/05_ai_nano_banana_c2pa_stripped.png` | likely_ai | 0.88 | 🤖 AI-generated |
| `image/06_ai_flux2_no_label.png` | likely_ai | 0.95 | 🤖 AI-generated |
| `image/07_ai_qwen_borderline.png` | inconclusive | 0.75 | 🤔 Can't say for sure. Best guess: AI |
| `image/08_palette_png_cannot_check.png` | inconclusive | n/a | ⚠️ Couldn't check this |
| `text/01_human_dickens.txt` | no_ai_signal | 0.00 | ✍️ Written by a human |
| `text/02_ai_written_by_claude.txt` | likely_ai | 1.00 | 🤖 Written by AI |
| `text/03_ai_with_hidden_characters.txt` | likely_ai | 1.00 | 🤖 Written by AI |
| `text/04_too_short.txt` | inconclusive | n/a | ⚠️ Couldn't check this |
| `text/05_not_english.txt` | inconclusive | n/a | ⚠️ Couldn't check this |
| `video/01_test_pattern.mp4` | inconclusive | n/a | ⚠️ Couldn't check this |

## What each one shows
- **image 01–02:** real photos. Expect no AI signal.
- **image 03–04:** AI images that carry signed Content Credentials (OpenAI, Google), so they are declared AI.
- **image 05:** image 04 re-saved, which strips the label. The pixel detector still catches it.
- **image 06–07:** AI images with no label. One is clear, the other is borderline: it's under the 0.85 cut-off, set high so real photos aren't flagged.
- **image 08:** a palette PNG. Its colours were reduced, which wipes the traces the detectors read, so the app asks for the original.
- **audio 01–03:** AI voices: two clear ElevenLabs clips and one borderline OpenAI clip.
- **audio 04–05:** real people. Expect no AI signal.
- **audio 06:** audio 04 with a Meta AudioSeal watermark embedded, which reads as declared AI.
- **text 03:** AI text with invisible and look-alike characters hidden in it. They are removed and reported.
- **text 04–05 and video 01:** out of scope (too short, not English, video not supported yet).

## Sources
- image 01: "Camponotus flavomarginatus ant" (Wikimedia Commons, CC BY-SA)
- image 02–07: `zr-zhang/MLLM-Generated-Image-Detection-Dataset` on Hugging Face
- audio: `data/eval_voices` (LibriSpeech, Common Voice, ElevenLabs and OpenAI clips)
- text 01: Dickens; text 05: Cervantes (both public domain); text 02: written by Claude
