"""Build a labelled text eval set: modern generators, human text by subgroup, and character-attacked AI text.

    uv run python scripts/fetch_text_eval.py [--out data/eval_text] [--seed 0]

Writes OUT/{real,ai}/<group>/*.txt, the layout scripts/eval_images.py reads:
  real/pan-{essays,news,fiction}           PAN'25 Voight-Kampff val split, human texts
  ai/pan-<generator>                       PAN'25 val split: GPT-4o, o3-mini, GPT-4.5, Gemini 2.0, Llama-3.x, ...
  real/ellipse                             ELLIPSE: English-learner essays, grades 8-12 (CC BY 4.0)
  real/liang-{toefl,college,cs224n}        Liang et al. 2023 human texts; TOEFL is their non-native set (MIT)
  ai/attack-{homoglyph,zerowidth,whitespace}  PAN AI texts with RAID-style character attacks

PAN'25 is "Copyrighted Material, No Redistribution", research use only: OUT is git-ignored, never publish it.
Texts the router rejects (under 150 tokens, not English) are skipped, so every row gets scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from ai_detector.detectors.image import HF_CACHE
from ai_detector.router import route

PAN_URL = "https://zenodo.org/records/14962653/files/pan25-generative-ai-detection-task1-train.zip?download=1"
ELLIPSE_URL = "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/ELLIPSE_Final_github_train.csv"
LIANG_GROUPS = {"Real TOEFL": "liang-toefl", "Real College Essays": "liang-college", "Real CS224N": "liang-cs224n"}

# Attack tables are independent of router._LOOKALIKES on purpose: the eval should catch what the defence misses.
HOMOGLYPHS = dict(zip("aeopcyxiAEOPCHKMTXB", "аеорсухіАЕОРСНКМТХВ"))


def homoglyph(text: str, rng: random.Random, rate: float = 0.5) -> str:
    return "".join(HOMOGLYPHS[c] if c in HOMOGLYPHS and rng.random() < rate else c for c in text)


def zero_width(text: str, rng: random.Random, rate: float = 0.2) -> str:
    return "".join(c + "​" if rng.random() < rate else c for c in text)


def whitespace(text: str, rng: random.Random, rate: float = 0.2) -> str:
    return " ".join(w + " " * rng.randint(1, 3) if rng.random() < rate else w for w in text.split(" "))


ATTACKS = {"homoglyph": homoglyph, "zerowidth": zero_width, "whitespace": whitespace}


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def in_domain(text: str) -> bool:
    return not route(text=text).out_of_domain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/eval_text"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pan-human-per-genre", type=int, default=300)
    parser.add_argument("--pan-per-generator", type=int, default=30)
    parser.add_argument("--ellipse", type=int, default=400)
    parser.add_argument("--per-attack", type=int, default=60)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cache = args.out / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    picked: dict[tuple[str, str], list[tuple[str, str]]] = {}  # (label, group) -> [(id, text)]

    # val split only: less likely than train to sit in the classifiers' training data
    with zipfile.ZipFile(fetch(PAN_URL, cache / "pan25.zip")) as zf:
        pan = [json.loads(line) for line in zf.read("val.jsonl").decode().splitlines()]
    by_group = defaultdict(list)
    for r in pan:
        group = f"pan-{r['genre']}" if r["label"] == 0 else f"pan-{r['model']}"
        by_group[("real" if r["label"] == 0 else "ai", group)].append((r["id"], r["text"]))
    for key, items in by_group.items():
        n = args.pan_human_per_genre if key[0] == "real" else args.pan_per_generator
        picked[key] = rng.sample(items, min(n, len(items)))

    with fetch(ELLIPSE_URL, cache / "ellipse_train.csv").open(newline="", encoding="utf-8") as fh:
        ellipse = [(r["text_id_kaggle"], r["full_text"]) for r in csv.DictReader(fh)]
    picked[("real", "ellipse")] = rng.sample(ellipse, args.ellipse)

    liang = json.loads(Path(hf_hub_download("WxWx/ChatGPT-Detector-Bias", "all_data.json", repo_type="dataset",
                                            cache_dir=HF_CACHE)).read_text())
    for i, r in enumerate(liang):
        if r["name"] in LIANG_GROUPS:
            picked.setdefault(("real", LIANG_GROUPS[r["name"]]), []).append((f"{i:04d}", r["text"]))

    # Attacked texts come from AI texts not picked above: a cleaned attack is a near-copy of its source, which
    # would leak across the fit/test split.
    used = {i for (label, _), items in picked.items() if label == "ai" for i, _ in items}
    spare = [(r["id"], r["text"]) for r in pan if r["label"] == 1 and r["id"] not in used]
    sources = rng.sample(spare, args.per_attack * len(ATTACKS))
    for k, (name, attack) in enumerate(ATTACKS.items()):
        chunk = sources[k * args.per_attack : (k + 1) * args.per_attack]
        picked[("ai", f"attack-{name}")] = [(f"{i}-{name}", attack(t, rng)) for i, t in chunk]

    for (label, group), items in sorted(picked.items()):
        dest = args.out / label / group
        dest.mkdir(parents=True, exist_ok=True)
        kept = [(i, t) for i, t in items if in_domain(t)]
        for i, t in kept:
            (dest / f"{i}.txt").write_text(t, encoding="utf-8")
        print(f"{label}/{group}: {len(kept)} kept, {len(items) - len(kept)} out of domain")


if __name__ == "__main__":
    main()
