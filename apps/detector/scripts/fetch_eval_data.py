"""Sample a labelled eval set from zr-zhang/MLLM-Generated-Image-Detection-Dataset (Apache-2.0).

    uv run python scripts/fetch_eval_data.py [--per-class 40] [--out data/eval] [--faces 0] [--screenshots]

Writes OUT/ai/<generator>/*.png and OUT/real/*.png, the layout scripts/eval_images.py reads.
--faces N adds N real FFHQ portraits (Flickr, collected 2018, before diffusion models;
CC BY-NC-SA 4.0, research use) under OUT/real/ffhq/.
--ai-faces N adds N FLUX.1-dev photoreal portraits (gaunernst/flux-dev-portrait) under
OUT/ai/flux_portrait/: same 1024px lossless webp as the FFHQ copy, so format can't give it away.
--screenshots adds a phone-screenshot copy (portrait crop, rescale, recompress, no metadata)
of every image, real and AI alike, under OUT/<label>/screenshot/.
The dataset licence limits use to detection research; do not use it to evade detectors.
"""

from __future__ import annotations

import argparse
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, hf_hub_url, list_repo_files
from PIL import Image

REPO = "zr-zhang/MLLM-Generated-Image-Detection-Dataset"
GENERATORS = ["GPT Image 2", "Nano Banana 2", "Qwen Image 2.1", "FLUX.2"]
# (repo, tar shards): lossless 1024px webp faces, already shuffled
FFHQ = ("gaunernst/ffhq-1024-wds", ["00000.tar"])
FLUX_PORTRAIT = ("gaunernst/flux-dev-portrait", [f"wds/part{i}.tar" for i in range(10)])


def fetch_faces(source: tuple[str, list[str]], dest: Path, prefix: str, n: int) -> int:
    repo, shards = source
    dest.mkdir(parents=True, exist_ok=True)
    got = 0
    for shard in shards:
        url = hf_hub_url(repo, shard, repo_type="dataset")
        # stream the tar and stop after n faces, instead of downloading whole shards
        with urllib.request.urlopen(url) as resp, tarfile.open(fileobj=resp, mode="r|") as tar:
            for member in tar:
                if got >= n:
                    return got
                if member.isfile() and member.name.endswith(".webp"):
                    # keep the shard folder in the name: each shard restarts at 000000
                    name = "_".join(Path(member.name).parts)
                    (dest / f"{prefix}_{name}").write_bytes(tar.extractfile(member).read())
                    got += 1
    return got


def screenshot(src: Path, dest: Path, rng: random.Random) -> None:
    """Mimic an image shown full-width on a phone, screenshotted, and shared."""
    im = Image.open(src).convert("RGB")
    w, h = im.size
    ratio = rng.uniform(0.46, 0.75)  # portrait crops from 9:19.5 to 3:4
    cw = min(w, round(h * ratio))
    ch = min(h, round(cw / ratio))
    im = im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    tw = rng.randint(600, 1170)  # common phone screenshot widths
    im = im.resize((tw, round(ch * tw / cw)), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if rng.random() < 0.5:
        # not with_suffix: "ss__FLUX.2__x" would lose ".2__x" as if it were an extension
        im.save(dest.parent / f"{dest.name}.png")  # screenshot sent as a file
    else:
        im.save(dest.parent / f"{dest.name}.jpg", quality=rng.randint(70, 92))  # sent through a chat app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=40, help="images per generator (real gets 4x)")
    parser.add_argument("--out", type=Path, default=Path("data/eval"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faces", type=int, default=0, help="real FFHQ portraits to add")
    parser.add_argument("--ai-faces", type=int, default=0, help="FLUX photoreal portraits to add")
    parser.add_argument("--screenshots", action="store_true", help="add a screenshot copy of every image")
    args = parser.parse_args()

    files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("images/")]
    rng = random.Random(args.seed)

    def pick(prefix: str, n: int) -> list[str]:
        pool = sorted(f for f in files if f.startswith(f"images/{prefix}/"))
        return rng.sample(pool, min(n, len(pool)))

    plan = {("real", None): pick("Real", args.per_class * len(GENERATORS))}
    plan |= {("ai", g.replace(" ", "_")): pick(g, args.per_class) for g in GENERATORS}

    for (label, gen), picked in plan.items():
        dest = args.out / label / (gen or "")
        dest.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(picked):
            cached = hf_hub_download(REPO, f, repo_type="dataset")
            # subset folders reuse names (0001.png), so prefix with the subset to stay unique
            subset = Path(f).parent.name.split()[0].lower()
            shutil.copyfile(cached, dest / f"{subset}_{Path(f).name}")
        print(f"{label}/{gen or ''}: {len(picked)}")

    if args.faces:
        print(f"real/ffhq: {fetch_faces(FFHQ, args.out / 'real' / 'ffhq', 'ffhq', args.faces)}")
    if args.ai_faces:
        n = fetch_faces(FLUX_PORTRAIT, args.out / "ai" / "flux_portrait", "fluxp", args.ai_faces)
        print(f"ai/flux_portrait: {n}")

    if args.screenshots:
        for label in ("ai", "real"):
            originals = [
                p for p in sorted((args.out / label).rglob("*"))
                if p.is_file() and "screenshot" not in p.parts and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            for p in originals:
                # name keeps the original's folder + stem so splits can keep a pair together
                rel = "__".join(p.relative_to(args.out / label).with_suffix("").parts)
                screenshot(p, args.out / label / "screenshot" / f"ss__{rel}", rng)
            print(f"{label}/screenshot: {len(originals)}")


if __name__ == "__main__":
    main()
