"""Train the text head (ai_detector/detectors/text.py HEAD_PATH): a logistic layer over Fakespot + Desklib pooled
features, both mean window logits and log word count.

    uv run python scripts/train_text_head.py features --muse muse.jsonl --pan pan25.zip --mage train.csv --out f.npz
    uv run python scripts/train_text_head.py fit --feats f.npz --l2 0.01 --save ai_detector/detectors/text_head.npz

Why: the two classifiers miss modern casual chat-model text (Meta Muse: 58% called human on 150-250 words) and
degrade under ~150 tokens. Each training text is scored as one random 40-450 word crop, so the head sees short text.
Data: Muse generations (scripts outside the repo; prompt types disjoint from the benchmark), PAN'25 train split
(22 generators + human essays/news/fiction; research use only, never publish) and MAGE train (27 generators + human
Reddit, Yelp, stories, news, science...). Validation holds out whole Muse prompt types and whole PAN generators.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import defaultdict

import numpy as np

MIN_WORDS, MAX_WORDS = 40, 450


def crop(text: str, rng: random.Random) -> str:
    words = text.split()
    n = rng.randint(MIN_WORDS, min(MAX_WORDS, len(words)))
    start = 0 if rng.random() < 0.5 else rng.randint(0, len(words) - n)
    return " ".join(words[start : start + n])


def load_rows(args, rng: random.Random) -> list[dict]:
    rows = []

    def add(text, label, source, group):
        if len(text.split()) >= MIN_WORDS:
            rows.append({"text": text, "y": label, "source": source, "group": group})

    for line in open(args.muse):
        r = json.loads(line)
        add(r["text"], 1, "muse", f"{r['kind']}:{r['task']}")
    with zipfile.ZipFile(args.pan) as zf:
        pan = [json.loads(line) for line in zf.read("train.jsonl").decode().split("\n") if line.strip()]
    by = defaultdict(list)
    for r in pan:
        by[(r["label"], r["genre"] if r["label"] == 0 else r["model"])].append(r["text"])
    for (label, group), texts in sorted(by.items()):
        for t in rng.sample(texts, min(len(texts), 400 if label == 0 else 70)):
            add(t, label, "pan", group)
    mage = defaultdict(list)
    for r in csv.DictReader(open(args.mage, encoding="utf-8-sig")):
        mage[(int(r["label"] == "0"), r["src"])].append(r["text"])  # MAGE: label "1" = human
    human = [k for k in mage if k[0] == 0]
    machine = [k for k in mage if k[0] == 1]
    for keys, total in ((human, 2000), (machine, 1000)):
        per = max(1, total // len(keys))
        for k in keys:
            for t in rng.sample(mage[k], min(len(mage[k]), per)):
                add(t, k[0], "mage", k[1])
    return rows


def features(args) -> None:
    from ai_detector.detectors import text as T

    rng = random.Random(0)
    rows = load_rows(args, rng)
    rng.shuffle(rows)
    X, meta = [], []
    for i, r in enumerate(rows):
        t = crop(r["text"], rng)
        X.append(T.head_inputs(T.score_classifiers(t), t))
        meta.append({k: r[k] for k in ("y", "source", "group")} | {"words": len(t.split())})
        if i % 250 == 0:
            print(i, "/", len(rows), flush=True)
    np.savez(args.out, X=np.stack(X), meta=np.array([json.dumps(m) for m in meta]))


def fit_logistic(X, y, l2, w_pos=1.0):
    """Full-batch L-BFGS logistic regression with L2 on standardised inputs. Returns mean, std, w, b."""
    import torch

    mean, std = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mean) / std, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    sw = torch.where(yt == 1, w_pos, 1.0)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")).mean()
        loss = loss + l2 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return mean, std, w.detach().numpy(), float(b.item())


def split(meta, rng: random.Random):
    """Validation = whole held-out Muse prompt types and PAN generators, plus 20% of everything else."""
    muse_groups = sorted({m["group"] for m in meta if m["source"] == "muse"})
    pan_ai = sorted({m["group"] for m in meta if m["source"] == "pan" and m["y"] == 1})
    held = set(rng.sample(muse_groups, len(muse_groups) // 5)) | set(rng.sample(pan_ai, 4))
    return np.array([m["group"] in held or rng.random() < 0.2 for m in meta])


def fit(args) -> None:
    d = np.load(args.feats)
    X, meta = d["X"], [json.loads(m) for m in d["meta"]]
    y = np.array([m["y"] for m in meta], dtype=np.float64)
    val = split(meta, random.Random(1))
    mean, std, w, b = fit_logistic(X[~val], y[~val], args.l2)
    z = ((X[val] - mean) / std) @ w + b
    yv, mv = y[val], [m for m, v in zip(meta, val, strict=True) if v]
    print(f"validation: {int(yv.sum())} AI / {int((1 - yv).sum())} human")
    words = np.array([m["words"] for m in mv])
    for lo, hi in ((40, 100), (100, 150), (150, 250), (250, 451)):
        sel = (words >= lo) & (words < hi)
        hz, az = z[sel & (yv == 0)], z[sel & (yv == 1)]
        thr = np.quantile(hz, 0.98)
        print(f"  {lo}-{hi} words: AI caught at 2% human FPR {np.mean(az > thr):.3f} (n AI {len(az)}, human {len(hz)})")
    # fusion.yaml stackers.text bands, as p = sigmoid(z): likely_ai = 2% of validation humans above (1% for short),
    # no_ai_signal = 3% of validation AI below. Short = the 80-135 word texts that fall under 150 tokens.
    p = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    full, short = words >= 85, (words >= 80) & (words < 135)
    print(
        "bands:",
        {
            "no_ai_signal_below": p(np.quantile(z[full & (yv == 1)], 0.03)),
            "likely_ai_above": p(np.quantile(z[full & (yv == 0)], 0.98)),
        },
    )
    print(
        "short_bands:",
        {
            "no_ai_signal_below": p(np.quantile(z[short & (yv == 1)], 0.03)),
            "likely_ai_above": p(np.quantile(z[short & (yv == 0)], 0.99)),
        },
    )
    if args.save:
        # Ship this train-split fit, not a refit on everything: the bands above are only valid for this model.
        arrays = {"mean": mean, "std": std, "w": w}
        np.savez(args.save, **{k: v.astype(np.float32) for k, v in arrays.items()}, b=np.float32(b))
        print("saved", args.save)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--muse", required=True)
    f.add_argument("--pan", required=True)
    f.add_argument("--mage", required=True)
    f.add_argument("--out", required=True)
    g = sub.add_parser("fit")
    g.add_argument("--feats", required=True)
    g.add_argument("--l2", type=float, default=0.01)
    g.add_argument("--save")
    args = p.parse_args()
    features(args) if args.cmd == "features" else fit(args)


if __name__ == "__main__":
    main()
