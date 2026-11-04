"""Measure every image and text detector on a labelled folder; fit Platt, a logistic stacker and fixed-FPR bands.

    uv run python scripts/eval_images.py DATASET_DIR [--csv out.csv] [--no-domain-gate]
    uv run python scripts/eval_images.py --from-csv out.csv [--stack txt.fakespot,txt.desklib,txt.binoculars]

DATASET_DIR holds ai/ and real/ subfolders (searched recursively) of images and/or .txt files. The first folder
under ai/ or real/ is the row's group (real/ellipse/…), reported separately so per-group FPR is visible.
Rows are split by a hash of their path: Platt, the stacker and its bands are fit on the fit split, and every
reported number comes from the held-out test split. --from-csv refits from a saved run without rescoring.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from ai_detector.analyze import analyze
from ai_detector.detectors import image as image_mod
from ai_detector.fusion import fuse, load_policy
from ai_detector.router import IMAGE_EXTS, RoutedInput
from ai_detector.schemas import Evidence

LEARNED = ("img.commfor", "img.clip", "img.spai", "txt.binoculars", "txt.fastdetect", "txt.fakespot", "txt.desklib")
LOWER_IS_AI = {"txt.binoculars"}  # metrics negate these; Platt fits the raw score (negative slope)
MODALITY = {"img": "image", "txt": "text"}
VERDICTS = ("likely_ai", "inconclusive", "no_ai_signal")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random AI item outscores a random real one (ties count half)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> float:
    threshold = np.quantile(scores[labels == 0], 1 - fpr)
    return float((scores[labels == 1] > threshold).mean())


def tpr_ci(scores: np.ndarray, labels: np.ndarray, fpr: float, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for TPR at a fixed FPR. Resamples each class and re-derives the threshold every time,
    because threshold noise dominates when there are few human texts."""
    rng = np.random.default_rng(seed)
    pos, neg = scores[labels == 1], scores[labels == 0]
    stats = [(rng.choice(pos, len(pos)) > np.quantile(rng.choice(neg, len(neg)), 1 - fpr)).mean() for _ in range(n_boot)]
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion k/n; sane at k = 0, unlike the normal approximation."""
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def fit_logistic(x: np.ndarray, labels: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """Logistic regression by Newton's method with Platt's smoothed targets, so separable data stays finite.

    x: [n, k] features. Returns k coefficients followed by the intercept; l2 shrinks the coefficients only.
    """
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    target = np.where(labels == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    x = np.column_stack([x, np.ones(len(x))])
    reg = np.diag([l2] * (x.shape[1] - 1) + [0.0]) + 1e-9 * np.eye(x.shape[1])
    w = np.zeros(x.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -500, 500)))
        step = np.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]) + reg, x.T @ (p - target) + reg @ w)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Platt (1999): logistic fit of label on raw score, with his smoothed targets."""
    a, b = fit_logistic(scores[:, None], labels)
    return float(a), float(b)


def in_test(path: str, share: float) -> bool:
    """Stable split by path, so a rerun or --from-csv puts every row on the same side."""
    return int(hashlib.sha1(path.encode()).hexdigest()[:8], 16) / 2**32 < share


def score_folder(dataset: Path) -> list[dict]:
    files = [
        (path, label)
        for label, sub in ((1, "ai"), (0, "real"))
        for path in sorted((dataset / sub).rglob("*"))
        if path.suffix.lower() in IMAGE_EXTS | {".txt"}
    ]
    if not {label for _, label in files} == {0, 1}:
        raise SystemExit(f"need files under both {dataset}/ai and {dataset}/real")
    rows = []
    for i, (path, label) in enumerate(files, 1):
        # shortcut: one analyze() per image, so SPAI reloads its model each time; batch it if this gets slow
        if path.suffix.lower() == ".txt":
            verdict = asyncio.run(analyze(text=path.read_text(encoding="utf-8")))
        else:
            verdict = asyncio.run(analyze(data=path.read_bytes(), filename=path.name))
        rel = path.relative_to(dataset)
        row = {"path": rel.as_posix(), "group": "/".join(rel.parts[:2]) if len(rel.parts) > 2 else rel.parts[0],
               "label": label, "modality": verdict.modality, "verdict": verdict.verdict,
               "ood": any(e.status == "out_of_domain" for e in verdict.evidence)}
        for e in verdict.evidence:
            row |= {f"{e.id}.status": e.status, f"{e.id}.raw": e.raw_score, f"{e.id}.p": e.p_ai, f"{e.id}.version": e.version}
        rows.append(row)
        print(f"[{i}/{len(files)}] {'ai  ' if label else 'real'} {verdict.verdict:16} {rel}", flush=True)
    return rows


def read_csv(path: Path) -> list[dict]:
    def parse(key: str, value: str):
        if key == "label":
            return int(value)
        if key == "ood":
            return value == "True"
        if key.endswith((".raw", ".p")):
            return float(value) if value else None
        return value

    with path.open(newline="") as fh:
        return [{k: parse(k, v) for k, v in row.items()} for row in csv.DictReader(fh)]


def refuse(row: dict, policy: dict) -> str:
    """The verdict this row gets under `policy`, rebuilt from its learned evidence only.

    Provenance and watermark evidence isn't stored, so declared verdicts are kept as scored.
    """
    if row["ood"] or str(row["verdict"]).startswith("declared"):
        return "inconclusive" if row["ood"] else row["verdict"]
    evidence = [
        Evidence(id=d, detector=d, version=row.get(f"{d}.version") or "", modality=row["modality"], signal="learned",
                 status=row[f"{d}.status"], raw_score=row.get(f"{d}.raw"), p_ai=row.get(f"{d}.p"))
        for d in LEARNED if row.get(f"{d}.status")
    ]
    routed = RoutedInput(modality=row["modality"], out_of_domain=False, reason=None, filename=None, mime=None,
                         size_bytes=0, duration_s=None, token_count=None, sha256="")
    return fuse(routed, evidence, request_id="eval", policy=policy).verdict


def detector_table(test: list[dict]) -> dict[str, dict]:
    """Prints held-out metrics per detector; returns {det: {auroc, tpr_at_1pct_fpr}} for the platt: entries."""
    measured = {}
    print("\nheld-out test split, per detector:")
    print("  detector        n_ok   AUROC   TPR@1%FPR [95% CI]    TPR@5%FPR [95% CI]   other statuses")
    for det in LEARNED:
        ok = [r for r in test if r.get(f"{det}.status") == "ok" and r.get(f"{det}.raw") is not None]
        other = Counter(r.get(f"{det}.status") or "missing" for r in test if r not in ok)
        labels = np.array([r["label"] for r in ok])
        if len(set(labels.tolist())) < 2:
            if ok or other.get("missing", 0) < len(test):
                print(f"  {det:14} {len(ok):4}   (needs ok scores on both classes)  {dict(other)}")
            continue
        scores = np.array([r[f"{det}.raw"] for r in ok], dtype=float)
        s = -scores if det in LOWER_IS_AI else scores
        cells = []
        for fpr in (0.01, 0.05):
            lo, hi = tpr_ci(s, labels, fpr)
            cells.append(f"{tpr_at_fpr(s, labels, fpr):.3f} [{lo:.2f}-{hi:.2f}]")
        auc = auroc(s, labels)
        measured[det] = {"auroc": round(auc, 4), "tpr_at_1pct_fpr": round(tpr_at_fpr(s, labels, 0.01), 4)}
        print(f"  {det:14} {len(ok):4}   {auc:.3f}   {cells[0]:20} {cells[1]:20} {dict(other)}")
    return measured


def fit_stacker(fit: list[dict], feats: list[str], target_fpr: float, miss: float, set_name: str) -> dict | None:
    rows = [r for r in fit if all(r.get(f"{f}.status") == "ok" and r.get(f"{f}.raw") is not None for f in feats)]
    labels = np.array([r["label"] for r in rows])
    if len(set(labels.tolist())) < 2:
        print(f"\nstacker: no fit rows with ok scores for all of {feats} on both classes")
        return None
    versions = {f: Counter(r[f"{f}.version"] for r in rows) for f in feats}
    if any(len(v) > 1 for v in versions.values()):
        raise SystemExit(f"rows mix model versions, refit per version: {versions}")
    x = np.array([[r[f"{f}.raw"] for f in feats] for r in rows], dtype=float)
    w = fit_logistic(x, labels, l2=1e-3)
    p = 1 / (1 + np.exp(-np.clip(x @ w[:-1] + w[-1], -500, 500)))
    # Fixed-FPR band: "likely AI" above the (1 - target_fpr) quantile of human scores; "no AI signal" below the
    # score that `miss` of AI texts fall under. Clamped at 0.5 so a verdict never contradicts its own p_ai.
    hi = max(float(np.quantile(p[labels == 0], 1 - target_fpr, method="higher")), 0.5)
    lo = min(float(np.quantile(p[labels == 1], miss, method="lower")), 0.5)
    n_human = int((labels == 0).sum())
    if n_human < 3 / target_fpr:
        print(f"\nstacker: only {n_human} human fit rows; the {target_fpr:.0%} FPR band is rough below {3 / target_fpr:.0f}")
    return {
        "intercept": round(float(w[-1]), 6),
        "coef": {f: round(float(c), 6) for f, c in zip(feats, w[:-1])},
        "versions": {f: next(iter(v)) for f, v in versions.items()},
        "bands": {"no_ai_signal_below": round(lo, 4), "likely_ai_above": round(hi, 4)},
        "target_fpr": target_fpr,
        "set": f"{set_name}-fit{len(rows)}",
    }


def verdict_table(test: list[dict], modality: str, policies: dict[str, dict]) -> None:
    rows = [r for r in test if r["modality"] == modality]
    print(f"\nheld-out test split, {modality} verdicts by group (likely_ai share = FPR on real/, TPR on ai/):")
    for name, policy in policies.items():
        got = [(r, refuse(r, policy)) for r in rows]
        print(f"  policy: {name}")
        print(f"    {'group':28} {'n':>5} {'likely_ai [95% CI]':>22} {'inconclusive':>13} {'no_ai_signal':>13}")
        groups = sorted({r["group"] for r in rows}, key=lambda g: (not g.startswith("real"), g))
        for group in groups + ["ALL real", "ALL ai"]:
            label = {"ALL real": 0, "ALL ai": 1}.get(group)
            v = Counter(verdict for r, verdict in got if (r["group"] == group if label is None else r["label"] == label))
            n = sum(v.values())
            if not n:
                continue
            lo, hi = wilson(v["likely_ai"], n)
            print(f"    {group:28} {n:5} {v['likely_ai'] / n:8.3f} [{lo:.3f}-{hi:.3f}] "
                  f"{v['inconclusive'] / n:13.3f} {v['no_ai_signal'] / n:13.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?")
    parser.add_argument("--csv", type=Path, help="write per-item rows here")
    parser.add_argument("--from-csv", type=Path, help="refit from a saved --csv run instead of scoring a folder")
    parser.add_argument("--no-domain-gate", action="store_true",
                        help="score palette/transparent/small images anyway, to test whether the gate is needed")
    parser.add_argument("--skip-spai", action="store_true",
                        help="skip SPAI (~8 s/image, weight 0 in fusion.yaml) for large sets")
    parser.add_argument("--test-share", type=float, default=0.5, help="share of rows held out for measurement")
    parser.add_argument("--stack", default="txt.fakespot,txt.desklib", help="detector ids the stacker combines")
    parser.add_argument("--target-fpr", type=float, default=0.01, help="FPR the likely_ai band is set for")
    # 0.01 keeps an inconclusive band; at 0.05 the band all but vanished on eval_text (0.8% of humans inconclusive)
    parser.add_argument("--miss", type=float, default=0.01, help="share of AI texts allowed in the no_ai_signal band")
    args = parser.parse_args()
    if args.no_domain_gate:
        image_mod.domain_issue = lambda _image: None
    if args.skip_spai:
        image_mod.spai_ready = lambda: False

    if args.from_csv:
        rows, set_name = read_csv(args.from_csv), args.from_csv.stem
    elif args.dataset:
        rows, set_name = score_folder(args.dataset), args.dataset.name
    else:
        parser.error("give DATASET_DIR or --from-csv")
    if args.csv and not args.from_csv:
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)

    fit = [r for r in rows if not in_test(r["path"], args.test_share)]
    test = [r for r in rows if in_test(r["path"], args.test_share)]
    print(f"\nsplit: fit {len(fit)} ({sum(r['label'] for r in fit)} ai) | "
          f"test {len(test)} ({sum(r['label'] for r in test)} ai) | out of domain: {sum(r['ood'] for r in rows)}")
    measured = detector_table(test)

    platt = {}
    for det in LEARNED:
        ok = [r for r in fit if r.get(f"{det}.status") == "ok" and r.get(f"{det}.raw") is not None]
        labels = np.array([r["label"] for r in ok])
        if len(set(labels.tolist())) == 2:
            a, b = fit_platt(np.array([r[f"{det}.raw"] for r in ok], dtype=float), labels)
            platt[det] = {"a": round(a, 6), "b": round(b, 6), "set": f"{set_name}-fit{len(ok)}", **measured.get(det, {})}

    feats = [f for f in args.stack.split(",") if f]
    modality = MODALITY[feats[0].split(".")[0]]
    stacker = fit_stacker(fit, feats, args.target_fpr, args.miss, set_name)
    current = load_policy()
    proposed = {**current, "platt": {**(current.get("platt") or {}), **platt}}
    if stacker:
        proposed["stackers"] = {**(current.get("stackers") or {}), modality: stacker}
        ok = [r for r in test if all(r.get(f"{f}.status") == "ok" and r.get(f"{f}.raw") is not None for f in feats)]
        s = np.array([stacker["intercept"] + sum(c * r[f"{f}.raw"] for f, c in stacker["coef"].items()) for r in ok])
        y = np.array([r["label"] for r in ok])
        if len(set(y.tolist())) == 2:
            lo, hi = tpr_ci(s, y, 0.01)
            print(f"\nstacker on the test split: n={len(ok)} AUROC {auroc(s, y):.3f} "
                  f"TPR@1%FPR {tpr_at_fpr(s, y, 0.01):.3f} [{lo:.2f}-{hi:.2f}]")
    for m in sorted({r["modality"] for r in test}):
        verdict_table(test, m, {"current fusion.yaml": current, "proposed (fit split)": proposed})

    print("\nproposed blocks for ai_detector/fusion.yaml (fit on the fit split only):")
    print(yaml.safe_dump({"platt": platt} | ({"stackers": {modality: stacker}} if stacker else {}),
                         sort_keys=False, default_flow_style=None, width=120))


if __name__ == "__main__":
    main()
