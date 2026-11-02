#!/usr/bin/env python3
"""Score a detector on the public split of the October 2026 image & video benchmark.

    pip install requests numpy

    # 1. Your own detector: a Python function file_path -> AI probability (0..1)
    python eval.py --adapter my_detector.py:predict

    # 2. The WasItAIGenerated API (uses your API credits)
    WAI_API_KEY=... python eval.py --adapter wai

    # 3. Scores you already computed elsewhere: CSV with columns id,score
    python eval.py --scores my_scores.csv

Files are downloaded from Hugging Face on demand into --data-dir (only the ones being scored).
Scores are appended to --out one row per file, so an interrupted run picks up where it stopped.
The same file can be passed back with --scores to print the report again.

Metrics, as in the write-up:
  AUROC                how well the scores separate AI from real files across all thresholds.
  Detected @ 1% FP     share of AI files above the threshold at which 1% of the real files of the
                       same condition are flagged (at most floor(1% of n_real) real files above it).
  Social copies        the WhatsApp, Instagram and re-shared copies of every image, pooled.
  Messenger copy       the messenger copy of every clip.

Writing an adapter for another detector
---------------------------------------
Create a .py file with one function that takes a local file path and returns the probability that the
file is AI-generated, as a float from 0 to 1 (any monotonic score works for AUROC and the 1% FP
threshold). Raise an exception to skip a file; it is reported as unscored and retried on the next run.

    # my_detector.py
    from PIL import Image

    model = load_my_model()               # runs once, when eval.py loads the file

    def predict(path: str) -> float:
        if path.endswith(".mp4"):
            # Image-only detector? The write-up scores the same frames of each clip and averages
            # them. Sample frames here (e.g. with OpenCV or ffmpeg) and return the mean, or run
            # this script with --media image.
            raise NotImplementedError("no video support")
        return float(model(Image.open(path).convert("RGB")))

    python eval.py --adapter my_detector.py:predict --out scores_my_detector.csv

An installed module works too: --adapter my_package.detector:predict
"""
import argparse
import csv
import importlib
import importlib.util
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

REPO = "wasitaigeneratedcom/frontier-image-video-bench-2026-10"
HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
SOCIAL = {"image": ("whatsapp", "instagram", "reshared"), "video": ("messenger",)}
CONDITIONS = [
    ("Images, original", "image", ("original",)),
    ("Images, social copies", "image", SOCIAL["image"]),
    ("Videos, original", "video", ("original",)),
    ("Videos, messenger copy", "video", SOCIAL["video"]),
]
VIDEO_EXT = (".mp4", ".mov", ".webm", ".avi")
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
        ".gif": "image/gif", ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
        ".avi": "video/x-msvideo"}
RETRY_STATUS = {429, 500, 502, 503, 504}


class Fatal(Exception):
    """Stops the run (bad key, no credits)."""


# ---------------------------------------------------------------------------------------------- data

def http_get(url: str, dest: str, tries: int = 5) -> None:
    for attempt in range(tries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                os.replace(tmp, dest)
                return
        except requests.RequestException as e:
            if attempt == tries - 1 or (isinstance(e, requests.HTTPError) and e.response.status_code < 429):
                raise
            time.sleep(2 ** attempt)


def local_file(data_dir: str, rev: str, path: str) -> str:
    dest = os.path.join(data_dir, path)
    if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        http_get(HF_FILE.format(repo=REPO, rev=rev, path=path), dest)
    return dest


def load_metadata(data_dir: str, rev: str) -> list:
    with open(local_file(data_dir, rev, "metadata.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------------------------------ adapters

class WasItAIGenerated:
    """WasItAIGenerated API: POST the file, use detailed.scores.ai / 100 as the probability.

    The API rejects images over 10 MB and clips over 4 MB; those files are skipped and reported as
    unscored. Every request uses credits, and a request retried after a server error can use them again.
    """

    BASE = "https://www.wasitaigenerated.com/api/v1/detect/"
    MAX_BYTES = {"image": 10 * 1024 * 1024, "video": 4 * 1024 * 1024}

    def __init__(self):
        key = os.environ.get("WAI_API_KEY")
        if not key:
            raise SystemExit("Set WAI_API_KEY to your WasItAIGenerated API key.")
        self.headers = {"Authorization": f"Bearer {key}"}

    def __call__(self, path: str) -> float:
        ext = os.path.splitext(path)[1].lower()
        media = "video" if ext in VIDEO_EXT else "image"
        size = os.path.getsize(path)
        if size > self.MAX_BYTES[media]:
            raise ValueError(f"{size / 1e6:.1f} MB is over the API limit for {media}")
        for attempt in range(6):
            try:
                with open(path, "rb") as f:
                    r = requests.post(self.BASE + media, headers=self.headers, timeout=300,
                                      files={"file": (os.path.basename(path), f, MIME.get(ext, "application/octet-stream"))})
            except (requests.ConnectionError, requests.Timeout):
                r = None
            if r is not None and r.ok:
                return float(r.json()["detailed"]["scores"]["ai"]) / 100
            if r is not None and r.status_code in (401, 402, 403):
                raise Fatal(f"HTTP {r.status_code}: {r.text[:300]}")
            if r is not None and r.status_code not in RETRY_STATUS:
                raise ValueError(f"HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(min(60, 2 ** attempt))
        raise ValueError("gave up after 6 attempts")


def load_adapter(spec: str):
    if spec == "wai":
        return WasItAIGenerated()
    target, _, func = spec.rpartition(":")
    if not target or not func:
        raise SystemExit("--adapter must be 'wai', 'file.py:function' or 'module:function'")
    if target.endswith(".py"):
        mod_spec = importlib.util.spec_from_file_location("detector_adapter", target)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return getattr(mod, func)


# ------------------------------------------------------------------------------------------- scoring

def read_scores(path: str, meta: list) -> dict:
    """Map file -> score. Accepts columns file,score or id,score (plus optional variant, default original)."""
    by_id = {(m["id"], m["variant"]): m["file"] for m in meta}
    scores = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("score") in (None, ""):
                continue
            file = row.get("file") or by_id.get((row.get("id"), row.get("variant") or "original"))
            if file:
                scores[file] = float(row["score"])
    return scores


def run_adapter(adapter, todo: list, args) -> None:
    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    lock = threading.Lock()
    failed = 0
    with open(args.out, "a", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        if new:
            writer.writerow(["file", "id", "variant", "score"])

        def one(m):
            return m, float(adapter(local_file(args.data_dir, args.revision, m["file"])))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(one, m) for m in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    m, score = fut.result()
                except Fatal as e:
                    for f in futures:
                        f.cancel()
                    raise SystemExit(f"Stopping: {e}")
                except Exception as e:  # noqa: BLE001 - any adapter error skips the file
                    failed += 1
                    print(f"  skipped: {e}", file=sys.stderr)
                    continue
                with lock:
                    writer.writerow([m["file"], m["id"], m["variant"], f"{score:.6g}"])
                    out.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} done", file=sys.stderr)
    if failed:
        print(f"{failed} files not scored; run the same command again to retry them.", file=sys.stderr)


# ------------------------------------------------------------------------------------------- metrics

def auroc(ai: np.ndarray, real: np.ndarray) -> float:
    """Probability that a random AI file scores above a random real file (ties count half)."""
    x = np.concatenate([ai, real])
    order = np.argsort(x, kind="mergesort")
    _, inverse, counts = np.unique(x[order], return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2
    ranks = np.empty(len(x))
    ranks[order] = avg_rank[inverse]
    n_ai, n_real = len(ai), len(real)
    return float((ranks[:n_ai].sum() - n_ai * (n_ai + 1) / 2) / (n_ai * n_real))


def detected_at_fpr(ai: np.ndarray, real: np.ndarray, fpr: float = 0.01):
    """Threshold = the score of the (k+1)-th highest real file, k = floor(fpr * n_real).
    Files strictly above it count as AI, so at most k real files are flagged."""
    k = int(np.floor(fpr * len(real) + 1e-9))
    desc = np.sort(real)[::-1]
    threshold = desc[k] if k < len(desc) else -np.inf
    return float(np.mean(ai > threshold)), float(threshold), int(np.sum(real > threshold))


def report(meta: list, scores: dict, fixed_threshold) -> None:
    partial = False
    print(f"\n{'Condition':24} {'AI':>9} {'Real':>11} {'AUROC':>7} {'Det@1%FP':>9} {'thresh':>8} {'real>t':>7}")
    print("-" * 81)
    for name, media, variants in CONDITIONS:
        rows = [m for m in meta if m["media"] == media and m["variant"] in variants]
        ai = np.array([scores[m["file"]] for m in rows if m["label"] == "ai" and m["file"] in scores])
        real = np.array([scores[m["file"]] for m in rows if m["label"] == "real" and m["file"] in scores])
        n_ai = sum(m["label"] == "ai" for m in rows)
        n_real = len(rows) - n_ai
        cov = f"{len(ai):>4}/{n_ai:<4} {len(real):>5}/{n_real:<5}"
        if len(ai) == 0 or len(real) == 0:
            print(f"{name:24} {cov}   not scored")
            continue
        partial = partial or len(ai) < n_ai or len(real) < n_real
        det, t, flagged = detected_at_fpr(ai, real)
        line = f"{name:24} {cov} {auroc(ai, real):7.3f} {det:9.1%} {t:8.4g} {flagged:>7}"
        if fixed_threshold is not None:
            line += f"   @{fixed_threshold:g}: {np.mean(ai >= fixed_threshold):.1%} detected, {np.mean(real >= fixed_threshold):.1%} real flagged"
        print(line)
    print("\nAI / Real = files scored / files in the public split. real>t = real files above the 1% FP threshold.")
    if partial:
        print("Rows with unscored files are computed on the scored files only; say so when you report them.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="'wai', 'file.py:function' or 'module:function'")
    src.add_argument("--scores", help="CSV with columns id,score (optional variant) or file,score")
    ap.add_argument("--out", help="where --adapter appends scores (default scores_<adapter>.csv)")
    ap.add_argument("--data-dir", default="data", help="download folder (default ./data)")
    ap.add_argument("--revision", default="main", help="Hugging Face dataset revision (default main)")
    ap.add_argument("--media", choices=["image", "video", "all"], default="video")
    ap.add_argument("--copies", choices=["original", "social", "all"], default="all",
                    help="originals, social-media copies, or both (default all)")
    ap.add_argument("--workers", type=int, default=1, help="files scored in parallel (default 1)")
    ap.add_argument("--threshold", type=float, help="also report detection and false-positive rate at this score")
    args = ap.parse_args()

    meta = load_metadata(args.data_dir, args.revision)
    if args.scores:
        report(meta, read_scores(args.scores, meta), args.threshold)
        return 0

    name = "wai" if args.adapter == "wai" else os.path.splitext(os.path.basename(args.adapter.split(":")[0]))[0]
    args.out = args.out or f"scores_{name}.csv"
    adapter = load_adapter(args.adapter)
    done = read_scores(args.out, meta) if os.path.exists(args.out) else {}
    todo = [m for m in meta
            if m["file"] not in done
            and args.media in ("all", m["media"])
            and (args.copies == "all" or (m["variant"] == "original") == (args.copies == "original"))]
    print(f"{len(done)} files already scored in {args.out}, {len(todo)} to go.", file=sys.stderr)
    if todo:
        run_adapter(adapter, todo, args)
    report(meta, read_scores(args.out, meta), args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
