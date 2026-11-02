# AI Video Detection Benchmark

An open benchmark for **AI-generated video detectors**: 324 clips from 12 current video models (Veo 3,
Kling 3, Seedance 2, Wan 2.6, Hailuo, Grok Video and more) and free-licensed real footage, each also as a messenger copy, because
that is how clips reach moderation and investigation teams. Run by [WasItAIGenerated](https://www.wasitaigenerated.com).

- Write-up with full tables: https://www.wasitaigenerated.com/research/ai-image-video-detector-benchmark-2026
- Public split on Hugging Face: https://huggingface.co/datasets/wasitaigeneratedcom/frontier-image-video-bench-2026-10

## Headline (hidden split, October 2026)

Files are assigned by hash: 60% public, 40% hidden. These numbers come from the hidden 40%: 63 AI and
60 real clips.

| Detector | AUROC, original clips | At our production threshold |
|---|---:|---|
| WasItAI V3 video | 0.987 | 60 of 63 AI clips caught, 1 of 60 real clips flagged |

Image-only detectors can take part too: the write-up scores the same frames of each clip and averages them.

**Caveat:** we built this set and our models have seen similar content in training, so read our numbers
as a home-field result. That is why the public split exists.

## Data

Real clips are under free licenses (CC0, public domain, CC BY, CC BY-SA), with author and source for every
file.

```
videos/<id>.mp4                   originals
videos_social/messenger/<id>.mp4  messenger copies
metadata.csv   file, id, media, label (ai/real), variant, generator, license, author, source_url
```

| | Public (60%) | Hidden (40%) |
|---|---:|---:|
| AI videos | 101 | 63 |
| Real videos | 100 | 60 |

## Evaluate your own detector

```bash
pip install requests numpy
cd 2026-10
python eval.py --adapter my_detector.py:predict
```

```python
# my_detector.py: score a few frames and average them
import cv2

def predict(path: str) -> float:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores = []
    for i in range(0, total, max(total // 16, 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            scores.append(my_frame_model(frame))
    return sum(scores) / len(scores)
```

Write a function that takes a file path and returns the probability that the file is AI-generated
(0 to 1), then point `eval.py` at it. If you already have scores, pass a CSV with columns `id,score`
(add a `variant` column for the copies; without it every row counts as an original).

Scores are appended to `scores_<adapter>.csv` one file at a time, so a run can be stopped and restarted.
Other options: `--copies original|social|all`, `--workers N`, `--threshold T` (also report rates at a fixed
score), `--revision` (pin a dataset commit).

The script reports:

- **AUROC**: how well the scores separate AI from real files across all thresholds.
- **Detected @ 1% FP**: the threshold is set on the real files of the same row so that at most 1% of them
  are above it; the number is the share of AI files above it.

To run the WasItAIGenerated API on the same files (uses your API credits; the API takes uploads up to 4 MB,
larger clips are skipped and reported as unscored):

```bash
WAI_API_KEY=wai_... python eval.py --adapter wai
```

## Hidden split integrity

The hidden 40% stays private so nobody can tune a detector on it. To show it does not change after the
fact, [`2026-10/hidden-split-sha256.txt`](2026-10/hidden-split-sha256.txt) lists the SHA-256 of every hidden
file (246 clips, originals and copies), sorted. The SHA-256 of that list is

```
d00f3794c4a4a10d48940b42d3c39e2cf4425609e50ea37a0871570de7e03de4
```

If we ever add, remove or swap a hidden file, this hash changes.

## Submit your detector

Run your detector on the public split and email **patrick@wasitaigenerated.com** with your results and a way
for us to run it (weights, code or an API). We score it on the hidden split and list it on the write-up
page, including results that beat ours.

## Related benchmarks

- [ai-text-detection-benchmark](https://github.com/wasitaigenerated/ai-text-detection-benchmark): AI text detectors (GPTZero, Pangram, ZeroGPT, tropa)
- [ai-image-detection-benchmark](https://github.com/wasitaigenerated/ai-image-detection-benchmark): AI image detectors, incl. WhatsApp and Instagram copies
- Data on Hugging Face: [text, September 2026](https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09) · [images and video, October 2026](https://huggingface.co/datasets/wasitaigeneratedcom/frontier-image-video-bench-2026-10)
- Want to audit a detector, ours included? [Call for benchmarks](https://www.wasitaigenerated.com/call-for-benchmarks) (free API credits for researchers)

## License

AI-generated files: CC BY 4.0 (WasItAIGenerated). Real files keep their original license, listed per
file in `metadata.csv` with author and source; CC BY and CC BY-SA files require attribution to the listed
author when reused. Code (`eval.py`): MIT (see `LICENSE`). Results and text in this repository: CC BY 4.0 (see `LICENSE-DATA`).
