# INT8 Detection Under Degradation

## Overview

This project quantizes a small object detector (YOLOv8n) from FP32 to INT8 using OpenVINO/NNCF,
then tests whether INT8 quantization makes the model *more* fragile than FP32 when input images are
degraded — motion blur, low light, JPEG compression, and downscaling. It closes with one targeted
intervention to address the worst-performing degradation.

## Problem Statement

Quantization is a standard way to shrink and speed up object detectors for CPU/edge deployment, but
its accuracy cost is usually measured only on clean, well-lit benchmark images. Real deployments see
blurry, dark, compressed, or low-resolution input. This project asks: **does the FP32→INT8 accuracy
gap widen under degraded conditions, or not — and why?**

## Dataset

- 500 images sampled (seed=42) from **COCO `val2017`**, restricted to images containing at least one of
  5 target classes: `person`, `bicycle`, `car`, `traffic light`, `stop sign`
- Ground truth filtered to a matching 500-image COCO-format subset (`instances_500.json`) for
  `pycocotools` evaluation
- Four degraded copies of the same 500 images generated for Part 2 (see Approach)

## Tech Stack

- **Model:** YOLOv8n (Ultralytics) — chosen for its small size (~6.2 MB `.pt`) and clean export path to
  OpenVINO IR, making single-threaded CPU latency benchmarking practical without paid compute
- **Quantization:** OpenVINO + NNCF (post-training static quantization)
- **Evaluation:** `pycocotools` (COCOeval) for mAP@0.5, mAP@0.5:0.95, per-class AP, AP by object size
- **Image processing:** OpenCV, NumPy
- **Environment:** Google Colab (CPU), AMD EPYC 7B12

## Project Structure

```
├── .gitignore                 
├── INT8_Detection_Under_Degradation.ipynb
├── INT8_Detection_Under_Degradation_Report.pdf 
└── README.md
```

## Approach

**Setup** — Sampled 500 COCO images across the 5 target classes, established the FP32 baseline
(mAP@0.5, mAP@0.5:0.95, per-class AP).

**Part 1 — Quantize** — Exported YOLOv8n to OpenVINO IR, ran NNCF post-training static quantization
calibrated on the same 500 images, then compared FP32 vs INT8 on: overall mAP, per-class AP delta,
AP by object size (small/medium/large), single-threaded CPU latency (mean + P95), and model size on disk.

**Part 2 — Degrade** — Built four degraded copies of the dataset:
- **Motion blur** — horizontal box kernel, size 15
- **Low light** — gamma correction, γ = 2.0
- **JPEG compression** — quality 30
- **Downscale** — resized to 50%, then upscaled back to original resolution

Ran both FP32 and INT8 across all four conditions, producing a 5×2 mAP@0.5 table (clean + 4
degradations × FP32/INT8), and computed the "gap widening vs. clean" for each — i.e. whether INT8's
accuracy penalty relative to FP32 grows or shrinks under degradation.

**Part 3 — Fix one thing** — Picked motion blur (worst absolute accuracy of any condition) and applied
one intervention: recalibrated NNCF quantization using a calibration set built from motion-blurred
versions of the same 500 images, hypothesizing that clean-image calibration mis-estimates activation
clipping ranges for blurred inputs.

## Results

### Part 1 — FP32 vs INT8 (clean images)

| Metric | FP32 | INT8 |
|---|---|---|
| mAP@0.5 | 0.527 | 0.450 |
| mAP@0.5:0.95 | 0.379 | 0.289 |
| AP – small objects | 0.153 | 0.082 |
| AP – medium objects | 0.518 | 0.421 |
| AP – large objects | 0.673 | 0.620 |
| Mean CPU latency (1 thread) | 395.4 ms | 177.2 ms |
| P95 CPU latency (1 thread) | 420.2 ms | 193.7 ms |
| Model size on disk (OpenVINO IR) | 12.34 MB | 3.59 MB |

**CPU:** AMD EPYC 7B12. **Speedup:** 2.23×. **Size reduction:** 70.9%.

**Per-class AP delta (IoU=0.5):**

| Class | FP32 | INT8 | Δ |
|---|---|---|---|
| person | 0.642 | 0.587 | −0.055 |
| bicycle | 0.467 | 0.382 | −0.085 |
| car | 0.508 | 0.454 | −0.054 |
| traffic light | 0.333 | 0.214 | **−0.119** |
| stop sign | 0.685 | 0.614 | −0.071 |

Traffic light — the smallest, thinnest object class — takes the largest hit, consistent with the
small-object AP drop above (0.153 → 0.082).

### Part 2 — Degradation (mAP@0.5, 5×2 table)

| Condition | FP32 | INT8 | FP32 − INT8 gap |
|---|---|---|---|
| Clean | 0.527 | 0.450 | 0.077 |
| Motion blur (k=15) | 0.264 | 0.242 | 0.022 |
| Low light (γ=2.0) | 0.579 | 0.510 | 0.069 |
| JPEG Q30 | 0.542 | 0.478 | 0.065 |
| Downscale 50%→100% | 0.567 | 0.509 | 0.058 |

**Does INT8 lose more than FP32 under degradation? No — the gap *narrows* under every degradation,**
most sharply under motion blur (0.077 → 0.022). Likely explanation: quantization noise and these
degradations both destroy fine, high-frequency detail — once blur/JPEG/downscaling has already removed
it, INT8 has comparatively less additional detail left to lose relative to FP32. Motion blur remains
the worst *absolute* performer for both models (~0.25–0.26 mAP@0.5), pointing to a detector limitation
rather than a quantization-specific one.

### Part 3 — Intervention

| | Standard INT8 | Blur-calibrated INT8 |
|---|---|---|
| mAP@0.5 (motion blur) | 0.242 | 0.248 |
| mAP@0.5:0.95 (motion blur) | 0.148 | 0.152 |
| Model size | 3.59 MB | 3.59 MB |

**Result: marginal improvement (+0.006 mAP@0.5), closing only ~10% of the standard-INT8-vs-FP32 gap
(0.022 → 0.016), at zero cost in size and negligible latency cost.** This is a well-diagnosed partial
result rather than a clean win — it suggests the residual gap under motion blur is mostly information
genuinely destroyed by the blur, not a calibration-range artifact, consistent with motion blur already
having the smallest INT8-relative penalty of all four conditions.

## Installation / How to Run

```bash
pip install ultralytics openvino nncf pycocotools
```

1. Open `INT8_Detection_Under_Degradation.ipynb` in Colab (or locally, Python 3.10+).
2. Run all cells top to bottom — later cells depend on variables/files created earlier
   (`image_files`, `coco_gt`, `image_id_map`, saved `.xml`/`.bin` models, etc.).
3. Expected runtime: ~15–20 min on Colab CPU.


## Author
**Sandhiya Jeganathan** — built as a trial task exercise.

Open to **Deep learning Intern**/**Data Science Intern** / **Junior Data Scientist** / **AI-ML Engineer** roles

📍 Bengaluru, India

- 🔗 LinkedIn: https://www.linkedin.com/in/sandhiyajegan
- 📧 Email: sandhiyajeganathan2002@gmail.com
- 💻 GitHub: https://github.com/sandhiyajeganathan2002-gif

---

<div align="center">
If you found this project useful, consider giving it a ⭐
</div>

