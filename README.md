# Using Explainability as a Training-Time Reliability Signal for Efficient ECG Classification

Training deep models on 12-lead ECGs is expensive, and a lot of that cost is wasted on samples the model has already mastered. This repo asks a simple question, instead of throwing data away based only on loss or confidence, what if we also ask *where the model is looking*? We use Grad-CAM as a cheap, per-sample **reliability signal** during training and drop the samples that are either already easy or that the model is "getting right for the wrong reasons."

The result is a family of progressive data-dropout schedules — with and without the explainability signal — that cut the number of samples actually backpropagated each epoch while trying to hold onto macro-F1.

It runs across three public ECG corpora (PTB-XL, CPSC 2018, Georgia 2020) and three backbones (EfficientNetV2-S, ResNet-18, MobileNetV2), and logs everything you need to compare runs: accuracy, macro-F1, effective epochs, kept-ratio curves, and Grad-CAM figures.

---

## What's in here

| File | What it does |
|------|--------------|
| `main.py` | Entry point. Parses args, loads a dataset, builds a model, runs the chosen training mode, then tests. |
| `data.py` | Multi-dataset loaders (PTB-XL, CPSC 2018, Chapman, Chapman-v2, Georgia). Reads WFDB/CSV, standardises signals, builds train/val/test loaders. |
| `model.py` | The three backbones, each adapted to single-channel 12-lead "image" input via a `build_model()` factory. |
| `baseline.py` | Standard supervised training loop — no dropout/filtering. The control to beat. |
| `selective_gradient.py` | All the progressive-dropout modes (DBPD, SMRD, SRD and their XAI variants) in one `TrainRevision` class. |
| `gradcam_utils.py` | Manual hook-based Grad-CAM (no `pytorch_grad_cam` dependency). Produces per-sample focus scores and heatmaps. |
| `visualize_gradcam.py` | Standalone script to render publication-quality Grad-CAM figures from a trained checkpoint. |
| `test.py` | Final evaluation pass — classification report, macro-F1, accuracy. |
| `utils.py` | Plotting + logging helpers (metric curves, confusion matrices, kept-ratio plots). |
| `requirements.txt` | Python dependencies. |

---

## The training modes

All modes except `baseline` live in `TrainRevision` and share the same idea: warm up on the full dataset for a few epochs, then progressively drop samples each epoch according to a rule.

| `--mode` | Name | Drop rule |
|----------|------|-----------|
| `baseline` | Baseline | None — trains on everything. |
| `train_with_revision` | DBPD | Drops samples the model is already confident/correct on (loss-/confidence-thresholded). |
| `smrd` | SMRD | Sample-matched revision dropout — keeps a budget-matched subset for a fair compute comparison. |
| `srd` | SRD | Scheduled random dropout — keeps a `decay^epoch` fraction, no signal, pure schedule. |
| `xai_dbpd` | XAI-DBPD | DBPD **plus** a Grad-CAM focus filter: keep = high `(1 − confidence) × cam_focus`. |
| `xai_smrd` | XAI-SMRD | SMRD with the same Grad-CAM focus filter. |
| `xai_srd` | XAI-SRD | SRD schedule with the Grad-CAM focus filter on top. |

The `xai_*` modes also do a **class-aware rescue**: if a filter would wipe out an entire class within a batch, the single best sample for that class is kept so rare classes don't vanish.

---

## Setup

You'll need Python 3.9+ and ideally a CUDA GPU (it runs on CPU/MPS, just slowly).

```bash
git clone <your-repo-url>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Getting the data

None of the datasets ship with this repo — download them and point `--data_path` at the root of whichever one you're using.

**PTB-XL** — https://physionet.org/content/ptb-xl/
Expected layout:
```
<data_path>/ptbxl_database.csv
<data_path>/scp_statements.csv
<data_path>/records100/...      # for --sampling_rate 100
<data_path>/records500/...      # for --sampling_rate 500
```
PTB-XL uses its native 5 diagnostic superclasses: `NORM, MI, STTC, CD, HYP`.

**CPSC 2018** — http://2018.icbeb.org/Challenge.html
**Georgia 12-Lead Challenge** — part of the PhysioNet/CinC 2020 challenge set.
Both store labels as SNOMED-CT codes in the `.hea` headers and are mapped to the unified 8-class space (`NORM, AF, LBBB, RBBB, ST, AVB, PVC, OTHER`). Georgia expects either `<data_path>/Georgia/*.hea` or a flat `<data_path>/*.hea` layout.

> Heads-up: signals are per-lead standardised at load time, padded/truncated to a fixed length (1000 samples for PTB-XL, 5000 for the WFDB challenge sets), so you don't need to pre-process anything yourself.

---

## Running it

The minimum you need is a `--mode`, a `--dataset`, a `--data_path`, and a `--model`.

**Baseline on PTB-XL with EfficientNetV2:**
```bash
python main.py \
  --mode baseline \
  --dataset ptbxl \
  --data_path /path/to/ptbxl \
  --model efficientnet \
  --epoch 100 \
  --batch_size 32 \
  --save_path ./output/ptbxl_baseline
```

**The headline method — XAI-DBPD on PTB-XL:**
```bash
python main.py \
  --mode xai_dbpd \
  --dataset ptbxl \
  --data_path /path/to/ptbxl \
  --model efficientnet \
  --epoch 100 \
  --warm_up_epochs 5 \
  --cam_keep_frac 0.7 \
  --cam_start_epoch 3 \
  --start_revision 50 \
  --save_path ./output/ptbxl_xai_dbpd
```

**A different backbone / dataset — XAI-SRD, ResNet-18, Georgia:**
```bash
python main.py \
  --mode xai_srd \
  --dataset georgia \
  --data_path /path/to/georgia \
  --model resnet18 \
  --srd_decay 0.99 \
  --save_path ./output/georgia_xai_srd_resnet
```

### All the flags

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` | `ptbxl` | `ptbxl`, `cpsc2018`, `chapman`, `chapman_v2`, `georgia` |
| `--mode` | *(required)* | see the modes table above |
| `--model` | `efficientnet` | `efficientnet`, `resnet18`, `mobilenetv2` |
| `--epoch` | `100` | total epochs |
| `--batch_size` | `32` | |
| `--data_path` | *(set this!)* | dataset root |
| `--save_path` | `./output` | where logs, plots and checkpoints go |
| `--sampling_rate` | `100` | PTB-XL only: `100` or `500` |
| `--warm_up_epochs` | `5` | full-dataset epochs before any filtering |
| `--start_revision` | `50` | epoch at which the main revision phase kicks in |
| `--loss_threshold` | `0.3` | confidence/loss cutoff for DBPD/SMRD modes |
| `--cam_keep_frac` | `0.7` | fraction of samples the Grad-CAM filter keeps |
| `--cam_start_epoch` | `3` | epochs after warm-up before the CAM filter activates |
| `--srd_decay` | `0.99` | per-epoch keep-fraction decay for SRD/XAI-SRD |
| `--seed` | `42` | full determinism is set in `seed_everything()` |

---

## What you get out

Everything lands in `--save_path`:

- `trained_model.pt` / `model_*.pt` — final and best (by val macro-F1) weights.
- `comparison_results.json` — accuracy, macro-F1 and **effective epochs** for the run, appended per mode so you can compare across runs in one file. (Effective epochs = total samples actually trained on ÷ dataset size — the real measure of how much compute you saved.)
- Metric curves — training/val loss and accuracy (`*_metrics.png`, `val_metrics.png`).
- `kept_ratio.png` — how much data each epoch actually retained.
- `confusion_matrix_*.png` — final confusion matrix with counts and percentages.
- `survival_log.json` / `label_log.json` — which sample indices survived each epoch and the per-class retention.

---

## Visualising Grad-CAM

Once you have a checkpoint, render the focus heatmaps over the raw ECG leads:

```bash
python visualize_gradcam.py \
  --model_path ./output/ptbxl_xai_dbpd/model_dbpd_percentile.pt \
  --data_path /path/to/ptbxl \
  --dataset ptbxl \
  --model efficientnet \
  --save_path ./figs \
  --n_samples 5
```

This produces per-sample figures with the Grad-CAM heatmap behind each of the 12 leads, plus a focus score per sample — the same signal the training filter uses.

---
