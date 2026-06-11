import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import build_model
from gradcam_utils import get_target_layer, compute_cam_scores
from data import get_class_info, load_dataset

LEAD_NAMES   = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_INDICES = [0, 1, 5, 6, 7, 11]  # I, II, aVF, V1, V2, V6

# Set dynamically in main() based on --dataset
CLASS_NAMES = []
NUM_CLASSES = 0


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM + focus score  (manual hook-based)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cam_and_scores(model, x_meta, y_ecg, z_true, target_layers, device):
    """
    Uses the manual hook-based GradCAM from gradcam_utils so we never
    depend on pytorch_grad_cam (which breaks on 2-input / non-standard models).

    Returns:
        grayscale_cam  : np.ndarray (B, 1, W)  — flattened heatmap per sample
        scores         : np.ndarray (B,)        — focus score per sample
    """
    scores_t, grayscale_cam = compute_cam_scores(
        model, x_meta, y_ecg, z_true,
        target_layers, device, return_maps=True
    )
    return grayscale_cam, scores_t.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Per-lead CAM normalisation (percentile clipping)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_per_lead(cam_1d):
    """Clip to 2nd–98th percentile then normalise to [0, 1]."""
    lo, hi = np.percentile(cam_1d, 2), np.percentile(cam_1d, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(cam_1d)
    return (np.clip(cam_1d, lo, hi) - lo) / (hi - lo)


def upsample_cam(cam_heatmap, h=12, w=1000):
    """
    cam_heatmap : (1, W) — the 1-D heatmap from the manual GradCAM.
    Returns     : (h, w) array where the same row is broadcast across all h leads.
    """
    # cam_heatmap is (1, W); interpolate to width=w, then tile h rows
    row = np.interp(
        np.linspace(0, 1, w),
        np.linspace(0, 1, cam_heatmap.shape[1]),
        cam_heatmap[0]
    )                           # (w,)
    return np.tile(row, (h, 1))  # (h, w) — same heatmap every lead


# ─────────────────────────────────────────────────────────────────────────────
# Core plotter — reference style:
#   1. blue ECG signal line.
#   2. Reds heatmap background via imshow.
# ─────────────────────────────────────────────────────────────────────────────

def plot_ecg_with_cam(ecg_signal, cam_heatmap, label, pred, score,
                      ax_list, title_prefix=""):
    """
    Plots 6 ECG leads.
    Each lead:
      - blue signal line drawn first
      - red heatmap overlaid as background (cmap='Reds', alpha=0.8)
      - colourbar on right
    Matches the style from the professor's reference notebook.
    """
    ecg        = ecg_signal[:, :, 0]              # (12, 1000)
    cam_full   = upsample_cam(cam_heatmap, 12, 1000)  # (12, 1000)
    t          = np.arange(1000) / 100.0           # seconds

    n_leads = min(6, len(ax_list))

    for i, lead_idx in enumerate(LEAD_INDICES[:n_leads]):
        ax     = ax_list[i]
        signal = ecg[lead_idx]

        # ── per-lead normalised heatmap ──────────────────────────────────
        cam_lead = normalise_per_lead(cam_full[lead_idx])  # (1000,)
        heatmap  = cam_lead.reshape(1, -1)                 # (1, 1000) for imshow

        # ── 1. draw ECG signal in blue (same colour as reference) ────────
        ax.plot(t, signal, color='blue', linewidth=0.9, zorder=2)

        # ── 2. overlay Reds heatmap as background ────────────────────────
        #    extent = [x_left, x_right, y_bottom, y_top]
        y_lo = signal.min() - 0.3
        y_hi = signal.max() + 0.3
        img = ax.imshow(
            heatmap,
            extent=[0, 10, y_lo, y_hi],
            cmap='Reds',
            alpha=0.8,
            aspect='auto',
            vmin=0, vmax=1,
            zorder=1,                    # behind the signal line
            origin='lower'
        )

        # ── axes ──────────────────────────────────────────────────────────
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(0, 10)
        ax.set_ylabel(LEAD_NAMES[lead_idx], fontsize=8, rotation=0, labelpad=22)
        ax.tick_params(labelsize=6)
        ax.grid(False)

        if i == 0:
            correct = "✓" if label == pred else "✗"
            ax.set_title(
                f"{title_prefix}  |  True: {CLASS_NAMES[label]}  "
                f"Pred: {CLASS_NAMES[pred]} {correct}  |  Focus: {score:.3f}",
                fontsize=9, fontweight='bold', pad=4
            )

        if i < n_leads - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)", fontsize=8)

    # shared colourbar on the last axis
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_list[-1], fraction=0.05, pad=0.04,
                 label='Attention', orientation='vertical')


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",     type=str,   required=True)
    parser.add_argument("--data_path",      type=str,   required=True)
    parser.add_argument("--save_path",      type=str,   required=True)
    parser.add_argument("--dataset",        type=str,   default="ptbxl",
                        choices=["ptbxl", "cpsc2018", "chapman"],
                        help="Dataset to load (default: ptbxl)")
    parser.add_argument("--cam_keep_frac",  type=float, default=0.7)
    parser.add_argument("--loss_threshold", type=float, default=0.3)
    parser.add_argument("--n_samples",      type=int,   default=5)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--model",          type=str,   default="efficientnet",
                        choices=["efficientnet", "resnet18", "mobilenetv2"])
    args = parser.parse_args()

    global CLASS_NAMES, NUM_CLASSES
    NUM_CLASSES, CLASS_NAMES = get_class_info(args.dataset)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    # ── data ────────────────────────────────────────────────────────────────
    print(f"Loading {args.dataset} from {args.data_path} ...")
    train_loader, val_loader, test_loader, _ = load_dataset(
        args.dataset, args.data_path, sampling_rate=100, batch_size=64
    )
    all_loaders = [train_loader, val_loader, test_loader]

    # ── model ────────────────────────────────────────────────────────────────
    model = build_model(args.model, num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    target_layers = [get_target_layer(model)]

    # ── full-dataset Grad-CAM pass ──────────────────────────────────────────
    print("Computing Grad-CAM scores on full dataset (train+val+test)...")
    all_ecgs, all_labels, all_preds = [], [], []
    all_cams, all_scores, all_confidences = [], [], []

    for loader in all_loaders:
        for batch in loader:
            x_meta, y_ecg, z_true = batch[:3]
            x_meta = x_meta.to(device)
            y_ecg  = y_ecg.to(device)
            z_true = z_true.to(device).long()

            with torch.no_grad():
                outputs = model(x_meta, y_ecg)
                preds   = outputs.argmax(dim=1)
                probs   = torch.softmax(outputs, dim=1)
                conf    = probs[torch.arange(z_true.size(0)), z_true]

            model.eval()
            cam_maps, cam_scores = compute_cam_and_scores(
                model, x_meta, y_ecg, z_true, target_layers, device)

            all_ecgs.append(y_ecg.cpu().numpy())
            all_labels.append(z_true.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_cams.append(cam_maps)
            all_scores.append(cam_scores)
            all_confidences.append(conf.cpu().numpy())

    all_ecgs        = np.concatenate(all_ecgs)
    all_labels      = np.concatenate(all_labels)
    all_preds       = np.concatenate(all_preds)
    all_cams        = np.concatenate(all_cams)
    all_scores      = np.concatenate(all_scores)
    all_confidences = np.concatenate(all_confidences)

    N = len(all_scores)
    print(f"Total: {N}  score range [{all_scores.min():.4f}, {all_scores.max():.4f}]"
          f"  mean={all_scores.mean():.4f}")

    # ── two-stage filter ─────────────────────────────────────────────────────
    hard_mask = all_confidences < args.loss_threshold
    easy_mask = ~hard_mask
    hard_idx  = np.where(hard_mask)[0]
    easy_idx  = np.where(easy_mask)[0]

    hard_scores     = all_scores[hard_idx]
    k_hard          = max(1, int(len(hard_scores) * args.cam_keep_frac))
    hard_sorted     = np.argsort(hard_scores)[::-1]
    cam_kept_idx    = hard_idx[hard_sorted[:k_hard]]
    cam_dropped_idx = hard_idx[hard_sorted[k_hard:]]

    k_all = max(1, int(N * args.cam_keep_frac))

    print(f"Kept={len(cam_kept_idx)}  CAM-dropped={len(cam_dropped_idx)}"
          f"  DBPD-dropped={len(easy_idx)}")

    # shuffle to get diverse samples in plots
    rng = np.random.default_rng(args.seed)
    rng.shuffle(cam_kept_idx)
    rng.shuffle(cam_dropped_idx)

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 1 — kept vs dropped ECG pairs  (FIXED: imshow heatmap style)
    # ════════════════════════════════════════════════════════════════════════
    n = min(args.n_samples, len(cam_kept_idx), len(cam_dropped_idx))
    if n == 0:
        print("[WARN] No samples for kept-vs-dropped plot.")
    else:
        for i in range(n):
            fig, axes = plt.subplots(
                6, 2, figsize=(20, 14),
                gridspec_kw={'hspace': 0.08, 'wspace': 0.35}
            )

            ki = cam_kept_idx[i]
            plot_ecg_with_cam(
                all_ecgs[ki], all_cams[ki],
                all_labels[ki], all_preds[ki], all_scores[ki],
                [axes[j, 0] for j in range(6)],
                title_prefix="✓ KEPT"
            )

            di = cam_dropped_idx[i]
            plot_ecg_with_cam(
                all_ecgs[di], all_cams[di],
                all_labels[di], all_preds[di], all_scores[di],
                [axes[j, 1] for j in range(6)],
                title_prefix="✗ DROPPED"
            )

            # column headers
            axes[0, 0].annotate("KEPT (high focus)", xy=(0.5, 1.18),
                                 xycoords='axes fraction', ha='center',
                                 fontsize=11, fontweight='bold', color='green')
            axes[0, 1].annotate("DROPPED (low focus)", xy=(0.5, 1.18),
                                 xycoords='axes fraction', ha='center',
                                 fontsize=11, fontweight='bold', color='red')

            fig.suptitle(
                f"Grad-CAM Pair {i+1}: Kept vs Dropped  "
                f"(DBPD t={args.loss_threshold}, CAM frac={args.cam_keep_frac})\n"
                f"Background colour = model attention  |  "
                f"Dark red = high attention,  White = low attention",
                fontsize=12, fontweight='bold', y=1.01
            )

            out = os.path.join(args.save_path,
                               f"gradcam_kept_vs_dropped_pair_{i+1}.png")
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {out}")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 2 — focus score histogram  (unchanged — professor approved)
    # ════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(all_scores[cam_kept_idx], bins=30, alpha=0.6, color='green',
             label=f'Kept (n={len(cam_kept_idx)})', density=True)
    if len(cam_dropped_idx):
        ax2.hist(all_scores[cam_dropped_idx], bins=30, alpha=0.6, color='orange',
                 label=f'CAM-dropped (n={len(cam_dropped_idx)})', density=True)
    ax2.hist(all_scores[easy_idx], bins=30, alpha=0.4, color='red',
             label=f'DBPD-dropped (n={len(easy_idx)})', density=True)
    ax2.set_xlabel('CAM Focus Score', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Focus Score Distribution: Full Two-Stage Filter',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    out2 = os.path.join(args.save_path, "gradcam_score_distribution.png")
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 3 — grouped class distribution  (unchanged — professor approved)
    # ════════════════════════════════════════════════════════════════════════
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    x     = np.arange(NUM_CLASSES)
    width = 0.25
    kept_counts     = [np.sum(all_labels[cam_kept_idx]    == c) for c in range(NUM_CLASSES)]
    cam_drop_counts = [np.sum(all_labels[cam_dropped_idx] == c) for c in range(NUM_CLASSES)]
    dbpd_counts     = [np.sum(all_labels[easy_idx]        == c) for c in range(NUM_CLASSES)]

    r1 = ax3.bar(x - width, kept_counts,     width, label=f'Kept (n={len(cam_kept_idx)})',
                 color='green',  alpha=0.7, edgecolor='black')
    r2 = ax3.bar(x,          cam_drop_counts, width, label=f'CAM-Dropped (n={len(cam_dropped_idx)})',
                 color='orange', alpha=0.7, edgecolor='black')
    r3 = ax3.bar(x + width,  dbpd_counts,     width, label=f'DBPD-Dropped (n={len(easy_idx)})',
                 color='red',    alpha=0.7, edgecolor='black')

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax3.annotate(str(int(h)),
                             xy=(rect.get_x() + rect.get_width()/2, h),
                             xytext=(0, 3), textcoords='offset points',
                             ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title(f'Class Distribution by Filter Stage',
                  fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    fig3.tight_layout()
    out3 = os.path.join(args.save_path, "gradcam_class_distribution.png")
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {out3}")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 4 — scatter confidence vs focus  (unchanged — professor approved)
    # ════════════════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    status = np.zeros(N, dtype=int)
    status[cam_dropped_idx] = 1
    status[easy_idx]        = 2
    colors_map = {0: 'green', 1: 'orange', 2: 'red'}
    labels_map = {0: f'Kept (n={len(cam_kept_idx)})',
                  1: f'CAM-dropped (n={len(cam_dropped_idx)})',
                  2: f'DBPD-dropped (n={len(easy_idx)})'}
    for s in [2, 1, 0]:
        m = status == s
        ax4.scatter(all_confidences[m], all_scores[m],
                    c=colors_map[s], alpha=0.4, s=10, label=labels_map[s])
    ax4.axvline(args.loss_threshold, color='blue', linestyle='--', linewidth=1.5,
                label=f'DBPD threshold ({args.loss_threshold})')
    ax4.axhline(np.sort(all_scores)[::-1][k_all - 1], color='purple',
                linestyle='--', linewidth=1.5,
                label=f'CAM threshold (top {args.cam_keep_frac*100:.0f}%)')
    ax4.set_xlabel('Model Confidence P(true class)', fontsize=12)
    ax4.set_ylabel('Grad-CAM Focus Score', fontsize=12)
    ax4.set_title('Confidence vs Focus: Two-Stage Filter',
                  fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='upper left')
    ax4.grid(alpha=0.3)
    out4 = os.path.join(args.save_path, "gradcam_scatter_confidence_vs_focus.png")
    fig4.savefig(out4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {out4}")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 5 — 4-quadrant ECG samples  (FIXED: imshow heatmap style)
    # ════════════════════════════════════════════════════════════════════════
    med_conf  = np.median(all_confidences)
    med_focus = np.median(all_scores)

    quadrants = {
        "High Conf + Good Focus\n(Easy & Clear)":
            np.where((all_confidences >= med_conf) & (all_scores >= med_focus))[0],
        "High Conf + Poor Focus\n(Easy but Noisy)":
            np.where((all_confidences >= med_conf) & (all_scores <  med_focus))[0],
        "Low Conf + Good Focus\n(Hard & Clear — BEST for training)":
            np.where((all_confidences <  med_conf) & (all_scores >= med_focus))[0],
        "Low Conf + Poor Focus\n(Hard & Noisy — DROPPED by XAI)":
            np.where((all_confidences <  med_conf) & (all_scores <  med_focus))[0],
    }

    for q_idx, (q_name, q_indices) in enumerate(quadrants.items()):
        if len(q_indices) == 0:
            print(f"[WARN] Empty quadrant: {q_name}")
            continue

        # most representative = closest to median focus score in this quadrant
        q_scores       = all_scores[q_indices]
        representative = q_indices[np.argmin(np.abs(q_scores - np.median(q_scores)))]

        fig5, axes5 = plt.subplots(
            6, 1, figsize=(14, 14),
            gridspec_kw={'hspace': 0.08}
        )

        plot_ecg_with_cam(
            all_ecgs[representative], all_cams[representative],
            all_labels[representative], all_preds[representative],
            all_scores[representative],
            list(axes5),
            title_prefix=f"{q_name.splitlines()[0]} (n={len(q_indices)})"
        )

        fig5.suptitle(
            f"Quadrant {q_idx+1}: {q_name}\n"
            f"median conf={med_conf:.3f}  |  median focus={med_focus:.3f}\n"
            f"Background = model attention  (dark red = high,  white = low)",
            fontsize=12, fontweight='bold', y=1.02
        )

        out5 = os.path.join(args.save_path, f"gradcam_quadrant_{q_idx+1}.png")
        fig5.savefig(out5, dpi=150, bbox_inches='tight')
        plt.close(fig5)
        print(f"Saved: {out5}")

    print(f"\nAll visualizations saved to {args.save_path}/")


if __name__ == "__main__":
    main()
