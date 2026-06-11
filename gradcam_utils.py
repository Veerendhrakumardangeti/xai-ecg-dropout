"""
gradcam_utils.py
================
Grad-CAM utilities — manual hook-based implementation matching the
professor's reference pattern:
  https://gist.github.com/abap34/2502f7eecd0c9f5b27b27d22e9e1aaf3

Key design decisions:
  - Manual GradCAM (NO pytorch_grad_cam library) using forward/backward hooks
  - heatmap = ReLU( sum_over_channels( mean_grad * activation ) ) normalised to [0,1]
  - One shared heatmap applied across all 12 leads via imshow (same as reference)
  - Plotting: blue signal + Reds imshow behind it, aspect='auto', alpha=0.8

Public API (drop-in replacement for the old library-based version):
  - ECGModelWrapper
  - get_target_layer
  - compute_cam_scores(model, x_meta, y_ecg, z_true, target_layers, device, return_maps=False)
  - save_cam_maps(model, x_meta, y_ecg, z_true, target_layers, device, save_dir, epoch, max_samples=16)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




class ECGModelWrapper(nn.Module):
    """Wraps the 2-input (meta + ECG) model into a single-input model."""
    def __init__(self, model, x_meta):
        super().__init__()
        self.model  = model
        self.x_meta = x_meta

    def forward(self, y_ecg):
        return self.model(self.x_meta, y_ecg)




def get_target_layer(model):
    """
    Returns the penultimate feature block for localised heatmaps.
    Matches reference: model.layers[-7] (second-to-last conv block).
    """
    if hasattr(model, 'ecg_features'):   # EfficientNetV2
        return model.ecg_features[-2]
    if hasattr(model, 'layer3'):          # ResNet18
        return model.layer3
    if hasattr(model, 'features'):        # MobileNetV2
        return model.features[-2]
    raise RuntimeError(
        "Cannot determine GradCAM target layer. "
        "Model must have 'ecg_features', 'layer3', or 'features'."
    )


# Manual GradCAM — pytorch_grad_cam skipped: breaks on 2-input models (shape mismatch on permute+pad)

def compute_cam_scores(model, x_meta, y_ecg, z_true, target_layers, device,
                       return_maps=False):
    """
    Batch GradCAM — single forward/backward pass for entire batch.
    Score = mean of top-10% brightest heatmap values per sample.
    Higher = more focused attention.

    Args:
        model        : the ECG model
        x_meta       : (B, meta_dim) tensor on device
        y_ecg        : (B, 12, 1000, 1) tensor on device
        z_true       : (B,) tensor of true class indices
        target_layers: ignored (kept for API compatibility)
        device       : torch.device (ignored, kept for API compatibility)
        return_maps  : if True, also returns grayscale_cam array (B, 1, W)

    Returns:
        scores          : torch.Tensor (B,)
        grayscale_cam   : np.ndarray (B, 1, W)  [only if return_maps=True]
    """
    was_training = model.training
    model.eval()

    layer = get_target_layer(model)
    _activations = {}
    _gradients   = {}

    def fwd_hook(module, inp, out):
        _activations['feat'] = out          # (B, C, H, W)

    def bwd_hook(module, grad_inp, grad_out):
        _gradients['grad'] = grad_out[0]    # (B, C, H, W)

    h_fwd = layer.register_forward_hook(fwd_hook)
    h_bwd = layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        output = model(x_meta, y_ecg)       # (B, num_classes)

        N = z_true.size(0)
        one_hot = torch.zeros_like(output)
        one_hot[torch.arange(N), z_true] = 1.0
        output.backward(gradient=one_hot)

        feat = _activations['feat']         # (B, C, H, W)
        grad = _gradients['grad']           # (B, C, H, W)

        pooled_grads = grad.mean(dim=(2, 3))                    # (B, C)
        weighted     = feat * pooled_grads[:, :, None, None]   # (B, C, H, W)
        heatmap      = weighted.sum(dim=1)                      # (B, H, W)

        heatmap_1d = F.relu(heatmap).mean(dim=1)               # (B, W)

        h_min = heatmap_1d.amin(dim=1, keepdim=True)
        h_max = heatmap_1d.amax(dim=1, keepdim=True)
        denom = torch.where(h_max > h_min, h_max - h_min, torch.ones_like(h_max))
        heatmap_norm = (heatmap_1d - h_min) / denom            # (B, W)

        k = max(1, heatmap_norm.shape[1] // 10)
        top_vals, _ = torch.topk(heatmap_norm, k, dim=1)
        scores = top_vals.mean(dim=1)                           # (B,)

    finally:
        h_fwd.remove()
        h_bwd.remove()

    if was_training:
        model.train()

    if return_maps:
        cam_np = heatmap_norm.detach().cpu().numpy()[:, None, :]  # (B, 1, W)
        return scores, cam_np

    return scores



LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

def save_cam_maps(model, x_meta, y_ecg, z_true, target_layers, device,
                  save_dir, epoch, max_samples=16):
    """
    Saves per-sample PNG — 12 leads, each with the GradCAM heatmap behind the
    ECG signal. Style matches the professor's reference gist exactly.
    """
    os.makedirs(save_dir, exist_ok=True)

    n = min(y_ecg.size(0), max_samples)

    scores, grayscale_cam = compute_cam_scores(
        model,
        x_meta[:n], y_ecg[:n], z_true[:n],
        target_layers, device,
        return_maps=True
    )

    for i in range(n):
        score   = scores[i].item()
        label   = int(z_true[i].item())
        heatmap = grayscale_cam[i]            # (1, W)

        ecg_np = y_ecg[i].detach().cpu().numpy()    # (12, 1000, 1) or (12, 1000)
        if ecg_np.ndim == 3:
            ecg_np = ecg_np[:, :, 0]               # → (12, 1000)
        L = ecg_np.shape[1]

        if heatmap.shape[1] != L:
            heatmap = np.interp(
                np.linspace(0, 1, L),
                np.linspace(0, 1, heatmap.shape[1]),
                heatmap[0]
            ).reshape(1, -1)

        fig, axes = plt.subplots(12, 1, figsize=(14, 24),
                                 gridspec_kw={'hspace': 0.05})
        fig.suptitle(
            f"Epoch {epoch} | Sample {i} | Label {label} | Score {score:.4f}",
            fontsize=11, fontweight='bold'
        )

        for lead_i in range(12):
            ax     = axes[lead_i]
            signal = ecg_np[lead_i]

            ax.set_ylim([-1.5, 1.5])
            ax.set_xlim([0, L])
            ax.set_title(LEAD_NAMES[lead_i], fontsize=7, pad=1)

            ax.plot(signal, color='b', linewidth=0.8, zorder=2)

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.imshow(
                heatmap,
                extent=[*xlim, *ylim],
                cmap='Reds', alpha=0.8, aspect='auto',
                vmin=0, vmax=1, zorder=1, origin='lower'
            )
            ax.grid(False)
            if lead_i < 11:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=5)

        out_path = os.path.join(
            save_dir, f"cam_epoch{epoch:03d}_sample{i:04d}_lbl{label}.png"
        )
        fig.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    print(f"[CAM] Saved {n} maps to {save_dir}")
