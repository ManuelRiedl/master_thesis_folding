"""
Fully automatic global structured pruning for YOLOv8.
Based on: https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py

No JSON config needed -> The model decides which channels to prune globally
using L2-norm importance ranking (GroupMagnitudeImportance).

The pruner automatically handles all layer dependencies (C2f blocks, residuals, concat, etc.)
via the Torch-Pruning DepGraph.
"""
import os
import math
import copy
import random
import itertools
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from ultralytics import YOLO
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
import torch_pruning as tp
import utils_new

# Disable ultralytics logs
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['YOLO_SKIP_CHECK'] = 'True'

# ANSI colours
C = {
    'b': '\033[94m', 'cy': '\033[96m', 'g': '\033[92m', 'y': '\033[93m',
    'r': '\033[91m', 'bold': '\033[1m', 'dim': '\033[2m', 'res': '\033[0m'
}


# ============================================================
# C2f -> C2f_v2 replacement (Required for Torch-Pruning compatibility)
# Code from: https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py
# The original C2f uses chunk() which Torch-Pruning cannot trace through the dependency graph.
# C2f_v2 splits into two separate Conv layers (cv0, cv1) so the pruner can track dependencies.
# ============================================================
class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def _infer_shortcut(bottleneck) -> bool:
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and getattr(bottleneck, "add", False)


def _transfer_weights(c2f, c2f_v2: C2f_v2) -> None:
    """Transfers weights from original C2f to C2f_v2."""
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m
    sd, sd_v2 = c2f.state_dict(), c2f_v2.state_dict()
    old_w = sd["cv1.conv.weight"]
    half = old_w.shape[0] // 2
    sd_v2["cv0.conv.weight"], sd_v2["cv1.conv.weight"] = old_w[:half], old_w[half:]
    for key in ("weight", "bias", "running_mean", "running_var"):
        old_bn = sd[f"cv1.bn.{key}"]
        sd_v2[f"cv0.bn.{key}"], sd_v2[f"cv1.bn.{key}"] = old_bn[:half], old_bn[half:]
    for key, val in sd.items():
        if not key.startswith("cv1."):
            sd_v2[key] = val
    c2f_v2.load_state_dict(sd_v2)


def replace_c2f_with_c2f_v2(module: nn.Module) -> None:
    """Recursively replaces all C2f modules with C2f_v2 for pruning compatibility."""
    for name, child in list(module.named_children()):
        if isinstance(child, C2f):
            shortcut = _infer_shortcut(child.m[0])
            c2f_v2 = C2f_v2(
                child.cv1.conv.in_channels,
                child.cv2.conv.out_channels,
                n=len(child.m),
                shortcut=shortcut,
                g=child.m[0].cv2.conv.groups,
                e=child.c / child.cv2.conv.out_channels
            )
            if hasattr(child, 'f'): c2f_v2.f = child.f
            if hasattr(child, 'i'): c2f_v2.i = child.i
            if hasattr(child, 'type'): c2f_v2.type = child.type
            _transfer_weights(child, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child)


# ============================================================
# Save / Repair
# ============================================================
def save_model(model, type_name, rate, config_name, num_calib_images=None):
    print(f"\n{C['dim']}Saving pruned model...{C['res']}")
    ckpt = {
        'model': copy.deepcopy(model).half(),
        'train_args': {},
        'epoch': -1,
    }
    if hasattr(model, 'names'):
        ckpt['names'] = model.names

    target_dir = os.path.join("weights", str(type_name), str(rate))
    os.makedirs(target_dir, exist_ok=True)

    if num_calib_images is None:
        file_name = f"{config_name}_pruned_{type_name}.pt"
    else:
        file_name = f"{config_name}_pruned_{type_name}_calib{num_calib_images}.pt"
    save_path = os.path.join(target_dir, file_name)
    torch.save(ckpt, save_path)
    print(f"{C['g']}{C['bold']}Model successfully saved to {save_path}!{C['res']}")


def repair_bn_forward_pass(model, loader, device, max_samples=1000, verbose=True):
    """Recalibrate ALL BN running statistics via a forward pass (standard REPAIR for pruning)."""
    all_bn_layers = {name: m for name, m in model.named_modules()
                     if isinstance(m, nn.BatchNorm2d)}

    if not all_bn_layers:
        if verbose:
            print(f"   {C['y']}[REPAIR] No BN layers found — skipping.{C['res']}")
        return model

    # Reset all BN running statistics
    for bn in all_bn_layers.values():
        bn.momentum = None
        bn.reset_running_stats()

    if verbose:
        print(f"\n{C['bold']}{C['cy']}--- REPAIR: BN Forward-Pass Recalibration ---{C['res']}")
        print(f"   {C['dim']}Resetting {len(all_bn_layers)} BN layers{C['res']}")

    model.train()
    seen = 0
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device=device, dtype=model_dtype)
            try:
                model(images)
            except Exception as e:
                if verbose:
                    print(f"   {C['r']}[REPAIR] Forward pass error: {e}{C['res']}")
                break
            seen += images.shape[0]
            if verbose:
                print(f"   {C['dim']}Samples seen: {seen}/{max_samples}{C['res']}", end="\r")
            if seen >= max_samples:
                break
    model.eval()
    if verbose:
        print(f"\n   {C['g']}REPAIR complete. {len(all_bn_layers)} BN layers recalibrated "
              f"on {seen} samples.{C['res']}")
    return model


# ============================================================
# Global Pruning
# ============================================================
def prune_model_global(model, pruning_ratio, iterative_steps=1, imgsz=640, device=None):
    """
    Fully automatic global structured pruning.
    Uses L2-norm (GroupMagnitudeImportance) to rank ALL channels across the entire model,
    then removes the least important ones up to the target pruning_ratio.

    The Detect head is always ignored (pruning detection outputs would break YOLO).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    example_inputs = torch.randn(1, 3, imgsz, imgsz, device=device)

    # Importance metric: L2-norm of grouped parameters (standard for structured pruning)
    importance = tp.importance.GroupMagnitudeImportance(p=2)

    # Ignore the Detect head -> pruning it would change the number of output classes / anchors
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, Detect):
            ignored_layers.append(m)

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"\n{C['bold']}[BEFORE PRUNING]{C['res']}")
    print(f"   MACs:   {base_macs / 1e9:.4f} G")
    print(f"   Params: {base_params:,}")

    # Iterative pruning: split the target ratio across multiple steps
    # This is mathematically equivalent to: 1 - (1 - step_ratio)^steps = pruning_ratio
    step_ratio = 1.0 - math.pow(1.0 - pruning_ratio, 1.0 / iterative_steps)

    for step in range(iterative_steps):
        pruner = tp.pruner.GroupNormPruner(
            model, example_inputs,
            importance=importance,
            iterative_steps=1,
            pruning_ratio=step_ratio,
            ignored_layers=ignored_layers,
            global_pruning=True,  # This is the key flag -> ranks ALL channels globally
        )
        pruner.step()
        del pruner

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    reduction = (1 - pruned_params / base_params) * 100

    print(f"\n{C['bold']}[AFTER PRUNING]{C['res']}")
    print(f"   MACs:   {pruned_macs / 1e9:.4f} G  ({base_macs / pruned_macs:.2f}x speedup)")
    print(f"   Params: {pruned_params:,}  ({C['g']}{reduction:.2f}% reduction{C['res']})")

    return {
        "base_macs": base_macs,
        "base_params": base_params,
        "pruned_macs": pruned_macs,
        "pruned_params": pruned_params,
    }


# ============================================================
# Experiment Runner
# ============================================================
def run_pruning_experiment(weights_path, pruning_ratio, number_calib_images, do_repair,
                           calib_ds="coco/images/train2017", config_name="yolo_global"):
    print(f"\n{C['bold']}============================================================{C['res']}")
    print(f"{C['b']}STARTING RUN: Global Pruning | PR: {pruning_ratio} | Calib N: {number_calib_images}{C['res']}")
    print(f"{C['bold']}============================================================{C['res']}")


    yolo = YOLO(weights_path)
    model = yolo.model
    # Replace C2f with C2f_v2 for Torch-Pruning compatibility
    replace_c2f_with_c2f_v2(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using {next(model.parameters()).device}")


    # Run global pruning
    stats = prune_model_global(model, pruning_ratio=pruning_ratio, device=device)

    # Test forward pass
    utils_new.test_forward_pass(model, device)

    # Save without repair
    save_model(model, "without_repair", pruning_ratio, config_name)

    if do_repair:
        # Build the data loader for repair
        full_dataset = utils_new.COCOImageFolder(
            image_dir=calib_ds,
            imgsz=640,
            max_images=None
        )

        total_images = len(full_dataset)
        random_indices = random.sample(range(total_images), number_calib_images)
        random_subset = Subset(full_dataset, random_indices)
        train_loader = DataLoader(
            random_subset,
            batch_size=16,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        # For pruning we repair ALL BN layers (unlike folding where we skip folded ones)
        repair_bn_forward_pass(model, train_loader, device, max_samples=number_calib_images)
        save_model(model, "forward_pass_repair", pruning_ratio, config_name,
                   num_calib_images=number_calib_images)

    del model
    del yolo
    torch.cuda.empty_cache()


def main():
    WEIGHTS_PATH = "weights/yolov8m.pt"
    CALIB_DS = "coco/images/train2017"

    # Grid search parameters
    manual_experiments = [
        {
            "pruning_rates": [0.1],
            "calib_images": [1000, 5000],
            "repair": [True]
        }
    ]

    print(f"{C['cy']}Queued {len(manual_experiments)} experiment profiles.{C['res']}\n")

    for exp in manual_experiments:
        combinations = itertools.product(
            exp.get("pruning_rates", [0.1]),
            exp.get("calib_images", [1000]),
            exp.get("repair", [True])
        )

        for pr, calib_n, do_rep in combinations:
            run_pruning_experiment(
                weights_path=WEIGHTS_PATH,
                pruning_ratio=pr,
                number_calib_images=calib_n,
                do_repair=do_rep,
                calib_ds=CALIB_DS,
            )


if __name__ == "__main__":
    main()
