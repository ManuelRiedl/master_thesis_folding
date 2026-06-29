import os
import json
import math
import copy
import cv2
import shutil
import torch
import torch.nn as nn
from typing import Sequence, Type
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from torch.utils.data import DataLoader, Dataset
import torch_pruning as tp
import numpy as np

"""
This code is mainly from here: https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py
"""


# =============================================================================
# 1. YOLO C2F COMPATIBILITY SHIM
# =============================================================================
def _try_import_yolo_modules():
    from ultralytics.nn.modules import C2f, Conv, Bottleneck, Detect
    return C2f, Conv, Bottleneck, Detect


class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        _, Conv, Bottleneck, _ = _try_import_yolo_modules()
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
    C2f, _, _, _ = _try_import_yolo_modules()
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


# =============================================================================
# 2. DATASET & PRUNING LOGIC
# =============================================================================
class UnlabeledImageDataset(Dataset):
    def __init__(self, img_dir, imgsz=640):
        self.img_dir = img_dir
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.imgsz = imgsz

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        if r != 1:
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        padded[:img.shape[0], :img.shape[1], :] = img
        img = padded.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img).float() / 255.0


def prune_yolov8_tp(
        model: nn.Module,
        pruning_ratio: float = 0.2,
        iterative_steps: int = 1,
        imgsz: int = 640,
        ignored_layer_types: Sequence[Type[nn.Module]] = (),
        config_path: str | None = None,
        device: torch.device | str | None = None,
        verbose: bool = True,
) -> dict:
    if device is None:
        device = next(model.parameters()).device

    importance = tp.importance.GroupMagnitudeImportance(p=1)
    ignored_layers = []

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_map = json.load(f)
        prunable_names = {name for name, info in config_map.items() if info.get("do_folding") is True}
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                if name not in prunable_names:
                    ignored_layers.append(m)
        if verbose:
            print(f"JSON Config Applied: {len(prunable_names)} layers prunable. Others protected.")

    for m in model.modules():
        if isinstance(m, tuple(ignored_layer_types)) if ignored_layer_types else False:
            ignored_layers.append(m)

    example_inputs = torch.randn(1, 3, imgsz, imgsz, device=device)
    model.eval()
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    if verbose:
        print(f"\nBase MACs:   {base_macs / 1e9:.4f} G")
        print(f"Base Params: {base_params / 1e6:.4f} M")

    step_ratio = 1.0 - math.pow(1.0 - pruning_ratio, 1.0 / iterative_steps)
    for step in range(iterative_steps):
        pruner = tp.pruner.GroupNormPruner(
            model, example_inputs, importance=importance,
            iterative_steps=1, pruning_ratio=step_ratio,
            ignored_layers=ignored_layers, global_pruning=True
        )
        pruner.step()
        del pruner

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    speedup = base_macs / pruned_macs if pruned_macs > 0 else 1.0

    if verbose:
        print(f"Pruned MACs:   {pruned_macs / 1e9:.4f} G  ({speedup:.2f}x speed-up)")
        print(f"Pruned Params: {pruned_params / 1e6:.4f} M")

    return {
        "base_macs": base_macs,
        "base_params": base_params,
        "pruned_macs": pruned_macs,
        "pruned_params": pruned_params,
        "speedup": speedup,
    }


# =============================================================================
# 3. FORWARD PASS REPAIR (BN CALIBRATION)
# =============================================================================
def repair_bn_forward_pass(
        model: nn.Module,
        loader,
        device,
        config_path: str | None = None,  # Kept so your main loop doesn't break
        max_samples: int = 1000,
        verbose: bool = True,
) -> nn.Module:
    # 1. Grab ALL Batch Normalization layers in the entire model
    all_bn = {name: m for name, m in model.named_modules()
              if isinstance(m, nn.BatchNorm2d)}

    if not all_bn:
        print("[REPAIR] No BN layers found — skipping.")
        return model

    # 2. Reset running statistics for all of them
    for bn in all_bn.values():
        bn.momentum = None
        bn.reset_running_stats()

    if verbose:
        print(f"\n[REPAIR] BN Forward-Pass Recalibration")
        print(f"  Resetting ALL {len(all_bn)} BN layers in the model.")

    # 3. Push data through to recalculate means/variances
    model.train()  # Must be in train mode to update BN stats
    model_dtype = next(model.parameters()).dtype
    seen = 0

    with torch.no_grad():
        while seen < max_samples:
            for images in loader:
                images = images.to(device=device, dtype=model_dtype)
                try:
                    model(images)
                except Exception as e:
                    print(f"\n  [REPAIR] Forward pass error: {e}")
                    return model

                seen += images.shape[0]
                if verbose:
                    print(f"  Samples seen: {seen}/{max_samples}", end="\r")

                if seen >= max_samples:
                    break

    model.eval()  # Return to eval mode for safe saving/inference

    if verbose:
        print(f"\n[REPAIR] Complete — {len(all_bn)} BN layers recalibrated on {seen} samples.")

    return model


def save_yolo_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ckpt = {
        'model': model,
        'train_args': {},
        'epoch': -1,
        'nc': model.nc if hasattr(model, 'nc') else 80,
    }
    torch.save(ckpt, path)
    print(f"Saved: {path}")


# =============================================================================
# 4. MAIN PIPELINE EXECUTION
# =============================================================================
if __name__ == "__main__":

    # ── Configuration ────────────────────────────────────────────────────────
    RATIOS = [0.3, 0.2, 0.1]
    MODEL_PATH = "weights/yolov8/yolov8m/yolov8m.pt"
    JSON_CONFIG = "config_folding/yolov8_m/yolov8_medium_conv4_to_conv8.json"

    # Forward Pass Repair Config
    COCO_IMGS = "coco/images/train2017"
    CALIB_SIZE = 5000

    # Backprop Fine-Tuning Config
    TRAIN_DATA_YAML = "data.yaml"  # Ensure this dataset is accessible for training
    TRAIN_EPOCHS = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup directories
    dir_no_repair = "weights/yolov8/yolov8m/prune_without_repair"
    dir_with_repair = "weights/yolov8/yolov8m/prune_with_repair"
    dir_backprop = "weights/yolov8/yolov8m/prune_with_backprop"

    os.makedirs(dir_no_repair, exist_ok=True)
    os.makedirs(dir_with_repair, exist_ok=True)
    os.makedirs(dir_backprop, exist_ok=True)

    # Load Calibration Dataset once
    calib_dataset = UnlabeledImageDataset(COCO_IMGS, imgsz=640)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )

    for ratio in RATIOS:
        print(f"\n{'=' * 70}\nProcessing Pruning Ratio: {ratio}\n{'=' * 70}")

        # ---------------------------------------------------------
        # PHASE 1: Prune Without Repair
        # ---------------------------------------------------------
        yolo = YOLO(MODEL_PATH)
        model = yolo.model
        replace_c2f_with_c2f_v2(model)

        stats = prune_yolov8_tp(
            model,
            pruning_ratio=ratio,
            config_path=JSON_CONFIG,
            ignored_layer_types=(Detect,),
        )

        # Added l1 marker
        path_no_repair = os.path.join(dir_no_repair, f"prune_l1_yolov8_medium_conv4_to_conv8_pr{ratio}_no_repair.pt")
        save_yolo_checkpoint(model, path_no_repair)

        # ---------------------------------------------------------
        # PHASE 2: Prune With Repair (Forward Pass Only)
        # ---------------------------------------------------------
        print(f"\n  -> Running BN Forward-Pass Repair ({CALIB_SIZE} Samples)")
        model_to_repair = copy.deepcopy(model).to(device)

        repair_bn_forward_pass(
            model_to_repair,
            calib_loader,
            device,
            config_path=JSON_CONFIG,
            max_samples=CALIB_SIZE
        )

        # Added l1 marker
        path_with_repair = os.path.join(dir_with_repair,
                                        f"prune_l1_yolov8_medium_conv4_to_conv8_pr{ratio}_with_repair.pt")
        save_yolo_checkpoint(model_to_repair, path_with_repair)

        # Free memory before training
        del model_to_repair
        torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # PHASE 3: Prune With Backprop (Fine-Tuning)
        # ---------------------------------------------------------
        print(f"\n  -> Running Backward Pass Fine-Tuning ({TRAIN_EPOCHS} Epochs)")

        from ultralytics.models.yolo.detect import DetectionTrainer


        # 1. Create a custom trainer that handles the pre-loaded pruned model
        class PrunedTrainer(DetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                if isinstance(weights, torch.nn.Module):
                    model = weights
                else:
                    ckpt = torch.load(path_no_repair, map_location=self.device)
                    model = ckpt['model']
                return model.to(self.device)


        # Define project directory for Ultralytics to dump runs into
        temp_project_dir = "runs/pruning_finetune"
        train_name = f"pr{ratio}_l1_finetuned"  # Added l1 to the run folder name!

        # 2. Initialize our custom trainer with the config
        trainer = PrunedTrainer(overrides={
            "data": TRAIN_DATA_YAML,
            "epochs": TRAIN_EPOCHS,
            "fraction": 0.0625,
            "imgsz": 640,
            "device": device.index if device.type == 'cuda' else 'cpu',
            "project": temp_project_dir,
            "name": train_name,
            "model": path_no_repair,
            "exist_ok": True
        })

        # 3. Run the training loop safely
        trainer.train()

        # 4. Fetch the trained weights and copy them to your designated directory
        # FIX: Ultralytics always saves the best epoch as 'prune_yolov8_medium_conv4_to_conv8_l1_pr0.1_with_backprop.pt'
        best_weights_path = os.path.join(temp_project_dir, train_name, "weights", "prune_yolov8_medium_conv4_to_conv8_l1_pr0.1_with_backprop.pt")

        # Added l1 marker and fixed dynamic ratio
        final_backprop_path = os.path.join(dir_backprop,
                                           f"prune_l1_yolov8_medium_conv4_to_conv8_pr{ratio}_with_backprop.pt")

        if os.path.exists(best_weights_path):
            shutil.copy(best_weights_path, final_backprop_path)
            print(f"Saved Fine-Tuned Model: {final_backprop_path}")
        else:
            print(f"\033[91mError: Fine-tuning completed but couldn't locate {best_weights_path}\033[0m")

        # Free memory for the next ratio
        del trainer
        torch.cuda.empty_cache()
