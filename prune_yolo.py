import os
import json
import math
import torch
import torch.nn as nn
from typing import Sequence, Type
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader

import torch_pruning as tp


"""
This code is mainly form here: https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py
"""

#C2f compability                                                     #
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
            # Copy Ultralytics metadata
            if hasattr(child, 'f'): c2f_v2.f = child.f
            if hasattr(child, 'i'): c2f_v2.i = child.i
            if hasattr(child, 'type'): c2f_v2.type = child.type

            _transfer_weights(child, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child)


#pruning
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

    #L2-Norm
    importance = tp.importance.GroupMagnitudeImportance(p=2)
    #ignored layers list (eg: Detection Head)
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
    #use of the pruning libary
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


#simple forwardpass to recalibrate the BN Layers
def repair_bn_forward_pass(
        model: nn.Module,
        loader,
        device,
        config_path: str | None = None,
        max_samples: int = 1000,
        verbose: bool = True,
) -> nn.Module:
    all_bn = {name: m for name, m in model.named_modules()
              if isinstance(m, nn.BatchNorm2d)}

    # Determine which BN layers to reset
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            plan = json.load(f)

        folded_convs = [n for n, cfg in plan.items() if cfg.get("do_folding")]
        affected_bns = set()
        for conv_name in folded_convs:
            if conv_name.endswith(".conv"):
                bn_name = conv_name[:-len(".conv")] + ".bn"
                if bn_name in all_bn:
                    affected_bns.add(bn_name)
                elif verbose:
                    print(f"  [REPAIR] Warning: no BN found for {conv_name} (looked for {bn_name})")

        bn_to_reset = {n: all_bn[n] for n in affected_bns}
    else:
        # No config_folding — reset all (only correct when entire model was pruned/folded)
        bn_to_reset = all_bn

    if not bn_to_reset:
        print("[REPAIR] No BN layers to reset — skipping.")
        return model

    # Reset running stats on affected BN layers only
    for bn in bn_to_reset.values():
        bn.momentum = None          # cumulative moving average
        bn.reset_running_stats()

    if verbose:
        print(f"\n[REPAIR] BN Forward-Pass Recalibration")
        print(f"  Resetting {len(bn_to_reset)}/{len(all_bn)} BN layers (folded only):")
        for name in sorted(bn_to_reset):
            print(f"    ↺ {name}")

    # Forward pass — BN accumulates fresh running stats, no weight updates
    model.train()
    model_dtype = next(model.parameters()).dtype
    seen = 0

    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device=device, dtype=model_dtype)

            try:
                model(images)
            except Exception as e:
                print(f"\n  [REPAIR] Forward pass error: {e}")
                break

            seen += images.shape[0]
            if verbose:
                print(f"  Samples seen: {seen}/{max_samples}", end="\r")

            if seen >= max_samples:
                break

    model.eval()

    if verbose:
        print(f"\n[REPAIR] Complete — {len(bn_to_reset)} BN layers recalibrated on {seen} samples.")

    return model


#save the model
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

def collate_fn(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


if __name__ == "__main__":

    MODEL_PATH  = "weights/yolov8m.pt"
    JSON_CONFIG = "config_folding/yolo_conv4_conv5.json"
    BASE_SAVE   = "weights/yolo_conv4_conv5"

    COCO_IMGS   = "coco/images/val2017"
    COCO_ANNS   = "coco/annotations/instances_val2017.json"
    RATIO       = 0.1

    yolo  = YOLO(MODEL_PATH)
    model = yolo.model
    replace_c2f_with_c2f_v2(model)

    #Prune step
    stats = prune_yolov8_tp(
        model,
        pruning_ratio=RATIO,
        config_path=JSON_CONFIG,
        ignored_layer_types=(Detect,),
    )

    reduction = (1 - stats['pruned_params'] / stats['base_params']) * 100
    print(f"\nBaseline Params: {stats['base_params']:,}")
    print(f"Pruned Params:   {stats['pruned_params']:,}  ({reduction:.2f}% reduction)")

    #save without repair
    save_yolo_checkpoint(model, f"{BASE_SAVE}_pruned_no_repair.pt")
    #COCO specific dataloader
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])


    calib_dataset = CocoDetection(COCO_IMGS, COCO_ANNS, transform=transform)
    calib_loader  = DataLoader(
        calib_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    #apply forward pass
    model = model.to("cuda")
    device = next(model.parameters()).device
    repair_bn_forward_pass(model, calib_loader, device, config_path=JSON_CONFIG, max_samples=1000)

    #save with forward pass
    save_yolo_checkpoint(model, f"{BASE_SAVE}_pruned_with_repair.pt")
