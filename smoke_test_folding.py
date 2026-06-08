"""
Smoke test for the model-folding pipeline on macOS (CPU / MPS).

Goal: prove the env is set up and the core folding primitives run end-to-end
on a tiny scope (one conv pair + a few dummy images). Does NOT run the full
experiment harness (which expects yolov8m.pt and COCO).

This file was written by https://github.com/nanguoyu
"""
import os
import sys
import tempfile

os.environ['YOLO_VERBOSE'] = 'False'

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from folding_main import (
    cumpute_cluster_matrix_u,
    merge_conv_bn,
    repair_bn_forward_pass,
)
from utils_new import CalibrationDataset
from torch.utils.data import DataLoader


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    device = pick_device()
    print(f"[smoke] device = {device}")

    y = YOLO('yolov8n.pt')
    model = y.model
    model.train(False)
    model = model.to(device)
    print(f"[smoke] loaded yolov8n; first param on {next(model.parameters()).device}")

    conv0 = model.model[0].conv      # Conv2d(3, 16)
    bn0   = model.model[0].bn        # BatchNorm2d(16)
    conv1 = model.model[1].conv      # Conv2d(16, 32)
    print(f"[smoke] before: conv0 out={conv0.out_channels}, conv1 in={conv1.in_channels}")

    x = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        y0 = model(x)
    print(f"[smoke] baseline forward OK; output type={type(y0).__name__}")

    pairing_rate = 0.5
    U = cumpute_cluster_matrix_u(conv0, bn0, conv1, pairing_rate)
    assert U.shape == (16, 8), f"U shape {U.shape} != (16, 8)"
    print(f"[smoke] U shape = {tuple(U.shape)}  (expected (16, 8))")

    new_conv0, new_bn0, _ = merge_conv_bn(conv0, bn0, None, U, order='output',
                                          name='model.0.bn')
    assert new_conv0.weight.shape[0] == 8
    assert new_bn0.weight.shape[0] == 8
    print(f"[smoke] folded conv0 weight shape = {tuple(new_conv0.weight.shape)}")
    print(f"[smoke] folded bn0 weight shape   = {tuple(new_bn0.weight.shape)}")
    model.model[0].bn = new_bn0

    _, _, new_conv1 = merge_conv_bn(None, None, conv1, U, order='input',
                                    name='model.1.conv')
    assert new_conv1.weight.shape[1] == 8
    print(f"[smoke] folded conv1 weight shape = {tuple(new_conv1.weight.shape)}")

    with torch.no_grad():
        y1 = model(x)
    print(f"[smoke] folded forward OK; output type={type(y1).__name__}")

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(10):
            arr = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(tmp, f"dummy_{i}.jpg"))
        ds = CalibrationDataset(tmp, img_size=640)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        repair_bn_forward_pass(model, loader, device,
                               folding_plan=None, max_samples=4, verbose=True)
        print("[smoke] REPAIR forward pass OK")

    print("\n[smoke] ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
