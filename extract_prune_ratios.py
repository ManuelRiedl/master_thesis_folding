import json
import torch
import os
import torch.nn as nn
from ultralytics import YOLO
from prune_yolo_global import C2f_v2  # Needed for PyTorch to successfully unpickle the model


def get_pruned_channels(pruned_model, orig_layer_name, orig_channels):
    """
    Finds the corresponding layer in the pruned model and returns its out_channels.
    Handles the special case where C2f was converted to C2f_v2.
    """
    if ".cv1.conv" in orig_layer_name and ".m." not in orig_layer_name:
        cv0_name = orig_layer_name.replace(".cv1.conv", ".cv0.conv")
        cv1_name = orig_layer_name
        try:
            m0 = pruned_model.get_submodule(cv0_name)
            m1 = pruned_model.get_submodule(cv1_name)
            return m0.out_channels + m1.out_channels
        except AttributeError:
            pass

    try:
        m = pruned_model.get_submodule(orig_layer_name)
        return m.out_channels
    except AttributeError:
        return orig_channels


def generate_full_config(pruned_model_path, output_json_path, baseline_weights="weights/yolov8m.pt"):
    print(f"Loading original baseline YOLOv8 model from {baseline_weights}...")
    orig_yolo = YOLO(baseline_weights)
    orig_model = orig_yolo.model

    print(f"Loading pruned model from {pruned_model_path}...")
    ckpt = torch.load(pruned_model_path, map_location="cpu", weights_only=False)
    pruned_model = ckpt["model"] if "model" in ckpt else ckpt

    config = {}

    # 1. Base input definition
    config["input"] = {
        "pre": None,
        "do_folding": False,
        "num_channels": 3,
        "consistent_map": None
    }

    last_main_output = "input"
    current_cv1 = None

    # Dictionary to collect all pruning ratios for block-level averaging
    block_ratios = {}

    print(f"\n{'Layer Name':<30} | {'Orig Ch':<7} | {'Pruned Ch':<9} | {'Prune Ratio':<10}")
    print("-" * 65)

    # 2. Dynamically trace all Conv2d layers in the baseline model (Pass 1)
    for name, module in orig_model.named_modules():
        if isinstance(module, nn.Conv2d):
            orig_channels = module.out_channels
            pruned_channels = get_pruned_channels(pruned_model, name, orig_channels)

            ratio = 1.0 - (pruned_channels / orig_channels)
            ratio = max(0.0, round(ratio, 4))

            # Group by block ID (e.g., 'model.4.m.0' -> block '4')
            parts = name.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                block_id = parts[1]
                if block_id not in block_ratios:
                    block_ratios[block_id] = []
                block_ratios[block_id].append(ratio)

            # --- TOPOLOGY HEURISTIC ---
            if name.endswith(".conv") and name.count(".") == 2:
                pre, cons_map = last_main_output, None
                last_main_output, current_cv1 = name, None
            elif name.endswith(".cv1.conv") and ".m." not in name:
                pre, cons_map = last_main_output, None
                current_cv1 = name
            elif ".m." in name and (".cv1.conv" in name or ".cv2.conv" in name):
                pre, cons_map = current_cv1, current_cv1
            elif name.endswith(".cv2.conv") and ".m." not in name:
                pre, cons_map = current_cv1, current_cv1
                last_main_output = name
            else:
                pre, cons_map = last_main_output, None
                last_main_output = name

            # Store the configuration skeleton (We will set do_folding and pr in Pass 2)
            config[name] = {
                "pre": pre,
                "do_folding": False,
                "num_channels": orig_channels,
                "consistent_map": cons_map,
                "raw_prune_ratio": ratio  # Kept for JSON transparency
            }

            # Console Print: Raw, unaggregated data
            print(f"{name:<30} | {orig_channels:<7} | {pruned_channels:<9} | {ratio:.2%}")

    print("-" * 65)
    print(f"\nAggregating pruning ratios into block-level averages for structurally safe folding...")

    # 3. Calculate Block Averages
    block_averages = {}
    for b_id, ratios in block_ratios.items():
        avg = sum(ratios) / len(ratios)
        block_averages[b_id] = round(avg, 4)
        print(f"   Block model.{b_id}: Calculated uniform pairing rate of {avg:.2%}")

    # 4. Apply Block Averages to Config (Pass 2)
    for name in config.keys():
        if name == "input":
            continue

        parts = name.split('.')
        if len(parts) > 1 and parts[1].isdigit():
            block_id = parts[1]
            avg_ratio = block_averages[block_id]

            # Detect head protection (Layer 22) and ignore noise thresholds (< 1%)
            safe_blocks = [4, 5, 6, 7, 8, 12, 15, 18, 21]
            is_safe_to_fold = block_id.isdigit() and int(block_id) in safe_blocks

            # Require at least 1% pruning to justify folding, AND ensure it's a safe block
            do_folding = bool(avg_ratio >= 0.01 and is_safe_to_fold)
            # Insert the pairing rate directly into the config so the folding engine uses it
            config[name]["pr"] = avg_ratio
            config[name]["do_folding"] = do_folding

    # 5. Save the generated JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nSuccessfully generated unified, pruning-guided folding config to: {output_json_path}")


if __name__ == "__main__":
    pruned_model_path = r"weights/forward_pass_repair/0.1/yolo_global_pruned_forward_pass_repair_calib5000.pt"
    output_json_path = r"config_folding\yolo_pruned_ratio_config.json"
    baseline_weights = r"weights\yolov8m.pt"

    generate_full_config(pruned_model_path, output_json_path, baseline_weights)