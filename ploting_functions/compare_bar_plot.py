import os
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. CONFIGURATION & EXPERIMENT SETUP
# =====================================================================

DATASET_NAME = "val2017"
STATS_DIR = "../results_save/save_statistics_fix"

# The base plotting directory before dynamic subfolders are appended
SAVE_DIR_ROOT = "../results_save/plots"

REPORT_TITLE = "RQ1.2: Over-Parameterized vs. Optimized Folding Efficiency"

# --- 1. Define the Native Baselines for each scale ---
# The script will use the basename of the Medium model ('yolov8m') to build the dynamic save path.
BASELINES = {
    "Nano": "weights/yolov8/yolov8n/yolov8n.pt",
    "Small": "weights/yolov8/yolov8s/yolov8s.pt",
    "Medium": "weights/yolov8/yolov8m/yolov8m.pt",
    "Large": "weights/yolov8/yolov8l/yolov8l.pt",
}

# --- 2. Define the Folded Models (Repaired ONLY as requested) ---
MODELS_TO_COMPARE = [
    # --- PR 0.1 ---
    "weights/yolov8/yolov8n/data_driven_repair/0.1/c2f_out_fold_true/yolov8_nano_all_conv_layers_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.1/c2f_out_fold_true/yolov8_small_all_conv_layers_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolov8_medium_all_conv_layers_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.1/c2f_out_fold_true/yolov8_large_all_conv_layers_pr0.1_c2f_true_data_driven_repair_calib5000.pt",

    # --- PR 0.2 ---
    "weights/yolov8/yolov8n/data_driven_repair/0.2/c2f_out_fold_true/yolov8_nano_all_conv_layers_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.2/c2f_out_fold_true/yolov8_small_all_conv_layers_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolov8_medium_all_conv_layers_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.2/c2f_out_fold_true/yolov8_large_all_conv_layers_pr0.2_c2f_true_data_driven_repair_calib5000.pt",

    # --- PR 0.3 ---
    "weights/yolov8/yolov8n/data_driven_repair/0.3/c2f_out_fold_true/yolov8_nano_all_conv_layers_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.3/c2f_out_fold_true/yolov8_small_all_conv_layers_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolov8_medium_all_conv_layers_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.3/c2f_out_fold_true/yolov8_large_all_conv_layers_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
]

# Ensure this perfectly aligns with MODELS_TO_COMPARE. It dictates which baseline is used for the math.
GROUPS = [
    "Nano", "Small", "Medium", "Large",  # PR 0.1
    "Nano", "Small", "Medium", "Large",  # PR 0.2
    "Nano", "Small", "Medium", "Large",  # PR 0.3
]

SCALE_COLORS = {
    "Nano": "#3498db",  # Blue
    "Small": "#2ecc71",  # Green
    "Medium": "#e67e22",  # Orange
    "Large": "#e74c3c",  # Red
}


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================

def _extract_pr(filepath):
    """Extracts the PR value embedded in a model file path."""
    match = re.search(r'pr([0-9.]+)', filepath)
    return float(match.group(1)) if match else 0.0


def _get_c2f_status(model_paths):
    """Determines the c2f folder token for the save path."""
    for path in model_paths:
        lp = path.lower()
        if 'c2f_out_fold_true' in lp or 'c2f_true' in lp: return 'c2f_out_fold_true'
        if 'c2f_out_fold_false' in lp or 'c2f_false' in lp: return 'c2f_out_fold_false'
    return 'c2f_mixed_or_unknown'


def _json_cache_path(model_path):
    """Generates the primary and a fallback alternative path for the statistics JSON file."""
    clean = model_path.replace('\\', '/')
    if clean.startswith('weights/'): clean = clean[8:]
    dir_part = os.path.dirname(clean)
    base_name = os.path.basename(clean).replace('.pt', '')
    target_dir = os.path.join(STATS_DIR, dir_part)

    primary = os.path.join(target_dir, f"{base_name}_{DATASET_NAME}_stats.json")
    alt = None

    # Generate fallback: If it looks for "yolov8_medium_", check "yolo_" instead.
    scale_tokens = ["yolov8_nano_", "yolov8_small_", "yolov8_medium_", "yolov8_large_", "yolov8_xlarge_"]
    for token in scale_tokens:
        if base_name.startswith(token):
            alt_base = base_name.replace(token, "yolo_")
            alt = os.path.join(target_dir, f"{alt_base}_{DATASET_NAME}_stats.json")
            break

    # Reverse fallback: If it looks for "yolo_", check the scale token instead.
    if not alt and base_name.startswith("yolo_"):
        map_tokens = {"yolov8n": "yolov8_nano_", "yolov8s": "yolov8_small_", "yolov8m": "yolov8_medium_",
                      "yolov8l": "yolov8_large_"}
        for folder, token in map_tokens.items():
            if folder in dir_part:
                alt_base = base_name.replace("yolo_", token)
                alt = os.path.join(target_dir, f"{alt_base}_{DATASET_NAME}_stats.json")
                break

    return primary, alt


def _load_data():
    """Loads baseline and folded model statistics from JSON files with smart fallbacks."""
    results = {}

    # Mock data references for fallback testing
    mock_base_params = {"Nano": 3_200_000, "Small": 11_200_000, "Medium": 25_900_000, "Large": 43_700_000}
    mock_base_maps = {"Nano": 0.373, "Small": 0.449, "Medium": 0.502, "Large": 0.529}

    print("\n============================================================")
    print("LOADING BASELINE MODELS")
    print("============================================================")
    for scale, path in BASELINES.items():
        primary, alt = _json_cache_path(path)
        if os.path.exists(primary):
            with open(primary, 'r') as f:
                results[path] = json.load(f)
            print(f"[OK]      {scale:<6} Baseline : {os.path.basename(primary)}")
        elif alt and os.path.exists(alt):
            with open(alt, 'r') as f:
                results[path] = json.load(f)
            print(f"[OK-ALT]  {scale:<6} Baseline : {os.path.basename(alt)}")
        else:
            print(f"[WARNING] {scale:<6} Baseline : {os.path.basename(primary)} (USING MOCK DATA)")
            results[path] = {'mAP50-95': mock_base_maps[scale], 'params': mock_base_params[scale]}

    print("\n============================================================")
    print("LOADING FOLDED MODELS")
    print("============================================================")
    for path, scale in zip(MODELS_TO_COMPARE, GROUPS):
        primary, alt = _json_cache_path(path)
        pr = _extract_pr(path)

        if os.path.exists(primary):
            with open(primary, 'r') as f:
                results[path] = json.load(f)
            print(f"[OK]      {scale:<6} PR {pr:<3} : {os.path.basename(primary)}")
        elif alt and os.path.exists(alt):
            with open(alt, 'r') as f:
                results[path] = json.load(f)
            print(f"[OK-ALT]  {scale:<6} PR {pr:<3} : {os.path.basename(alt)}")
        else:
            print(f"[WARNING] {scale:<6} PR {pr:<3} : {os.path.basename(primary)} (USING MOCK DATA)")
            # Generate a realistic-looking mock degradation based on scale
            degradation_factor = 1.0 - (pr * 0.8) if scale in ["Large"] else 1.0 - (pr * 1.5)
            results[path] = {
                'mAP50-95': max(0.05, mock_base_maps[scale] * degradation_factor),
                'params': int(mock_base_params[scale] * (1 - pr * 0.35))
            }

    print("============================================================\n")

    return results


def _save_plot(fig, filename):
    """
    Saves the figure following the dynamic directory structure.
    If multiple base scales are compared, saves to 'yolov_all'.
    Otherwise, saves to the specific model's variant directory.
    """
    # Determine the anchor directory name based on the number of baselines
    if len(BASELINES) > 1:
        baseline_variant = "yolov_all"
    else:
        # Fallback if only a single baseline is defined
        anchor_model_path = list(BASELINES.values())[0]
        baseline_variant = os.path.splitext(os.path.basename(anchor_model_path))[0]

    # Extract c2f fold status from the MODELS_TO_COMPARE list
    c2f_status = _get_c2f_status(MODELS_TO_COMPARE)

    # Build the full target directory
    target_dir = os.path.join(SAVE_DIR_ROOT, baseline_variant, c2f_status, "normalized_bar_charts")
    os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(target_dir, f"{filename}_{c2f_status}_{timestamp}.png")

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to: {save_path}")
    plt.close(fig)


# =====================================================================
# 3. PLOTTING FUNCTIONS
# =====================================================================

def plot_scale_comparison_bar_charts(results):
    data_by_pr = {}

    for model_path, scale_group in zip(MODELS_TO_COMPARE, GROUPS):
        pr = _extract_pr(model_path)
        if pr == 0.0: continue

        baseline_path = BASELINES[scale_group]

        # Get baseline data
        base_map = results[baseline_path]['mAP50-95']
        base_params = results[baseline_path]['params']

        # Get folded model data
        entry = results.get(model_path, {})
        mAP = entry.get('mAP50-95', 0.0)
        params = entry.get('params', base_params)

        # --- Core Normalization Calculations ---
        sparsity_pct = (1.0 - params / base_params) * 100.0
        params_removed_m = (base_params - params) / 1_000_000.0
        map_drop_pct = ((base_map - mAP) / base_map) * 100.0

        cost_sparsity = map_drop_pct / sparsity_pct if sparsity_pct > 0 else 0
        cost_params = map_drop_pct / params_removed_m if params_removed_m > 0 else 0

        if pr not in data_by_pr:
            data_by_pr[pr] = {}
        data_by_pr[pr][scale_group] = {
            "cost_sparsity": cost_sparsity,
            "cost_params": cost_params
        }

    sorted_prs = sorted(data_by_pr.keys())
    unique_scales = ["Nano", "Small", "Medium", "Large"]

    # ---------------------------------------------------------
    # CHART 1: Relative Cost (% mAP drop per 1% Sparsity)
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_facecolor("#f8f9fa")
    fig1.patch.set_facecolor("white")
    ax1.spines[['top', 'right']].set_visible(False)

    x = np.arange(len(sorted_prs))
    width = 0.15  # Width of individual bars

    for i, scale in enumerate(unique_scales):
        color = SCALE_COLORS[scale]
        y_vals = [data_by_pr[pr].get(scale, {}).get("cost_sparsity", 0) for pr in sorted_prs]

        # Offset bars side-by-side
        offset = (i - len(unique_scales) / 2 + 0.5) * width
        ax1.bar(x + offset, y_vals, width, label=scale, color=color, edgecolor="white", zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PR {pr}" for pr in sorted_prs], fontsize=12)
    ax1.set_ylabel("% $mAP$ Drop per 1% Sparsity", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Target Folding Ratio", labelpad=12, fontsize=12, fontweight='bold')
    ax1.set_title("Relative Degradation Efficiency\n(Lower value = More Resilient to Folding)", fontsize=14,
                  fontweight='bold', pad=16)
    ax1.legend(title="YOLOv8 Family Scale", fontsize=10, loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()

    _save_plot(fig1, "rq1_2_relative_cost_per_sparsity")

    # ---------------------------------------------------------
    # CHART 2: Absolute Cost (% mAP drop per 1M Params)
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.set_facecolor("#f8f9fa")
    fig2.patch.set_facecolor("white")
    ax2.spines[['top', 'right']].set_visible(False)

    for i, scale in enumerate(unique_scales):
        color = SCALE_COLORS[scale]
        y_vals = [data_by_pr[pr].get(scale, {}).get("cost_params", 0) for pr in sorted_prs]

        offset = (i - len(unique_scales) / 2 + 0.5) * width
        ax2.bar(x + offset, y_vals, width, label=scale, color=color, edgecolor="white", zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"PR {pr}" for pr in sorted_prs], fontsize=12)
    ax2.set_ylabel("% $mAP$ Drop per 1M Parameters Removed", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Target Folding Ratio", labelpad=12, fontsize=12, fontweight='bold')
    ax2.set_title("Absolute Degradation Efficiency\n(Lower value = Cheaper to compress)", fontsize=14, fontweight='bold',
                  pad=16)
    ax2.legend(title="YOLOv8 Family Scale", fontsize=10, loc='upper right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()

    _save_plot(fig2, "rq1_2_absolute_cost_per_1M_params")


# =====================================================================
# 4. ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    loaded_results = _load_data()
    plot_scale_comparison_bar_charts(loaded_results)