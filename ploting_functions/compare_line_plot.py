import os
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

# =====================================================================
# 1. CONFIGURATION & EXPERIMENT SETUP
# =====================================================================

DATASET_NAME = "val2017"
STATS_DIR = "../results_save/save_statistics_fix"
SAVE_DIR_ROOT = "../results_save/plots"
REPORT_TITLE = "YOLOv8m Backbone Compression: Folding vs. Pruning"

# --- The first entry is always treated as the BASELINE ---
BASELINE_MODEL = "weights/yolov8/yolov8m/yolov8m.pt"

MODELS_TO_COMPARE = [
    BASELINE_MODEL,

    # --- STATIC ASSIGNMENT (Flat Rate Folding: 0.05 to 0.30) ---
    "weights/yolov8/yolov8m/data_driven_repair/0.05/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_pr0.05_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.15/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_pr0.15_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.25/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_pr0.25_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",

    # --- STATIC PRUNING (Traditional Backprop Pruning: 0.1 to 0.3) ---
    "weights/yolov8/yolov8m/prune_with_backprop/0.1/prune_yolov8_medium_conv4_to_conv8_pr0.1_with_backprop.pt",
    "weights/yolov8/yolov8m/prune_with_backprop/0.2/prune_yolov8_medium_conv4_to_conv8_pr0.2_with_backprop.pt",
    "weights/yolov8/yolov8m/prune_with_backprop/0.3/prune_yolov8_medium_conv4_to_conv8_pr0.3_with_backprop.pt",

    # --- AUTO ALLOCATOR (Smart Folding) ---
    "weights/yolov8/yolov8m/data_driven_repair/auto/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_auto_target_0.0_487710_prauto_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/auto/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_auto_target_0.1_1219276_prauto_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/auto/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_auto_target_0.1_1706987_prauto_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/auto/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_auto_target_0.1_2438553_prauto_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/auto/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_auto_target_0.1_2926264_prauto_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/auto/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_auto_target_0.1_3657830_prauto_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/auto/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_auto_target_0.2_4877107_prauto_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/auto/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_auto_target_0.2_5364817_prauto_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/auto/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_auto_target_0.2_6096384_prauto_c2f_true_data_driven_repair_calib5000.pt"
]

# Total items: 1 (Baseline) + 6 (Static Fold) + 3 (Static Prune) + 9 (Auto Fold) = 19
GROUPS = ["Baseline"] + ["Static Assignment (Folding)"] * 6 + ["Static Pruning (Backprop)"] * 3 + [
    "Auto Allocator (Folding)"] * 9

GROUP_COLORS = {
    "Static Assignment (Folding)": "#3498db",  # Blue
    "Static Pruning (Backprop)": "#2ecc71",  # Green
    "Auto Allocator (Folding)": "#e67e22",  # Orange
}

GROUP_MARKERS = {
    "Static Assignment (Folding)": "o",
    "Static Pruning (Backprop)": "^",
    "Auto Allocator (Folding)": "s",
}

BACKGROUND_GROUPS: set = set()


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================

def _extract_label(filepath):
    """Gracefully extracts either the static PR or the Auto params removed."""
    if "auto_target" in filepath:
        # Pulls the parameter drop count out of the filename (e.g., 1219276)
        match = re.search(r'auto_target_[0-9.]+_([0-9]+)_prauto', filepath)
        if match:
            dropped_m = int(match.group(1)) / 1e6
            return f"Auto (-{dropped_m:.1f}M)"
        return "Auto"

    match = re.search(r'pr([0-9.]+)', filepath)
    return match.group(1) if match else "Unk"


def _extract_pr(filepath):
    auto_match = re.search(r'auto_target_([0-9.]+)', filepath)
    if auto_match:
        return float(auto_match.group(1))

    match = re.search(r'pr([0-9.]+)', filepath)
    return float(match.group(1)) if match else 0.0


def _get_c2f_status(model_paths):
    for path in model_paths:
        lp = path.lower()
        if 'c2f_out_fold_true' in lp or 'c2f_true' in lp:
            return 'c2f_out_fold_true'
        if 'c2f_out_fold_false' in lp or 'c2f_false' in lp:
            return 'c2f_out_fold_false'
    return 'c2f_mixed_or_unknown'


_SIZE_MAP = {
    'yolov8n': 'yolov8_nano',
    'yolov8s': 'yolov8_small',
    'yolov8m': 'yolov8_medium',
    'yolov8l': 'yolov8_large',
    'yolov8x': 'yolov8_xlarge',
}


def _json_cache_path(model_path):
    clean = model_path.replace('\\', '/')
    if clean.startswith('weights/'):
        clean = clean[8:]
    dir_part = os.path.dirname(clean)
    base_name = os.path.basename(clean).replace('.pt', '')
    target_dir = os.path.join(STATS_DIR, dir_part)
    primary = os.path.join(target_dir, f"{base_name}_{DATASET_NAME}_stats.json")

    alt = None
    parts = dir_part.lower().replace('\\', '/').split('/')
    size_token = None
    for part in parts:
        if part in _SIZE_MAP:
            size_token = _SIZE_MAP[part]
            break
    if size_token:
        if base_name.startswith('yolo_'):
            alt_base = size_token + '_' + base_name[len('yolo_'):]
        else:
            alt_base = size_token + '_' + base_name
        alt = os.path.join(target_dir, f"{alt_base}_{DATASET_NAME}_stats.json")

    return primary, alt


def _load_data():
    results = {}
    still_missing = []

    for path in [BASELINE_MODEL] + MODELS_TO_COMPARE:
        primary, alt = _json_cache_path(path)

        if os.path.exists(primary):
            with open(primary, 'r') as f:
                results[path] = json.load(f)
            print(f"[OK]      {primary}")
        elif alt and os.path.exists(alt):
            with open(alt, 'r') as f:
                results[path] = json.load(f)
            print(f"[OK-ALT]  {alt}")
        else:
            still_missing.append(path)
            print(f"[WARNING] Not found (primary): {primary}")
            if alt:
                print(f"[WARNING] Not found (alt):     {alt}")

    # Fallback mock data if you haven't generated the stats yet
    if still_missing:
        print("\n[INFO] Falling back to mock data for missing entries.\n")
        mock_map = {0.0: 0.524, 0.05: 0.50, 0.1: 0.47, 0.15: 0.44,
                    0.2: 0.41, 0.25: 0.38, 0.3: 0.35}
        mock_params = 25_800_000

        if BASELINE_MODEL not in results:
            results[BASELINE_MODEL] = {'mAP50-95': mock_map[0.0], 'params': mock_params}

        for path in still_missing:
            if path == BASELINE_MODEL:
                continue
            pr = _extract_pr(path)

            if "auto_target" in path:
                auto_params_match = re.search(r'auto_target_[0-9.]+_([0-9]+)_prauto', path)
                if auto_params_match:
                    params_dropped = int(auto_params_match.group(1))
                    final_params = mock_params - params_dropped
                    final_map = mock_map.get(pr, 0.4) + 0.015
                else:
                    final_params = int(mock_params * (1 - pr))
                    final_map = mock_map.get(pr, 0.3)
            elif "prune_with_backprop" in path:
                final_params = int(mock_params * (1 - pr))
                final_map = mock_map.get(pr, 0.3) + 0.025  # Slight mock bump to distinguish from static
            else:
                final_params = int(mock_params * (1 - pr))
                final_map = mock_map.get(pr, 0.3)

            results[path] = {
                'mAP50-95': final_map,
                'params': final_params,
            }

    return results


def _save_plot(fig, title_suffix):
    baseline_variant = os.path.splitext(os.path.basename(BASELINE_MODEL))[0]
    c2f_status = _get_c2f_status(MODELS_TO_COMPARE)

    target_dir = os.path.join(SAVE_DIR_ROOT, baseline_variant, c2f_status, "line_plot_standalone")
    os.makedirs(target_dir, exist_ok=True)

    safe_title = "".join(c if c.isalnum() else "_" for c in REPORT_TITLE.split('\n')[0])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(target_dir, f"{title_suffix}_{c2f_status}_{safe_title}_{timestamp}.png")

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Plot saved to: {save_path}\n")
    plt.close(fig)


# =====================================================================
# 3. PLOTTING
# =====================================================================

def plot_line_chart():
    results = _load_data()

    base_map = results[BASELINE_MODEL]['mAP50-95']
    base_params = results[BASELINE_MODEL]['params']

    # ── Collect per-group data ────────────────────────────────────────────
    group_data = {}

    for model_path, group in zip(MODELS_TO_COMPARE, GROUPS):
        if group == "Baseline": continue

        if group not in group_data:
            group_data[group] = {"sparsity": [], "map": [], "label": []}

        entry = results.get(model_path, {})
        mAP = entry.get('mAP50-95', 0.0)
        params = entry.get('params', base_params)

        # Calculate exact parameter reduction (Sparsity)
        sparsity = (1.0 - params / base_params) * 100.0

        group_data[group]["sparsity"].append(sparsity)
        group_data[group]["map"].append(mAP)
        group_data[group]["label"].append(_extract_label(model_path))

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#cccccc')
    ax.tick_params(colors='#444444')

    # ── Baseline dashed reference line ────────────────────────────────────
    ax.axhline(
        y=base_map,
        color='#404040',
        linestyle='--',
        linewidth=1.8,
        alpha=0.85,
        zorder=1,
        label=f"Baseline ({base_map:.4f})",
    )

    _pending_labels = []

    def _draw_group(group, gdata, is_bg):
        color = GROUP_COLORS.get(group, "#000000")
        marker = GROUP_MARKERS.get(group, "o")

        # Anchor everything at the baseline (0% sparsity)
        all_sparsities = [0.0] + gdata["sparsity"]
        all_maps = [base_map] + gdata["map"]
        all_labels = [""] + gdata["label"]

        # Sort values by sparsity so the line draws cleanly left-to-right
        order = np.argsort(all_sparsities)
        xs = np.array(all_sparsities)[order]
        ys = np.array(all_maps)[order]
        ls = np.array(all_labels)[order]

        ax.plot(
            xs, ys,
            marker=marker,
            color=color,
            linewidth=2.5,
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=1.2,
            label=group,
            zorder=3,
        )

        # Skip the baseline point [0] so we don't overlap labels at 0%
        for xi, yi, text_label in zip(xs[1:], ys[1:], ls[1:]):
            bin_x = round(xi, 1)
            # Use the extracted label (e.g. 0.1, 0.2, or Auto)
            display_text = f"{text_label}"
            _pending_labels.append((bin_x, xi, yi, display_text, color, is_bg))

    # Draw the paths
    for group, gdata in group_data.items():
        _draw_group(group, gdata, is_bg=False)

    # ── Stack overlapping annotations vertically ──────────────────────
    from collections import defaultdict
    _by_x = defaultdict(list)

    for bin_x, exact_xi, yi, txt, col, is_bg in _pending_labels:
        _by_x[bin_x].append((exact_xi, yi, txt, col, is_bg))

    _LINE_HEIGHT_PTS = 11
    for bin_x, items in _by_x.items():
        # Sort descending by Y so highest mAP is at the top of the stack
        items_sorted = sorted(items, key=lambda t: -t[1])

        for stack_idx, (exact_xi, yi, txt, col, is_bg) in enumerate(items_sorted):
            y_offset = 9 + stack_idx * _LINE_HEIGHT_PTS
            ax.annotate(
                txt,
                xy=(exact_xi, yi),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=col,
                fontweight="bold",
                alpha=1.0,
            )

    # ── X-Axis Formatting ──────────────────────────────────────────────────
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("Parameter Reduction (Sparsity)", labelpad=12, fontsize=12, fontweight='bold', color='#333333')
    ax.set_ylabel("$mAP_{50-95}$", fontsize=12, fontweight='bold', color='#333333')
    ax.set_title(REPORT_TITLE, fontsize=14, fontweight='bold', pad=16, color='#222222')
    ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc', zorder=0)

    # Automatically set Y-Limits to frame the data tightly
    all_map_vals = [base_map] + [v for g in group_data.values() for v in g["map"]]
    ax.set_ylim(max(0.0, min(all_map_vals) - 0.03), max(all_map_vals) + 0.04)

    legend = ax.legend(fontsize=10, framealpha=0.9, edgecolor='#cccccc', loc='lower left')
    for text in legend.get_texts():
        text.set_color('#333333')

    plt.tight_layout()
    _save_plot(fig, "rq3_foldingstatic_foldingauto_vs_pruning")


# =====================================================================
# 4. ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    plot_line_chart()