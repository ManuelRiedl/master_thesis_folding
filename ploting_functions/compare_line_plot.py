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
SAVE_DIR_ROOT = "../results_save/plots"

# --- The first entry is always treated as the BASELINE ---
BASELINE_MODEL = "weights/yolov8/yolov8m/yolov8m.pt"

# =====================================================================
# 1. STRATEGY COMPARISON: No-Repair vs. Repair vs. DD-Repair vs. Backprop
# =====================================================================

BASELINE_MODEL = "weights/yolov8/yolov8m/yolov8m.pt"

MODELS_TO_COMPARE = [
    BASELINE_MODEL,

    # --- PR 0.1 Quad ---
    "weights/yolov8/yolov8m/prune_without_repair/0.1/prune_yolov8_medium_conv4_to_conv8_pr0.1_no_repair.pt",
    "weights/yolov8/yolov8m/prune_with_repair/0.1/prune_yolov8_medium_conv4_to_conv8_pr0.1_with_repair.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/prune_with_backprop/0.1/prune_yolov8_medium_conv4_to_conv8_pr0.1_with_backprop.pt",

    # --- PR 0.2 Quad ---
    "weights/yolov8/yolov8m/prune_without_repair/0.2/prune_yolov8_medium_conv4_to_conv8_pr0.2_no_repair.pt",
    "weights/yolov8/yolov8m/prune_with_repair/0.2/prune_yolov8_medium_conv4_to_conv8_pr0.2_with_repair.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/prune_with_backprop/0.2/prune_yolov8_medium_conv4_to_conv8_pr0.2_with_backprop.pt",

    # --- PR 0.3 Quad ---
    "weights/yolov8/yolov8m/prune_without_repair/0.3/prune_yolov8_medium_conv4_to_conv8_pr0.3_no_repair.pt",
    "weights/yolov8/yolov8m/prune_with_repair/0.3/prune_yolov8_medium_conv4_to_conv8_pr0.3_with_repair.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/prune_with_backprop/0.3/prune_yolov8_medium_conv4_to_conv8_pr0.3_with_backprop.pt",
]

# We maintain 13 total items (Baseline + 12 variants)
GROUPS = (
        ["Baseline"] +
        ["No Repair Strategy"] * 3 +
        ["BN-Repair Strategy"] * 3 +
        ["Data-Driven Repair (Calib 5000)"] * 3 +
        ["Backprop Strategy"] * 3
)

GROUP_COLORS = {
    "No Repair Strategy": "#95a5a6",  # Grey
    "BN-Repair Strategy": "#3498db",  # Blue
    "Data-Driven Repair (Calib 5000)": "#e67e22",  # Orange
    "Backprop Strategy": "#2ecc71",  # Green
}

GROUP_MARKERS = {
    "No Repair Strategy": "o",
    "BN-Repair Strategy": "s",
    "Data-Driven Repair (Calib 5000)": "D",
    "Backprop Strategy": "^",
}
BACKGROUND_GROUPS: set = set()

REPORT_TITLE = "YOLOv8m Backbone Folding - Pruning VS Folding"


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================

def _extract_pr(filepath):
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

    if still_missing:
        print("\n[INFO] Falling back to mock data for still-missing entries.\n")
        mock_map = {0.0: 0.524, 0.05: 0.44, 0.1: 0.38, 0.15: 0.31,
                    0.2: 0.22, 0.25: 0.16, 0.3: 0.10, 0.4: 0.07, 0.5: 0.04}
        mock_map_r = {0.05: 0.50, 0.1: 0.47, 0.15: 0.44, 0.2: 0.41,
                      0.25: 0.38, 0.3: 0.35, 0.4: 0.30, 0.5: 0.25}
        mock_params = 43_700_000

        if BASELINE_MODEL not in results:
            results[BASELINE_MODEL] = {'mAP50-95': mock_map[0.0], 'params': mock_params}

        for path in still_missing:
            if path == BASELINE_MODEL:
                continue
            pr = _extract_pr(path)
            is_repair = 'data_driven' in path.lower()
            results[path] = {
                'mAP50-95': mock_map_r.get(pr, 0.3) if is_repair else mock_map.get(pr, 0.1),
                'params': int(mock_params * (1 - pr * 0.6)),
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

_AUTO_PALETTE = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#f39c12", "#e91e63"]
_AUTO_MARKERS = ["D", "^", "v", "P", "X", "*"]


def plot_line_chart():
    results = _load_data()

    base_map = results[BASELINE_MODEL]['mAP50-95']
    base_params = results[BASELINE_MODEL]['params']

    # ── Collect per-group data ────────────────────────────────────────────
    group_data = {}
    pr_to_sparsity = {0.0: 0.0}

    for model_path, group in zip(MODELS_TO_COMPARE, GROUPS):
        if group not in group_data:
            group_data[group] = {"pr": [], "map": [], "sparsity": []}

        entry = results.get(model_path, {})
        mAP = entry.get('mAP50-95', 0.0)
        params = entry.get('params', base_params)

        pr = _extract_pr(model_path)
        sparsity = (1.0 - params / base_params) * 100.0
        pr_to_sparsity[pr] = sparsity

        group_data[group]["pr"].append(pr)
        group_data[group]["map"].append(mAP)
        group_data[group]["sparsity"].append(sparsity)

    # ── Build sorted X-axis tick list ─────────────────────────────────────
    sorted_prs = sorted(pr_to_sparsity.keys())
    x_pos = list(range(len(sorted_prs)))
    pr_to_xpos = {pr: i for i, pr in enumerate(sorted_prs)}

    x_labels = []
    for pr in sorted_prs:
        sp = pr_to_sparsity[pr]
        if pr == 0.0:
            x_labels.append(f"Baseline\n({sp:.1f}%)")
        else:
            x_labels.append(f"{pr}\n({sp:.1f}%)")

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

    auto_color_idx = 0
    auto_marker_idx = 0
    _pending_labels = []  # (xi, yi, text, color, is_bg)

    def _draw_group(group, gdata, is_bg):
        nonlocal auto_color_idx, auto_marker_idx

        color = GROUP_COLORS.get(group, _AUTO_PALETTE[auto_color_idx % len(_AUTO_PALETTE)])
        marker = GROUP_MARKERS.get(group, _AUTO_MARKERS[auto_marker_idx % len(_AUTO_MARKERS)])
        if group not in GROUP_COLORS:  auto_color_idx += 1
        if group not in GROUP_MARKERS: auto_marker_idx += 1

        all_prs = [0.0] + gdata["pr"]
        all_maps = [base_map] + gdata["map"]

        order = np.argsort(all_prs)
        prs_sorted = np.array(all_prs)[order]
        maps_sorted = np.array(all_maps)[order]
        xs = [pr_to_xpos[p] for p in prs_sorted]

        if is_bg:
            ax.plot(
                xs, maps_sorted,
                marker=marker,
                color=color,
                linewidth=1.4,
                markersize=6,
                markeredgecolor='none',
                alpha=0.35,
                linestyle='--',
                label=f"{group} (bg)",
                zorder=2,
            )
        else:
            ax.plot(
                xs, maps_sorted,
                marker=marker,
                color=color,
                linewidth=2.5,
                markersize=8,
                markeredgecolor='white',
                markeredgewidth=1.2,
                label=group,
                zorder=3,
            )

        for xi, yi in zip(xs[1:], maps_sorted[1:]):
            _pending_labels.append((xi, yi, f"{yi:.3f}", color, is_bg))

    for group, gdata in group_data.items():
        if group in BACKGROUND_GROUPS:
            _draw_group(group, gdata, is_bg=True)

    for group, gdata in group_data.items():
        if group not in BACKGROUND_GROUPS:
            _draw_group(group, gdata, is_bg=False)

    # ── Stack overlapping annotations by x-position ──────────────────────
    from collections import defaultdict
    _by_x = defaultdict(list)

    # [FIXED] Unpacking exactly 5 values:
    for xi, yi, txt, col, is_bg in _pending_labels:
        _by_x[xi].append((yi, txt, col, is_bg))

    _LINE_HEIGHT_PTS = 11
    for xi, items in _by_x.items():
        # [FIXED] Sort descending by y (index 0) and use is_bg (index 3) to put bg items last
        items_sorted = sorted(items, key=lambda t: (t[3], -t[0]))

        # [FIXED] Correctly unpack the 4 tuple elements stored in items_sorted
        for stack_idx, (yi, txt, col, is_bg) in enumerate(items_sorted):
            y_offset = 9 + stack_idx * _LINE_HEIGHT_PTS
            ax.annotate(
                txt,
                xy=(xi, yi),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=col,
                fontweight="bold",
                alpha=0.45 if is_bg else 1.0,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9.5)
    ax.set_xlabel("Folding Ratio & Parameter Reduction", labelpad=12,
                  fontsize=12, fontweight='bold', color='#333333')
    ax.set_ylabel("$mAP_{50-95}$", fontsize=12, fontweight='bold', color='#333333')
    ax.set_title(REPORT_TITLE, fontsize=14, fontweight='bold', pad=16, color='#222222')
    ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc', zorder=0)

    all_map_vals = [base_map] + [v for g in group_data.values() for v in g["map"]]
    ax.set_ylim(max(0.0, min(all_map_vals) - 0.03), max(all_map_vals) + 0.04)

    legend = ax.legend(fontsize=10, framealpha=0.9, edgecolor='#cccccc', loc='upper right')
    for text in legend.get_texts():
        text.set_color('#333333')

    plt.tight_layout()
    _save_plot(fig, "rq1_repair_efficacy_line")


# =====================================================================
# 4. ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    plot_line_chart()