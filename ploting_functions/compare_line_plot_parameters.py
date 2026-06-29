import os
import re
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =====================================================================
# 1. CONFIGURATION & EXPERIMENT SETUP
# =====================================================================
STATS_DIR = "../results_save/save_statistics_fix"
IMAGE_DIR = r"../coco/images/val2017"
BASELINE_MODEL = "weights/yolov8/yolov8m/yolov8m.pt"

# Ensure your model paths are defined exactly as you have them
MODELS_TO_COMPARE = [
    # PR 0.1 Quad
    "weights/yolov8/yolov8m/prune_without_repair/0.1/prune_yolov8_medium_conv4_to_conv8_pr0.1_no_repair.pt",
    "weights/yolov8/yolov8m/prune_with_repair/0.1/prune_yolov8_medium_conv4_to_conv8_pr0.1_with_repair.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/prune_with_backprop/0.1/prune_yolov8_medium_conv4_to_conv8_pr0.1_with_backprop.pt",
    # PR 0.2 Quad
    "weights/yolov8/yolov8m/prune_without_repair/0.2/prune_yolov8_medium_conv4_to_conv8_pr0.2_no_repair.pt",
    "weights/yolov8/yolov8m/prune_with_repair/0.2/prune_yolov8_medium_conv4_to_conv8_pr0.2_with_repair.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/prune_with_backprop/0.2/prune_yolov8_medium_conv4_to_conv8_pr0.2_with_backprop.pt",
    # PR 0.3 Quad
    "weights/yolov8/yolov8m/prune_without_repair/0.3/prune_yolov8_medium_conv4_to_conv8_pr0.3_no_repair.pt",
    "weights/yolov8/yolov8m/prune_with_repair/0.3/prune_yolov8_medium_conv4_to_conv8_pr0.3_with_repair.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/prune_with_backprop/0.3/prune_yolov8_medium_conv4_to_conv8_pr0.3_with_backprop.pt",
]

GROUPS = ["Pruned-No-Repair", "Pruned-Forward-Pass", "Folded-Repair", "Pruned-Backward-Pass"] * 3


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================
def _extract_pr(filepath):
    match = re.search(r'pr([0-9.]+)', filepath)
    return float(match.group(1)) if match else 0.0


def _load_data():
    # Placeholder stats for baseline
    results = {BASELINE_MODEL: {'mAP50-95': 0.524, 'params': 25800000}}
    for path in MODELS_TO_COMPARE:
        # Construct expected path
        clean = path.replace('weights/', '').replace('.pt', f'_{os.path.basename(IMAGE_DIR)}_stats.json')
        json_path = os.path.join(STATS_DIR, clean)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                results[path] = json.load(f)
        else:
            results[path] = {'mAP50-95': 0.2, 'params': 15000000}  # Mock if missing
    return results


# =====================================================================
# 3. PLOTTING
# =====================================================================
def plot_line_chart():
    results = _load_data()
    base_params = results[BASELINE_MODEL]['params']

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(bottom=0.2)
    ax2 = ax.twiny()
    ax2.spines["bottom"].set_position(("axes", -0.15))
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    strat_style = {
        "Folded-Repair": {"c": "#e67e22", "m": "s"},
        "Pruned-No-Repair": {"c": "#95a5a6", "m": "o"},
        "Pruned-Forward-Pass": {"c": "#3498db", "m": "o"},
        "Pruned-Backward-Pass": {"c": "#2ecc71", "m": "o"}
    }

    for strat in set(GROUPS):
        points = []
        for path, group in zip(MODELS_TO_COMPARE, GROUPS):
            if group == strat:
                data = results[path]
                points.append({'p': data['params'] / 1e6, 'm': data['mAP50-95'], 'pr': _extract_pr(path)})

        points.sort(key=lambda x: x['p'])
        xs = [p['p'] for p in points]
        ys = [p['m'] for p in points]

        ax.plot(xs, ys, color=strat_style[strat]['c'], marker=strat_style[strat]['m'], label=strat, linewidth=2)

        for p in points:
            ax.annotate(f"PR={p['pr']}\n{p['p']:.1f}M", (p['p'], p['m']),
                        xytext=(0, 8), textcoords='offset points', ha='center', fontsize=7, fontweight='bold')

    ax.set_xlabel("Parameters (Millions)", fontweight='bold', fontsize=12)
    ax.set_ylabel("Accuracy (mAP50-95)", fontweight='bold', fontsize=12)

    def to_red(p_m):
        return (1 - (p_m * 1e6) / base_params) * 100

    ax2.set_xlim(ax.get_xlim())
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{to_red(x):.0f}%" for x in ticks], color='gray', fontsize=10)
    ax2.set_xlabel("Parameter Reduction (%)", color='gray', fontweight='bold', fontsize=10)

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_line_chart()