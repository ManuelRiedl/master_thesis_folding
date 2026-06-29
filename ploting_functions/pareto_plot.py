import os
import json
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
STATS_DIR = "../results_save/save_statistics_fix"
SAVE_DIR = "../results_save/plots/multi_trajectory_analysis"
IMAGE_DIR = r"../coco/images/val2017"

MODELS_TO_COMPARE = [
    # --- Baselines ---
    "weights/yolov8/yolov8n/yolov8n.pt",
    "weights/yolov8/yolov8s/yolov8s.pt",
    "weights/yolov8/yolov8m/yolov8m.pt",
    "weights/yolov8/yolov8l/yolov8l.pt",

    # --- Folded Small ---
    "weights/yolov8/yolov8s/data_driven_repair/0.1/c2f_out_fold_true/yolov8_small_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.2/c2f_out_fold_true/yolov8_small_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.3/c2f_out_fold_true/yolov8_small_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",

    # --- Folded Medium ---
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",

    # --- Folded Large ---
    "weights/yolov8/yolov8l/data_driven_repair/0.1/c2f_out_fold_true/yolov8_large_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.2/c2f_out_fold_true/yolov8_large_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.3/c2f_out_fold_true/yolov8_large_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
]

CUSTOM_LABELS = [
    # Baselines
    "Nano", "Small", "Medium", "Large",
    # Small Series
    "S-PR 0.1", "S-PR 0.2", "S-PR 0.3",
    # Medium Series
    "M-PR 0.1", "M-PR 0.2", "M-PR 0.3",
    # Large Series
    "L-PR 0.1", "L-PR 0.2", "L-PR 0.3",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_cache_path(model_path, image_dir):
    clean_path = model_path.replace('\\', '/')
    if clean_path.startswith("weights/"):
        clean_path = clean_path[8:]

    dir_structure = os.path.dirname(clean_path)
    base_name = os.path.basename(clean_path).replace('.pt', '')
    dataset_name = os.path.basename(image_dir)

    target_dir = os.path.join(STATS_DIR, dir_structure)
    return os.path.join(target_dir, f"{base_name}_{dataset_name}_stats.json")


def load_data():
    data_points = []
    for path, label in zip(MODELS_TO_COMPARE, CUSTOM_LABELS):
        json_path = get_cache_path(path, IMAGE_DIR)

        if not os.path.exists(json_path):
            print(f"\033[93m[Warning] Cache missing for {label}. Skipping.\033[0m")
            continue

        with open(json_path, 'r') as f:
            stats = json.load(f)

            # Identify the base architecture (n, s, m, l)
            match = re.search(r'(yolov8[nsml])', path.lower())
            base_arch = match.group(1) if match else "unknown"

            is_baseline = "data_driven" not in path.lower() and "no_repair" not in path.lower()

            data_points.append({
                'label': label,
                'gflops': stats.get('gflops', 0.0),
                'params': stats.get('params', 0.0) / 1e6,  # Millions
                'map': stats.get('mAP50-95', 0.0),
                'base_arch': base_arch,
                'is_baseline': is_baseline
            })
    return data_points


def plot_multi_origin_trajectories(points, metric_x, title, filename):
    if not points:
        print("No data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 9))

    # Text styling to prevent line overlap
    text_bbox = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.3, boxstyle="round,pad=0.2")

    # 1. DRAW THE BACKBONE (Deep Blue line, Deep Blue squares with black edges)
    baseline_points = [p for p in points if p['is_baseline']]
    baseline_points.sort(key=lambda p: p[metric_x])

    b_x = [p[metric_x] for p in baseline_points]
    b_y = [p['map'] for p in baseline_points]

    ax.plot(b_x, b_y, color='#003399', linestyle='-', linewidth=3.0, zorder=3)
    ax.scatter(b_x, b_y, c='#003399', marker='s', s=150, edgecolors='black', linewidths=1.5, zorder=4)

    # Offset baselines slightly below and to the right
    for p in baseline_points:
        ax.annotate(p['label'], (p[metric_x], p['map']), xytext=(12, -12),
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    bbox=text_bbox, zorder=5)

    # 2. DRAW THE TRAJECTORIES
    colors = ['#e74c3c', '#2ecc71', '#e67e22', '#9b59b6']
    folded_bases = set(p['base_arch'] for p in points if not p['is_baseline'])

    for idx, base in enumerate(sorted(folded_bases)):
        c = colors[idx % len(colors)]
        origin_point = next((p for p in baseline_points if p['base_arch'] == base), None)
        f_points = [p for p in points if not p['is_baseline'] and p['base_arch'] == base]

        if not f_points:
            continue

        if origin_point:
            f_points.append(origin_point)

        f_points.sort(key=lambda p: p[metric_x], reverse=True)
        fx = [p[metric_x] for p in f_points]
        fy = [p['map'] for p in f_points]

        ax.plot(fx, fy, color=c, linestyle='--', linewidth=2.5, zorder=2)

        # Scatter folded points as circles with black edges
        just_folded = [p for p in f_points if not p['is_baseline']]
        jfx = [p[metric_x] for p in just_folded]
        jfy = [p['map'] for p in just_folded]

        ax.scatter(jfx, jfy, c=c, marker='o', s=110, edgecolors='black', linewidths=1.5, zorder=4)

        # Offset folded variants slightly above and to the left
        for p in just_folded:
            ax.annotate(p['label'], (p[metric_x], p['map']), xytext=(-5, 12),
                        textcoords='offset points', fontsize=10, fontweight='bold', color='#222222',
                        ha='center', bbox=text_bbox, zorder=5)

    # Formatting
    x_label_str = "Compute Cost (GFLOPs)" if metric_x == 'gflops' else "Parameters (Millions)"
    ax.set_xlabel(x_label_str, fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy (mAP@50-95)", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Custom Legend
    custom_lines = [
        Line2D([0], [0], color='#003399', lw=3.0, label='Standard YOLO Backbone'),
        Line2D([0], [0], color='gray', linestyle='--', lw=2.5, label='Structural Folding Trajectories'),
        Line2D([0], [0], marker='s', color='#003399', markeredgecolor='black', markeredgewidth=1.5, markersize=11,
               linestyle='None', label='Baseline Model'),
        Line2D([0], [0], marker='o', color='gray', markeredgecolor='black', markeredgewidth=1.5, markersize=10,
               linestyle='None', label='Folded Variant')
    ]
    ax.legend(handles=custom_lines, loc='lower right', fontsize=11, framealpha=0.9)

    # Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{filename}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\033[92m[Success] Plot saved to: {save_path}\033[0m")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading multi-scale data from JSON cache...")
    data = load_data()

    if data:
        print("Generating Cross-Scale GFLOPs Plot...")
        plot_multi_origin_trajectories(
            points=data,
            metric_x='gflops',
            title="Multi-Scale Degradation: Folded vs Natural Architecture (GFLOPs)",
            filename="multi_scale_gflops"
        )

        print("Generating Cross-Scale Parameters Plot...")
        plot_multi_origin_trajectories(
            points=data,
            metric_x='params',
            title="Multi-Scale Degradation: Folded vs Natural Architecture (Params)",
            filename="multi_scale_params"
        )