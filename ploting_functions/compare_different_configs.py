import os
import json
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
STATS_DIR = "../results_save/save_statistics_fix"
SAVE_DIR = "../results_save/plots/rq2_target_areas"
IMAGE_DIR = r"../coco/images/val2017"

# Updated to use YOLOv8m (Medium)
MODELS_TO_COMPARE = [
    # Baseline
    "weights/yolov8/yolov8m/yolov8m.pt",

    # Backbone (conv4_to_conv8)
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",

    # All Conv Layers
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolov8_medium_all_conv_layers_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolov8_medium_all_conv_layers_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolov8_medium_all_conv_layers_pr0.3_c2f_true_data_driven_repair_calib5000.pt",

    # Full Architecture
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolov8_medium_full_architecture_protected_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolov8_medium_full_architecture_protected_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolov8_medium_full_architecture_protected_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
]

TARGET_NAMES = {
    "conv4_to_conv8": "Backbone",
    "all_conv_layers": "All Conv Layers",
    "full_architecture_protected": "Full Architecture"
}


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
    for path in MODELS_TO_COMPARE:
        json_path = get_cache_path(path, IMAGE_DIR)

        strat_match = re.search(r'(conv4_to_conv8|all_conv_layers|full_architecture_protected)', path)
        pr_match = re.search(r'_pr(\d\.\d)_', path)

        is_baseline = "yolov8m.pt" in path

        if is_baseline:
            label = "YOLOv8m Baseline"
            strategy = "Baseline"
        else:
            raw_strat = strat_match.group(1) if strat_match else "unknown"
            pr_val = pr_match.group(1) if pr_match else "??"
            strategy = TARGET_NAMES.get(raw_strat, raw_strat)
            label = f"PR {pr_val}"

        if not os.path.exists(json_path):
            print(f"\033[93m[Warning] Cache missing for {strategy} {label}. Skipping.\033[0m")
            continue

        with open(json_path, 'r') as f:
            stats = json.load(f)
            data_points.append({
                'label': label,
                'strategy': strategy,
                'gflops': stats.get('gflops', 0.0),
                'params': stats.get('params', 0.0) / 1e6,  # Millions
                'map': stats.get('mAP50-95', 0.0),
                'is_baseline': is_baseline
            })
    return data_points


def plot_rq2_degradation_curve(points, metric_x, title, filename):
    if not points:
        print("No data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    text_bbox = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.3, boxstyle="round,pad=0.2")

    # Styling Dictionary (No edgecolors)
    styles = {
        "Backbone": {"color": "#3498db", "marker": "o", "size": 150},  # Blue Circle
        "All Conv Layers": {"color": "#e74c3c", "marker": "^", "size": 180},  # Red Triangle
        "Full Architecture": {"color": "#2ecc71", "marker": "s", "size": 150}  # Green Square
    }

    # Extract Baseline
    baseline = next((p for p in points if p['is_baseline']), None)

    if baseline:
        # Draw the dashed grey horizontal line for the baseline mAP
        ax.axhline(y=baseline['map'], color='grey', linestyle='--', linewidth=2, zorder=1)
        ax.annotate(f"Uncompressed {baseline['label']} ({baseline['map']:.4f})",
                    xy=(ax.get_xlim()[0], baseline['map']), xytext=(15, 5),
                    textcoords='offset points', color='grey', fontsize=11, fontweight='bold', zorder=2)

    # Plot the branches for each strategy
    for strat, st in styles.items():
        # Get points for this strategy
        strat_points = [p for p in points if p['strategy'] == strat]

        if not strat_points:
            continue

        # Add baseline point so the line connects back to the origin
        if baseline:
            strat_points.append(baseline)

        # Sort descending by X-axis so lines draw from baseline down to PR 0.3
        strat_points.sort(key=lambda p: p[metric_x], reverse=True)

        fx = [p[metric_x] for p in strat_points]
        fy = [p['map'] for p in strat_points]

        # Draw the connecting line matching the color of the shape
        ax.plot(fx, fy, color=st['color'], linestyle='-', linewidth=2.5, zorder=3)

        # Scatter the points (excluding the baseline point to keep it clean)
        just_folded = [p for p in strat_points if not p['is_baseline']]
        jfx = [p[metric_x] for p in just_folded]
        jfy = [p['map'] for p in just_folded]

        # edgecolors='none' removes the black outer line
        ax.scatter(jfx, jfy, c=st['color'], marker=st['marker'], s=st['size'],
                   edgecolors='none', zorder=4)

        # Add PR Labels
        for p in just_folded:
            ax.annotate(p['label'], (p[metric_x], p['map']), xytext=(0, -15),
                        textcoords='offset points', fontsize=10, ha='center', bbox=text_bbox, zorder=5)

    # Formatting
    x_label_str = "Compute Cost (GFLOPs)" if metric_x == 'gflops' else "Parameters (Millions)"
    ax.set_xlabel(x_label_str, fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy (mAP@50-95)", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Custom Legend
    custom_lines = [Line2D([0], [0], color='grey', linestyle='--', lw=2, label='Medium Baseline')]
    for strat, st in styles.items():
        custom_lines.append(Line2D([0], [0], marker=st['marker'], color='w',
                                   markerfacecolor=st['color'], markersize=12, label=strat))

    ax.legend(handles=custom_lines, loc='lower left', fontsize=11, framealpha=0.9)

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
    print("Loading RQ2 Medium data from JSON cache...")
    data = load_data()

    if data:
        print("Generating Degradation Curve by Target Area (GFLOPs)...")
        plot_rq2_degradation_curve(
            points=data,
            metric_x='gflops',
            title="RQ2: Medium Architecture Degradation by Target Area (GFLOPs)",
            filename="rq2_degradation_gflops_medium"
        )

        print("Generating Degradation Curve by Target Area (Params)...")
        plot_rq2_degradation_curve(
            points=data,
            metric_x='params',
            title="RQ2: Medium Architecture Degradation by Target Area (Params)",
            filename="rq2_degradation_params_medium"
        )