import os
import json
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & PATHS
# ─────────────────────────────────────────────────────────────────────────────
STATS_DIR = "../results_save/save_statistics_fix"
SAVE_DIR = "../results_save/plots/multi_trajectory_analysis"
IMAGE_DIR = r"../coco/images/val2017"

# Explicit paths for all architectures and folding targets
MODELS_TO_COMPARE = [
    # Baselines
    "weights/yolov8/yolov8n/yolov8n.pt",
    "weights/yolov8/yolov8s/yolov8s.pt",
    "weights/yolov8/yolov8m/yolov8m.pt",
    "weights/yolov8/yolov8l/yolov8l.pt",
    # Nano
    "weights/yolov8/yolov8n/data_driven_repair/0.1/c2f_out_fold_true/yolov8_nano_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8n/data_driven_repair/0.2/c2f_out_fold_true/yolov8_nano_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8n/data_driven_repair/0.3/c2f_out_fold_true/yolov8_nano_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8n/data_driven_repair/0.1/c2f_out_fold_true/yolov8_nano_all_conv_layers_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8n/data_driven_repair/0.2/c2f_out_fold_true/yolov8_nano_all_conv_layers_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8n/data_driven_repair/0.3/c2f_out_fold_true/yolov8_nano_all_conv_layers_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8n/data_driven_repair/0.1/c2f_out_fold_true/yolov8_nano_full_architecture_protected_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8n/data_driven_repair/0.2/c2f_out_fold_true/yolov8_nano_full_architecture_protected_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8n/data_driven_repair/0.3/c2f_out_fold_true/yolov8_nano_full_architecture_protected_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    # Small
    "weights/yolov8/yolov8s/data_driven_repair/0.1/c2f_out_fold_true/yolov8_small_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.2/c2f_out_fold_true/yolov8_small_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.3/c2f_out_fold_true/yolov8_small_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.1/c2f_out_fold_true/yolov8_small_all_conv_layers_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.2/c2f_out_fold_true/yolov8_small_all_conv_layers_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.3/c2f_out_fold_true/yolov8_small_all_conv_layers_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.1/c2f_out_fold_true/yolov8_small_full_architecture_protected_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.2/c2f_out_fold_true/yolov8_small_full_architecture_protected_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8s/data_driven_repair/0.3/c2f_out_fold_true/yolov8_small_full_architecture_protected_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    # Medium
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolov8_medium_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolov8_medium_all_conv_layers_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolov8_medium_all_conv_layers_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolov8_medium_all_conv_layers_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolov8_medium_full_architecture_protected_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolov8_medium_full_architecture_protected_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolov8_medium_full_architecture_protected_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    # Large
    "weights/yolov8/yolov8l/data_driven_repair/0.1/c2f_out_fold_true/yolov8_large_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.2/c2f_out_fold_true/yolov8_large_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.3/c2f_out_fold_true/yolov8_large_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.1/c2f_out_fold_true/yolov8_large_all_conv_layers_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.2/c2f_out_fold_true/yolov8_large_all_conv_layers_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.3/c2f_out_fold_true/yolov8_large_all_conv_layers_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.1/c2f_out_fold_true/yolov8_large_full_architecture_protected_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.2/c2f_out_fold_true/yolov8_large_full_architecture_protected_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
    "weights/yolov8/yolov8l/data_driven_repair/0.3/c2f_out_fold_true/yolov8_large_full_architecture_protected_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def get_cache_path(model_path):
    clean = model_path.replace('\\', '/')
    if clean.startswith("weights/"): clean = clean[8:]
    return os.path.join(STATS_DIR, os.path.dirname(clean),
                        f"{os.path.basename(clean).replace('.pt', '')}_{os.path.basename(IMAGE_DIR)}_stats.json")


def load_data():
    points = []
    for path in MODELS_TO_COMPARE:
        json_path = get_cache_path(path)
        if not os.path.exists(json_path):
            print(f"[Warning] Missing stats for: {os.path.basename(path)}")
            continue
        with open(json_path, 'r') as f:
            s = json.load(f)
            arch = re.search(r'(yolov8[nsml])', path.lower()).group(1)
            is_baseline = "data_driven" not in path.lower()
            target = "conv4_to_conv8" if "conv4" in path else (
                "full_architecture" if "full" in path else "all_conv_layers")
            points.append({
                'path': path, 'gflops': s.get('gflops', 0),
                'params': s.get('params', 0) / 1e6, 'map': s.get('mAP50-95', 0),
                'arch': arch, 'is_baseline': is_baseline, 'target': target
            })
    return points


# ─────────────────────────────────────────────────────────────────────────────
# 3. PARETO PLOTTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def plot_pareto_trajectories(points, metric_x='params'):
    fig, ax = plt.subplots(figsize=(14, 9))

    # 1. Styles
    arch_color_shades = {
        'yolov8n': ['#ff9999', '#ff4d4d', '#b30000'],
        'yolov8s': ['#99ffcc', '#2ecc71', '#008000'],
        'yolov8m': ['#99ccff', '#3498db', '#0055aa'],
        'yolov8l': ['#cc99ff', '#9b59b6', '#6600cc']
    }
    markers = {"conv4_to_conv8": "^", "full_architecture": "o", "all_conv_layers": "*"}
    target_labels = {"conv4_to_conv8": "B-PR", "full_architecture": "A-PR", "all_conv_layers": "C-PR"}

    # 2. Draw Baseline
    baselines = sorted([p for p in points if p['is_baseline']], key=lambda p: p[metric_x])
    ax.plot([p[metric_x] for p in baselines], [p['map'] for p in baselines], color='#003399', lw=4, zorder=10)
    ax.scatter([p[metric_x] for p in baselines], [p['map'] for p in baselines], c='#003399', marker='s', s=200,
               edgecolors='black', zorder=11)
    for p in baselines:
        arch_name = p['arch'].replace('yolov8', '').capitalize()
        ax.annotate(arch_name, (p[metric_x], p['map']),
                    xytext=(0, 18), textcoords='offset points',
                    ha='center', fontweight='bold', fontsize=12, color='#003399')
    # 3. Draw Folded Trajectories with Smart Labeling
    label_history = []  # Stores (x, y) to check for overlap

    for arch, shades in arch_color_shades.items():
        base_orig = next((p for p in baselines if p['arch'] == arch), None)
        if not base_orig: continue

        for i, (target, marker) in enumerate(markers.items()):
            folded = [p for p in points if not p['is_baseline'] and p['arch'] == arch and p['target'] == target]
            if not folded: continue

            color = shades[i]
            traj_x = [base_orig[metric_x]] + [p[metric_x] for p in folded]
            traj_y = [base_orig['map']] + [p['map'] for p in folded]
            ax.plot(traj_x, traj_y, color=color, linestyle='--', lw=2, zorder=1)

            for p in folded:
                ax.scatter(p[metric_x], p['map'], color=color, marker=marker, s=120, edgecolors='black', zorder=5)

                # Collision Avoidance Logic
                pr_val = re.search(r'0\.[123]', p['path']).group()
                label_text = f"{target_labels[target]} {pr_val}"

                # Dynamic Y-Offset based on label density
                offset = 10
                for hx, hy in label_history:
                    if abs(hx - p[metric_x]) < 1.5 and abs(hy - p['map']) < 0.02:
                        offset += 15  # Push up if crowded

                ax.annotate(label_text, (p[metric_x], p['map']), xytext=(0, offset),
                            textcoords='offset points', ha='center', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
                label_history.append((p[metric_x], p['map']))

    # 4. Custom Legend
    legend_elements = [
        Line2D([0], [0], color='#003399', lw=4, label='Baseline Model'),
        Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Folded Trajectory'),
        Line2D([0], [0], marker='s', color='none', markeredgecolor='black', markerfacecolor='#003399', markersize=10,
               label='Baseline'),
        Line2D([0], [0], marker='^', color='none', markeredgecolor='black', markerfacecolor='gray', markersize=10,
               label='B: Backbone'),
        Line2D([0], [0], marker='o', color='none', markeredgecolor='black', markerfacecolor='gray', markersize=10,
               label='A: Full Arch'),
        Line2D([0], [0], marker='*', color='none', markeredgecolor='black', markerfacecolor='gray', markersize=10,
               label='C: All Conv')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.set_xlabel("Parameters (Millions)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy (mAP@50-95)", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = load_data()
    if data:
        plot_pareto_trajectories(data, 'params')
        plot_pareto_trajectories(data, 'gflops')