import os
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import cv2
import textwrap
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO


# --- UI CONSTANTS ---
C = {'b': '\033[94m', 'g': '\033[92m', 'y': '\033[93m', 'r': '\033[91m', 'bold': '\033[1m', 'dim': '\033[2m',
     'res': '\033[0m'}


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


class FoldingComparator:
    def __init__(self, model_paths, image_dir, imgsz=640, model_labels=None, report_title=None, batch_size=16,
                 groups=None):
        self.model_paths = model_paths
        self.image_dir = image_dir
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.groups = groups

        self.model_labels = model_labels if model_labels else [os.path.basename(p) for p in model_paths]
        self.report_title = report_title if report_title else f"Structural Folding Impact Analysis\nDataset: {os.path.basename(image_dir)}"

        if len(self.model_labels) != len(self.model_paths):
            raise ValueError("The number of model_labels must match the number of model_paths.")
        if self.groups and len(self.groups) != len(self.model_paths):
            raise ValueError("The number of groups must match the number of model_paths.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_yaml = self._create_temp_yaml()
        self.results = {}

        self.save_dir = "results_save/plots"
        self.stats_dir = "results_save/save_statistics"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

    def _create_temp_yaml(self):
        abs_dir = os.path.abspath(self.image_dir).replace('\\', '/')
        base_path = os.path.dirname(os.path.dirname(abs_dir))

        yaml_content = {
            'path': base_path,
            'train': abs_dir,
            'val': abs_dir,
            'names': {i: f"class_{i}" for i in range(80)}
        }
        yaml_path = "temp_val_config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        return yaml_path

    def _get_cache_path(self, model_path):
        safe_path_name = model_path.replace('/', '_').replace('\\', '_').replace('.pt', '')
        dataset_name = os.path.basename(self.image_dir)
        return os.path.join(self.stats_dir, f"{safe_path_name}_{dataset_name}_stats.json")

    def _preprocess(self, img_path):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        if r != 1:
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        padded[:img.shape[0], :img.shape[1], :] = img
        img = padded.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img).float() / 255.0

    def run_all_benchmarks(self):
        image_files = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for idx, path in enumerate(self.model_paths):
            m_label = self.model_labels[idx]
            print(f"\n{C['bold']}Testing Model: {m_label} ({os.path.basename(path)}){C['res']}")

            cache_path = self._get_cache_path(path)
            if os.path.exists(cache_path):
                print(f"   {C['g']}Found cached statistics! Skipping inference and loading from JSON...{C['res']}")
                with open(cache_path, 'r') as f:
                    self.results[path] = json.load(f)

                # CRITICAL CACHE FIX: Forcefully update the label in memory so hatching logic applies
                self.results[path]['label'] = m_label
                continue

            print(f"   {C['dim']}No cache found. Running full validation...{C['res']}")
            model = YOLO(path)
            val_metrics = model.val(data=self.data_yaml, split='val', verbose=False, imgsz=self.imgsz)

            raw_model = model.model.to(self.device).eval()
            confs, counts = [], []
            param_count = sum(p.numel() for p in raw_model.parameters())

            for i in tqdm(range(0, len(image_files), self.batch_size), desc=f"Scanning {m_label[:15]}"):
                batch_paths = image_files[i:i + self.batch_size]
                batch_imgs = [self._preprocess(p) for p in batch_paths]
                batch_tensor = torch.stack(batch_imgs, dim=0).to(self.device)

                with torch.no_grad():
                    preds = raw_model(batch_tensor)[0]
                    for b in range(preds.size(0)):
                        conf_map = preds[b, 4:, :].max(dim=0)[0]
                        mask = conf_map > 0.25
                        counts.append(mask.sum().item())
                        if mask.any():
                            confs.extend(conf_map[mask].tolist())

            mp = val_metrics.box.mp
            mr = val_metrics.box.mr
            f1 = 2 * (mp * mr) / (mp + mr + 1e-6)

            self.results[path] = {
                'label': m_label,
                'params': param_count,
                'mAP50': val_metrics.box.map50,
                'mAP50-95': val_metrics.box.map,
                'precision': mp,
                'recall': mr,
                'f1_score': f1,
                'avg_conf': np.mean(confs) if confs else 0,
                'confs_list': confs if confs else [0],
                'avg_anchors': np.mean(counts) if counts else 0
            }

            with open(cache_path, 'w') as f:
                json.dump(self.results[path], f)
            print(f"   {C['g']}Statistics saved to {cache_path}{C['res']}")

            del model
            del raw_model
            torch.cuda.empty_cache()

    def generate_report(self):
        baseline_path = self.model_paths[0]
        baseline_params = self.results[baseline_path]['params']

        print("\n" + "=" * 125)
        grp_col = f"{'Group':<18} | " if self.groups else ""
        header = f"{grp_col}{'Model Name':<25} | {'Params (M)':<10} | {'Reduct.%':<9} | {'mAP50-95':<8} | {'F1-Score':<8} | {'Avg Conf':<8} | {'Anchors'}"
        print(f"{C['bold']}{header}{C['res']}")
        print("-" * 125)

        for idx, path in enumerate(self.model_paths):
            data = self.results[path]
            name = data['label']
            params_millions = data['params'] / 1e6
            reduction = (1 - (data['params'] / baseline_params)) * 100
            red_str = f"-{reduction:.2f}%" if reduction > 0.01 else "-"

            display_name = name[:23] + ".." if len(name) > 25 else name
            grp_str = f"{self.groups[idx][:17]:<18} | " if self.groups else ""

            print(
                f"{grp_str}{display_name:<25} | {params_millions:<10.2f} | {red_str:<9} | {data['mAP50-95']:.4f}   | {data['f1_score']:.4f}   | {data['avg_conf']:.4f}   | {data['avg_anchors']:.1f}")
        print("=" * 125)

        self._plot_comparison(baseline_params)

    def _plot_comparison(self, baseline_params):
        fig = plt.figure(figsize=(24, 24))
        gs = fig.add_gridspec(3, 6, height_ratios=[3.0, 3.5, 1.0], hspace=0.5, wspace=0.8)

        ax1 = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:])
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')

        ax_rank1 = fig.add_subplot(gs[2, 0:2])
        ax_rank2 = fig.add_subplot(gs[2, 2:4])
        ax_rank3 = fig.add_subplot(gs[2, 4:6])
        ax_rank1.axis('off')
        ax_rank2.axis('off')
        ax_rank3.axis('off')

        names = [self.results[p]['label'] for p in self.model_paths]

        # Extract base strategy (e.g., "1k Calib") for unified coloring
        base_strategies = []
        for n in names:
            if "Baseline" in n or "YOLO" in n:
                base_strategies.append("Baseline")
            else:
                base_strategies.append(n.split(': ')[-1] if ': ' in n else n)

        unique_strategies = list(dict.fromkeys(base_strategies))
        palette = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c']
        color_map = {strat: palette[i % len(palette)] for i, strat in enumerate(unique_strategies)}

        bar_colors = [color_map[strat] for strat in base_strategies]

        # Apply dense hatching to Pruned models
        bar_hatches = ['////' if 'Pruned' in n else '' for n in names]
        conf_data = [self.results[p]['confs_list'] for p in self.model_paths]

        # --- DYNAMIC INTRA-GROUP SPACING LOGIC ---
        positions, group_centers, group_names_unique, current_group_positions = [], [], [], []
        current_pos = 1.0
        last_group = self.groups[0] if self.groups else None

        for i in range(len(names)):
            # 1. Large gap for Pairing Rate changes
            if i > 0 and self.groups and self.groups[i] != self.groups[i - 1]:
                group_centers.append(np.mean(current_group_positions))
                group_names_unique.append(last_group)

                current_pos += 2.0
                ax1.axvline(current_pos - 1.0, color='gray', linestyle=':', alpha=0.5)
                ax2.axvline(current_pos - 1.0, color='gray', linestyle=':', alpha=0.5)

                current_group_positions = []
                last_group = self.groups[i]

            # 2. Small gap for Calibration Strategy changes within the same PR
            elif i > 0 and base_strategies[i] != base_strategies[i - 1] and self.groups[i] == self.groups[i - 1]:
                current_pos += 0.5

            positions.append(current_pos)
            current_group_positions.append(current_pos)
            current_pos += 1.0

        if current_group_positions:
            group_centers.append(np.mean(current_group_positions))
            group_names_unique.append(last_group if last_group else "Models")

        max_f1_per_group = {}
        if self.groups:
            for idx, p in enumerate(self.model_paths):
                g = self.groups[idx]
                f1 = self.results[p].get('f1_score', 0)
                if g not in max_f1_per_group or f1 > max_f1_per_group[g]:
                    max_f1_per_group[g] = f1

        # --- Plotting Boxplots ---
        bplot = ax1.boxplot(conf_data, positions=positions, patch_artist=True,
                            medianprops=dict(color='black', linewidth=2.0))

        for patch, color, hatch in zip(bplot['boxes'], bar_colors, bar_hatches):
            rgba_color = mcolors.to_rgba(color, alpha=0.7)
            patch.set_facecolor(rgba_color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            patch.set_hatch(hatch)

        ax1.set_title("Confidence Distribution Shift", fontweight='bold')
        ax1.axhline(y=0.25, color='r', linestyle='--')
        ax1.set_xticks(group_centers)
        ax1.set_xticklabels(group_names_unique, rotation=0, fontweight='bold', fontsize=12)

        # --- Plotting Bar Charts ---
        maps = [self.results[p]['mAP50-95'] for p in self.model_paths]
        rgba_bar_colors = [mcolors.to_rgba(c, alpha=0.8) for c in bar_colors]

        bars = ax2.bar(positions, maps, color=rgba_bar_colors, edgecolor='black', linewidth=1.2, width=0.8)

        for bar, hatch in zip(bars, bar_hatches):
            bar.set_hatch(hatch)

        ax2.set_xticks(group_centers)
        ax2.set_xticklabels(group_names_unique, rotation=0, fontweight='bold', fontsize=12)
        ax2.set_title("mAP@50-95 Accuracy Comparison", fontweight='bold')
        ax2.set_ylim(0, max(maps) * 1.2)

        # --- Dual Legend (Colors + Hatches) ---
        legend_patches = [mpatches.Patch(color=color_map[strat], label=strat) for strat in unique_strategies]
        legend_patches.append(mpatches.Patch(color='white', label=''))
        legend_patches.append(mpatches.Patch(facecolor='white', edgecolor='black', label='Folded Model'))
        legend_patches.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='Pruned Model'))
        ax2.legend(handles=legend_patches, loc='upper right', title="Legend", fontsize=10)

        # --- Main Table ---
        columns = ("Group", "Model Name", "Params (M)", "Reduct.%", "mAP50-95", "F1-Score", "Avg Conf",
                   "Anchors") if self.groups else ("Model Name", "Params (M)", "Reduct.%", "mAP50-95", "F1-Score",
                                                   "Avg Conf", "Anchors")

        cell_text = []
        last_group_table = None

        for idx, path in enumerate(self.model_paths):
            data = self.results[path]
            name = data['label']
            params_m = data['params'] / 1e6
            reduction = (1 - data['params'] / baseline_params) * 100
            red_str = f"-{reduction:.2f}%" if reduction > 0.01 else "-"

            row = []
            if self.groups:
                current_group = self.groups[idx]
                if current_group == last_group_table:
                    row.append("")
                else:
                    row.append(textwrap.fill(current_group, width=15))
                last_group_table = current_group

            row.extend([
                textwrap.fill(name, width=28),
                f"{params_m:.2f}",
                red_str,
                f"{data['mAP50-95']:.4f}",
                f"{data.get('f1_score', 0):.4f}",
                f"{data['avg_conf']:.4f}",
                f"{data['avg_anchors']:.1f}"
            ])
            cell_text.append(row)

        table = ax_table.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.35)

        model_name_col_idx = 1 if self.groups else 0
        f1_col_idx = 5 if self.groups else 4

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e0e0e0')
            elif row > 0:
                path_idx = row - 1

                if col == model_name_col_idx:
                    color_hex = bar_colors[path_idx]
                    subtle_color = mcolors.to_rgba(color_hex, alpha=0.25)
                    cell.set_facecolor(subtle_color)

                elif col == f1_col_idx and self.groups:
                    p = self.model_paths[path_idx]
                    g = self.groups[path_idx]
                    f1 = self.results[p].get('f1_score', 0)

                    if abs(f1 - max_f1_per_group.get(g, -1)) < 1e-6:
                        cell.get_text().set_color('#27ae60')
                        cell.get_text().set_weight('bold')

        # --- Ranking Tables ---
        ax_rank1.set_title("Top 5 Overall (F1-Score)", fontweight='bold', pad=10)
        sorted_f1 = sorted(self.model_paths, key=lambda p: self.results[p].get('f1_score', 0), reverse=True)[:5]
        cols1 = ["Rank", "Pairing Rate", "Model", "F1-Score"]
        cells1 = []
        for i, p in enumerate(sorted_f1):
            idx = self.model_paths.index(p)
            grp = self.groups[idx].replace("Pairing Rate: ", "PR ") if self.groups else "-"
            cells1.append([f"#{i + 1}", grp, textwrap.fill(self.results[p]['label'], 15),
                           f"{self.results[p].get('f1_score', 0):.4f}"])

        t1 = ax_rank1.table(cellText=cells1, colLabels=cols1, loc='center', cellLoc='center')
        t1.auto_set_font_size(False)
        t1.set_fontsize(11)
        t1.scale(1, 1.8)
        for (r, c), cell in t1.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        ax_rank2.set_title("Best per Group (F1-Score)", fontweight='bold', pad=10)
        cols2 = ["Group", "Top Model", "F1-Score"]
        cells2 = []
        if self.groups:
            group_rankings = {}
            for idx, p in enumerate(self.model_paths):
                g = self.groups[idx].replace("Pairing Rate: ", "PR ")
                if g not in group_rankings: group_rankings[g] = []
                group_rankings[g].append(p)

            for g, paths in group_rankings.items():
                if "Baseline" in g: continue
                best_p = max(paths, key=lambda p: self.results[p].get('f1_score', 0))
                cells2.append([g, textwrap.fill(self.results[best_p]['label'], 15),
                               f"{self.results[best_p].get('f1_score', 0):.4f}"])

        t2 = ax_rank2.table(cellText=cells2, colLabels=cols2, loc='center', cellLoc='center')
        t2.auto_set_font_size(False)
        t2.set_fontsize(11)
        t2.scale(1, 1.8)
        for (r, c), cell in t2.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        ax_rank3.set_title("Best per Group (mAP@50-95)", fontweight='bold', pad=10)
        cols3 = ["Group", "Top Model", "mAP50-95"]
        cells3 = []
        if self.groups:
            group_rankings = {}
            for idx, p in enumerate(self.model_paths):
                g = self.groups[idx].replace("Pairing Rate: ", "PR ")
                if g not in group_rankings: group_rankings[g] = []
                group_rankings[g].append(p)

            for g, paths in group_rankings.items():
                if "Baseline" in g: continue
                best_p = max(paths, key=lambda p: self.results[p]['mAP50-95'])
                cells3.append(
                    [g, textwrap.fill(self.results[best_p]['label'], 15), f"{self.results[best_p]['mAP50-95']:.4f}"])

        t3 = ax_rank3.table(cellText=cells3, colLabels=cols3, loc='center', cellLoc='center')
        t3.auto_set_font_size(False)
        t3.set_fontsize(11)
        t3.scale(1, 1.8)
        for (r, c), cell in t3.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        plt.suptitle(self.report_title, fontsize=24, fontweight='bold', y=0.96)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05)

        safe_title = "".join(c if c.isalnum() else "_" for c in self.report_title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = os.path.join(self.save_dir, f"comparison_{safe_title}_{timestamp}.png")

        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"\n{C['g']}Success: Plot saved to {save_name}{C['res']}")
        plt.show()

    def cleanup(self):
        if os.path.exists(self.data_yaml):
            os.remove(self.data_yaml)


# --- EXECUTION ---
if __name__ == "__main__":

    MODELS_TO_COMPARE = [
        # --- BASELINE ---
        'weights/yolov8m.pt',

        # --- PAIRING RATE 0.1 ---
        'weights/without_repair/0.1/yolo_conv4_to_conv8_folded_without_repair.pt',
        'weights/prune/0.1/yolo_conv4_to_conv8_pruned_without_repair.pt',
        'weights/forward_pass_repair/0.1/yolo_conv4_to_conv8_folded_forward_pass_repair_calib1000.pt',
        'weights/prune/0.1/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib1000.pt',
        'weights/forward_pass_repair/0.1/yolo_conv4_to_conv8_folded_forward_pass_repair_calib5000.pt',
        'weights/prune/0.1/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib5000.pt',
        'weights/forward_pass_repair/0.1/yolo_conv4_to_conv8_folded_forward_pass_repair_calib20000.pt',
        'weights/prune/0.1/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib20000.pt',
        'weights/forward_pass_repair/0.1/yolo_conv4_to_conv8_folded_forward_pass_repair_calib60000.pt',
        'weights/prune/0.1/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib60000.pt',

        # --- PAIRING RATE 0.2 ---
        'weights/without_repair/0.2/yolo_conv4_to_conv8_folded_without_repair.pt',
        'weights/prune/0.2/yolo_conv4_to_conv8_pruned_without_repair.pt',
        'weights/forward_pass_repair/0.2/yolo_conv4_to_conv8_folded_forward_pass_repair_calib1000.pt',
        'weights/prune/0.2/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib1000.pt',
        'weights/forward_pass_repair/0.2/yolo_conv4_to_conv8_folded_forward_pass_repair_calib5000.pt',
        'weights/prune/0.2/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib5000.pt',
        'weights/forward_pass_repair/0.2/yolo_conv4_to_conv8_folded_forward_pass_repair_calib20000.pt',
        'weights/prune/0.2/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib20000.pt',
        'weights/forward_pass_repair/0.2/yolo_conv4_to_conv8_folded_forward_pass_repair_calib60000.pt',
        'weights/prune/0.2/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib60000.pt',

        # --- PAIRING RATE 0.3 ---
        'weights/without_repair/0.3/yolo_conv4_to_conv8_folded_without_repair.pt',
        'weights/prune/0.3/yolo_conv4_to_conv8_pruned_without_repair.pt',
        'weights/forward_pass_repair/0.3/yolo_conv4_to_conv8_folded_forward_pass_repair_calib1000.pt',
        'weights/prune/0.3/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib1000.pt',
        'weights/forward_pass_repair/0.3/yolo_conv4_to_conv8_folded_forward_pass_repair_calib5000.pt',
        'weights/prune/0.3/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib5000.pt',
        'weights/forward_pass_repair/0.3/yolo_conv4_to_conv8_folded_forward_pass_repair_calib20000.pt',
        'weights/prune/0.3/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib20000.pt',
        'weights/forward_pass_repair/0.3/yolo_conv4_to_conv8_folded_forward_pass_repair_calib60000.pt',
        'weights/prune/0.3/yolo_conv4_to_conv8_pruned_forward_pass_repair_calib60000.pt'
    ]

    CUSTOM_LABELS = [
        "YOLOv8m",

        "Folded: No Repair", "Pruned: No Repair",
        "Folded: 1k Calib", "Pruned: 1k Calib",
        "Folded: 5k Calib", "Pruned: 5k Calib",
        "Folded: 20k Calib", "Pruned: 20k Calib",
        "Folded: 60k Calib", "Pruned: 60k Calib",

        "Folded: No Repair", "Pruned: No Repair",
        "Folded: 1k Calib", "Pruned: 1k Calib",
        "Folded: 5k Calib", "Pruned: 5k Calib",
        "Folded: 20k Calib", "Pruned: 20k Calib",
        "Folded: 60k Calib", "Pruned: 60k Calib",

        "Folded: No Repair", "Pruned: No Repair",
        "Folded: 1k Calib", "Pruned: 1k Calib",
        "Folded: 5k Calib", "Pruned: 5k Calib",
        "Folded: 20k Calib", "Pruned: 20k Calib",
        "Folded: 60k Calib", "Pruned: 60k Calib"
    ]

    GROUPS = [
        "Baseline",

        "PR: 0.1", "PR: 0.1", "PR: 0.1", "PR: 0.1", "PR: 0.1", "PR: 0.1", "PR: 0.1", "PR: 0.1", "PR: 0.1", "PR: 0.1",
        "PR: 0.2", "PR: 0.2", "PR: 0.2", "PR: 0.2", "PR: 0.2", "PR: 0.2", "PR: 0.2", "PR: 0.2", "PR: 0.2", "PR: 0.2",
        "PR: 0.3", "PR: 0.3", "PR: 0.3", "PR: 0.3", "PR: 0.3", "PR: 0.3", "PR: 0.3", "PR: 0.3", "PR: 0.3", "PR: 0.3"
    ]

    REPORT_TITLE = "conv4_to_conv8 compare: Structural Folding vs. Pruning"
    IMG_PATH = r"coco/images/val2017"

    comp = FoldingComparator(
        model_paths=MODELS_TO_COMPARE,
        image_dir=IMG_PATH,
        model_labels=CUSTOM_LABELS,
        report_title=REPORT_TITLE,
        batch_size=16,
        groups=GROUPS
    )

    try:
        comp.run_all_benchmarks()
        comp.generate_report()
    finally:
        comp.cleanup()