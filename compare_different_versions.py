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

# ── optional FLOPs backend (thop preferred, fvcore as fallback) ──────────────
try:
    from thop import profile as thop_profile
    _FLOPS_BACKEND = "thop"
except ImportError:
    try:
        from fvcore.nn import FlopCountAnalysis
        _FLOPS_BACKEND = "fvcore"
    except ImportError:
        _FLOPS_BACKEND = None

# --- UI CONSTANTS ---
C = {'b': '\033[94m', 'g': '\033[92m', 'y': '\033[93m', 'r': '\033[91m', 'bold': '\033[1m', 'dim': '\033[2m',
     'res': '\033[0m'}

# ─────────────────────────────────────────────────────────────────────────────
#  FLOPs helper
# ─────────────────────────────────────────────────────────────────────────────
def _compute_gflops(model: nn.Module, imgsz: int, device) -> float:
    if _FLOPS_BACKEND is None:
        return 0.0

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    try:
        if _FLOPS_BACKEND == "thop":
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            return macs * 2 / 1e9
        else:
            flops = FlopCountAnalysis(model, dummy)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            return flops.total() / 1e9
    except Exception as e:
        print(f"   {C['y']}Warning: FLOPs computation failed ({e}). Storing 0.{C['res']}")
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
#  C2f_v2 compatibility shim
# ─────────────────────────────────────────────────────────────────────────────
class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules import Conv, Bottleneck
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

# ─────────────────────────────────────────────────────────────────────────────
#  Main comparator class
# ─────────────────────────────────────────────────────────────────────────────
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

        # ── output dirs ──────────────────────────────────────────────────────
        self.save_dir = "results_save/plots"
        self.stats_dir = "results_save/save_statistics_fix"
        self.report_dir = "results_save/overall_reports"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

        if _FLOPS_BACKEND:
            print(f"   {C['g']}FLOPs backend: {_FLOPS_BACKEND}{C['res']}")
        else:
            print(f"   {C['y']}No FLOPs backend found. Install 'thop' to enable GFLOPs.{C['res']}")

    # ── helpers ───────────────────────────────────────────────────────────────
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
        # 1. Clean the path and remove the leading "weights/" folder
        clean_path = model_path.replace('\\', '/')
        if clean_path.startswith("weights/"):
            clean_path = clean_path[8:]

        # 2. Extract the nested directory structure and the filename
        dir_structure = os.path.dirname(clean_path)
        base_name = os.path.basename(clean_path).replace('.pt', '')
        dataset_name = os.path.basename(self.image_dir)

        # 3. Create the exact matching nested directory inside stats_dir
        target_dir = os.path.join(self.stats_dir, dir_structure)
        os.makedirs(target_dir, exist_ok=True)

        # 4. Construct the final JSON path
        return os.path.join(target_dir, f"{base_name}_{dataset_name}_stats.json")

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

    def _save_overall_txt_report(self, baseline_params):
        """Generates a single, comprehensive text report grouped by model."""
        baseline_filename = os.path.basename(self.model_paths[0])
        baseline_variant = os.path.splitext(baseline_filename)[0]

        target_dir = os.path.join(self.report_dir, baseline_variant)
        os.makedirs(target_dir, exist_ok=True)

        safe_title = "".join(c if c.isalnum() else "_" for c in self.report_title.split('\n')[0])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(target_dir, f"Overall_Report_{safe_title}_{timestamp}.txt")

        sep = "=" * 135
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{sep}\n")
            f.write(f" MASTER BENCHMARK REPORT\n")
            f.write(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f" {self.report_title.replace(chr(10), ' | ')}\n")
            f.write(f"{sep}\n\n")

            header = (f"{'Group':<20} | {'Model Name':<25} | {'Params (M)':<10} | {'Saved (M)':<9} | "
                      f"{'Reduct.%':<9} | {'GFLOPs':<8} | {'mAP50':<7} | {'mAP50-95':<8} | "
                      f"{'F1-Score':<8} | {'Anchors'}")
            f.write(f"{header}\n")
            f.write("-" * 135 + "\n")

            for idx, path in enumerate(self.model_paths):
                data = self.results[path]
                group_name = self.groups[idx] if self.groups else "-"
                params_m = data['params'] / 1e6
                saved_m = (baseline_params - data['params']) / 1e6
                gflops = data.get('gflops', 0.0)
                reduction = (1 - (data['params'] / baseline_params)) * 100

                red_str = f"-{reduction:.2f}%" if reduction > 0.01 else "-"
                sav_str = f"{saved_m:.2f}" if saved_m > 0 else "-"

                f.write(
                    f"{group_name[:19]:<20} | {data['label'][:24]:<25} | {params_m:<10.2f} | {sav_str:<9} | "
                    f"{red_str:<9} | {gflops:<8.3f} | {data.get('mAP50', 0.0):.4f}  | {data['mAP50-95']:.4f}   | "
                    f"{data['f1_score']:.4f}   | {data['avg_anchors']:.1f}\n")

            f.write(f"\n{sep}\n")
        print(f"   {C['g']}Master readable report saved to {txt_path}{C['res']}")

    # ── benchmark runner ──────────────────────────────────────────────────────
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
                self.results[path]['label'] = m_label
                continue

            print(f"   {C['dim']}No cache found. Running full validation...{C['res']}")
            model = YOLO(path)
            val_metrics = model.val(data=self.data_yaml, split='val', verbose=False, imgsz=self.imgsz)

            raw_model = model.model.to(self.device).eval()
            confs, counts = [], []
            param_count = sum(p.numel() for p in raw_model.parameters())

            print(f"   {C['dim']}Computing GFLOPs...{C['res']}")
            gflops = _compute_gflops(raw_model, self.imgsz, self.device)
            print(f"   GFLOPs: {gflops:.3f}")

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
                'gflops': gflops,
                'mAP50': val_metrics.box.map50,
                'mAP50-95': val_metrics.box.map,
                'precision': mp,
                'recall': mr,
                'f1_score': f1,
                'avg_conf': float(np.mean(confs)) if confs else 0.0,
                'confs_list': confs if confs else [0.0],
                'avg_anchors': float(np.mean(counts)) if counts else 0.0
            }

            with open(cache_path, 'w') as f:
                json.dump(self.results[path], f, indent=2)
            print(f"   {C['g']}Statistics (JSON) saved to {cache_path}{C['res']}")

            del model
            del raw_model
            torch.cuda.empty_cache()

    # ── console report ────────────────────────────────────────────────────────
    def generate_report(self):
        baseline_path = self.model_paths[0]
        baseline_params = self.results[baseline_path]['params']

        # Save master text report
        self._save_overall_txt_report(baseline_params)

        print("\n" + "=" * 155)
        grp_col = f"{'Group':<18} | " if self.groups else ""
        header = (f"{grp_col}{'Model Name':<25} | {'Params (M)':<10} | {'Saved (M)':<9} | {'GFLOPs':<8} | "
                  f"{'Reduct.%':<9} | {'mAP50':<7} | {'mAP50-95':<8} | "
                  f"{'F1-Score':<8} | {'Avg Conf':<8} | {'Anchors'}")
        print(f"{C['bold']}{header}{C['res']}")
        print("-" * 155)

        for idx, path in enumerate(self.model_paths):
            data = self.results[path]
            name = data['label']
            params_millions = data['params'] / 1e6
            saved_m = (baseline_params - data['params']) / 1e6
            gflops = data.get('gflops', 0.0)
            reduction = (1 - (data['params'] / baseline_params)) * 100

            red_str = f"-{reduction:.2f}%" if reduction > 0.01 else "-"
            sav_str = f"{saved_m:.2f}" if saved_m > 0 else "-"
            display_name = name[:23] + ".." if len(name) > 25 else name
            grp_str = f"{self.groups[idx][:17]:<18} | " if self.groups else ""

            print(
                f"{grp_str}{display_name:<25} | {params_millions:<10.2f} | {sav_str:<9} | {gflops:<8.3f} | "
                f"{red_str:<9} | {data.get('mAP50', 0.0):.4f}  | {data['mAP50-95']:.4f}   | "
                f"{data['f1_score']:.4f}   | {data['avg_conf']:.4f}   | {data['avg_anchors']:.1f}")
        print("=" * 155)

        self._generate_all_plots(baseline_params)

    # ── shared colour / position helpers ─────────────────────────────────────
    def _build_color_and_positions(self, names):
        base_strategies = []
        for n in names:
            if "Baseline" in n or "YOLO" in n:
                base_strategies.append("Baseline")
            else:
                base_strategies.append(n.split(': ')[-1] if ': ' in n else n)

        unique_strategies = list(dict.fromkeys(base_strategies))
        palette = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c', '#f39c12']
        color_map = {strat: palette[i % len(palette)] for i, strat in enumerate(unique_strategies)}
        bar_colors = [color_map[strat] for strat in base_strategies]
        bar_hatches = ['////' if 'Static' in n else '' for n in names]

        positions, group_centers, group_names_unique, current_group_positions = [], [], [], []
        current_pos = 1.0
        last_group = self.groups[0] if self.groups else None

        for i in range(len(names)):
            if i > 0 and self.groups and self.groups[i] != self.groups[i - 1]:
                group_centers.append(np.mean(current_group_positions))
                group_names_unique.append(last_group)
                current_pos += 2.0
                current_group_positions = []
                last_group = self.groups[i]
            elif i > 0 and base_strategies[i] != base_strategies[i - 1] and (
                    not self.groups or self.groups[i] == self.groups[i - 1]):
                current_pos += 0.5
            positions.append(current_pos)
            current_group_positions.append(current_pos)
            current_pos += 1.0

        if current_group_positions:
            group_centers.append(np.mean(current_group_positions))
            group_names_unique.append(last_group if last_group else "Models")

        return base_strategies, color_map, bar_colors, bar_hatches, positions, group_centers, group_names_unique, unique_strategies

    # ── individual plot helpers ───────────────────────────────────────────────
    def _savefig(self, fig, name_suffix, subfolder):
        """Save figure to a specific subfolder inside plots/<baseline_model>/."""
        # Basis-Modell Namen aus dem ersten Pfad extrahieren (z.B. 'yolov8m')
        baseline_filename = os.path.basename(self.model_paths[0])
        baseline_variant = os.path.splitext(baseline_filename)[0]

        # Neuen verschachtelten Pfad erstellen
        target_dir = os.path.join(self.save_dir, baseline_variant, subfolder)
        os.makedirs(target_dir, exist_ok=True)

        safe_title = "".join(c if c.isalnum() else "_" for c in self.report_title.split('\n')[0])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = os.path.join(target_dir, f"{name_suffix}_{safe_title}_{timestamp}.png")

        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        print(f"   {C['dim']}Saved plot: {baseline_variant}/{subfolder}/{os.path.basename(save_name)}{C['res']}")
        plt.close(fig)

    def _draw_sparsity_line(self, ax, metric_key, metric_label, names, baseline_params, color_map):
        """Helper to draw the sparsity line plot on a given axis."""
        strategy_paths = {}
        for idx, path in enumerate(self.model_paths):
            name = names[idx]
            strat = "Baseline" if ("Baseline" in name or "YOLO" in name) else (
                name.split(': ')[-1] if ': ' in name else name)
            strategy_paths.setdefault(strat, []).append(path)

        palette = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c', '#f39c12']

        for s_idx, (strat, paths) in enumerate(strategy_paths.items()):
            color = color_map.get(strat, palette[s_idx % len(palette)])
            xs, ys = [], []
            for p in paths:
                data = self.results[p]
                sparsity = (1 - data['params'] / baseline_params) * 100
                xs.append(sparsity)
                ys.append(data.get(metric_key, 0.0))

            pairs = sorted(zip(xs, ys))
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
            ax.plot(xs, ys, marker='o', linewidth=2, markersize=6, color=color, label=strat)

        ax.set_xlabel("Sparsity (% Parameter Reduction)", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"Sparsity vs. {metric_label}", fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.5)

    def _plot_confidence_boxplot(self, names, bar_colors, bar_hatches, positions, group_centers, group_names_unique,
                                 color_map, unique_strategies):
        fig, ax = plt.subplots(figsize=(14, 6))
        conf_data = [self.results[p]['confs_list'] for p in self.model_paths]

        bplot = ax.boxplot(conf_data, positions=positions, patch_artist=True,
                           medianprops=dict(color='black', linewidth=2.0))
        for patch, color, hatch in zip(bplot['boxes'], bar_colors, bar_hatches):
            patch.set_facecolor(mcolors.to_rgba(color, alpha=0.7))
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            patch.set_hatch(hatch)

        for i in range(1, len(self.model_paths)):
            if self.groups and self.groups[i] != self.groups[i - 1]:
                ax.axvline((positions[i - 1] + positions[i]) / 2, color='gray', linestyle=':', alpha=0.5)

        ax.axhline(y=0.25, color='r', linestyle='--', label='Conf threshold (0.25)')
        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_names_unique, rotation=15, fontweight='bold', fontsize=11)
        ax.set_ylabel("Detection Confidence")
        ax.set_title("Confidence Distribution Shift", fontweight='bold', fontsize=14)

        legend_patches = [mpatches.Patch(color=color_map[s], label=s) for s in unique_strategies]
        legend_patches += [mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='Static Model')]
        ax.legend(handles=legend_patches, fontsize=9, loc='upper right')
        plt.suptitle(self.report_title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._savefig(fig, "plot_confidence_boxplot", "boxplots")

    def _plot_map_bar(self, metric_key, metric_label, names, bar_colors, bar_hatches, positions, group_centers,
                      group_names_unique, color_map, unique_strategies):
        fig, ax = plt.subplots(figsize=(14, 6))
        values = [self.results[p].get(metric_key, 0.0) for p in self.model_paths]
        rgba_colors = [mcolors.to_rgba(c, alpha=0.85) for c in bar_colors]

        bars = ax.bar(positions, values, color=rgba_colors, edgecolor='black', linewidth=1.2, width=0.8)
        for bar, hatch in zip(bars, bar_hatches): bar.set_hatch(hatch)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.4f}", ha='center', va='bottom',
                    fontsize=8)

        for i in range(1, len(self.model_paths)):
            if self.groups and self.groups[i] != self.groups[i - 1]:
                ax.axvline((positions[i - 1] + positions[i]) / 2, color='gray', linestyle=':', alpha=0.5)

        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_names_unique, rotation=15, fontweight='bold', fontsize=11)
        ax.set_title(f"{metric_label} Accuracy Comparison", fontweight='bold', fontsize=14)
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, max(values) * 1.18 if values else 1.0)

        legend_patches = [mpatches.Patch(color=color_map[s], label=s) for s in unique_strategies]
        legend_patches += [mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='Static Model')]
        ax.legend(handles=legend_patches, fontsize=9, loc='upper right')
        plt.suptitle(self.report_title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._savefig(fig, f"plot_{metric_key.replace('-', '_').replace('@', '')}_bar", "bar_charts")

    def _plot_sparsity_vs_accuracy(self, metric_key, metric_label, names, baseline_params, color_map, bar_colors,
                                   unique_strategies):
        fig, ax = plt.subplots(figsize=(9, 6))
        self._draw_sparsity_line(ax, metric_key, metric_label, names, baseline_params, color_map)
        plt.suptitle(self.report_title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._savefig(fig, f"plot_sparsity_vs_{metric_key.replace('-', '_').replace('@', '')}", "line_plots")

    def _plot_ranking_tables(self, names, baseline_params):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        for ax in (ax1, ax2, ax3): ax.axis('off')

        ax1.set_title("Top 5 Overall (F1-Score)", fontweight='bold', pad=10)
        sorted_f1 = sorted(self.model_paths, key=lambda p: self.results[p].get('f1_score', 0), reverse=True)[:5]
        cells1 = [[f"#{i + 1}",
                   self.groups[self.model_paths.index(p)].replace("Pairing Rate: ", "PR ") if self.groups else "-",
                   self.results[p]['label'], f"{self.results[p].get('f1_score', 0):.4f}"] for i, p in
                  enumerate(sorted_f1)]
        t1 = ax1.table(cellText=cells1, colLabels=["Rank", "Group", "Model", "F1-Score"], loc='center',
                       cellLoc='center')
        t1.auto_set_font_size(False)
        t1.set_fontsize(10)
        t1.scale(1, 1.8)
        t1.auto_set_column_width(col=list(range(4)))
        for (r, c), cell in t1.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        ax2.set_title("Best per Group (F1-Score)", fontweight='bold', pad=10)
        cells2 = []
        if self.groups:
            gr = {}
            for i, p in enumerate(self.model_paths): gr.setdefault(self.groups[i].replace("Pairing Rate: ", "PR "),
                                                                   []).append(p)
            for g, ps in gr.items():
                if "Baseline" in g: continue
                bp = max(ps, key=lambda p: self.results[p].get('f1_score', 0))
                cells2.append(
                    [g, self.results[bp]['label'], f"{self.results[bp].get('f1_score', 0):.4f}"])
        t2 = ax2.table(cellText=cells2 or [["—", "—", "—"]], colLabels=["Group", "Top Model", "F1-Score"], loc='center',
                       cellLoc='center')
        t2.auto_set_font_size(False)
        t2.set_fontsize(10)
        t2.scale(1, 1.8)
        t2.auto_set_column_width(col=list(range(3)))
        for (r, c), cell in t2.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        ax3.set_title("Best per Group (mAP@50-95)", fontweight='bold', pad=10)
        cells3 = []
        if self.groups:
            gr = {}
            for i, p in enumerate(self.model_paths): gr.setdefault(self.groups[i].replace("Pairing Rate: ", "PR "),
                                                                   []).append(p)
            for g, ps in gr.items():
                if "Baseline" in g: continue
                bp = max(ps, key=lambda p: self.results[p].get('mAP50-95', 0))
                cells3.append(
                    [g, self.results[bp]['label'], f"{self.results[bp].get('mAP50-95', 0):.4f}"])
        t3 = ax3.table(cellText=cells3 or [["—", "—", "—"]], colLabels=["Group", "Top Model", "mAP50-95"], loc='center',
                       cellLoc='center')
        t3.auto_set_font_size(False)
        t3.set_fontsize(10)
        t3.scale(1, 1.8)
        t3.auto_set_column_width(col=list(range(3)))
        for (r, c), cell in t3.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        plt.suptitle(self.report_title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._savefig(fig, "plot_ranking_tables", "tables")

    def _plot_combined_dashboard(self, baseline_params, names, bar_colors, bar_hatches, positions, group_centers,
                                 group_names_unique, color_map, unique_strategies):
        """Updated to include line plots and the Saved (M) column."""
        fig = plt.figure(figsize=(24, 30))
        gs = fig.add_gridspec(4, 6, height_ratios=[3.0, 3.0, 3.5, 1.0], hspace=0.4, wspace=0.8)

        ax1 = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:])
        ax_line1 = fig.add_subplot(gs[1, :3])
        ax_line2 = fig.add_subplot(gs[1, 3:])
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        ax_rank1 = fig.add_subplot(gs[3, 0:2])
        ax_rank1.axis('off')
        ax_rank2 = fig.add_subplot(gs[3, 2:4])
        ax_rank2.axis('off')
        ax_rank3 = fig.add_subplot(gs[3, 4:6])
        ax_rank3.axis('off')

        conf_data = [self.results[p]['confs_list'] for p in self.model_paths]
        maps_95 = [self.results[p]['mAP50-95'] for p in self.model_paths]

        def _add_vsep(ax_):
            for i in range(1, len(self.model_paths)):
                if self.groups and self.groups[i] != self.groups[i - 1]:
                    ax_.axvline((positions[i - 1] + positions[i]) / 2, color='gray', linestyle=':', alpha=0.5)

        # 1. Boxplot
        bplot = ax1.boxplot(conf_data, positions=positions, patch_artist=True,
                            medianprops=dict(color='black', linewidth=2.0))
        for patch, color, hatch in zip(bplot['boxes'], bar_colors, bar_hatches):
            patch.set_facecolor(mcolors.to_rgba(color, alpha=0.7))
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            patch.set_hatch(hatch)
        _add_vsep(ax1)
        ax1.set_title("Confidence Distribution Shift", fontweight='bold')
        ax1.axhline(y=0.25, color='r', linestyle='--')
        ax1.set_xticks(group_centers)
        ax1.set_xticklabels(group_names_unique, rotation=0, fontweight='bold', fontsize=12)

        # 2. Bar Chart
        bars = ax2.bar(positions, maps_95, color=[mcolors.to_rgba(c, alpha=0.8) for c in bar_colors], edgecolor='black',
                       linewidth=1.2, width=0.8)
        for bar, hatch in zip(bars, bar_hatches): bar.set_hatch(hatch)
        _add_vsep(ax2)
        ax2.set_xticks(group_centers)
        ax2.set_xticklabels(group_names_unique, rotation=0, fontweight='bold', fontsize=12)
        ax2.set_title("mAP@50-95 Accuracy Comparison", fontweight='bold')
        ax2.set_ylim(0, max(maps_95) * 1.2 if maps_95 else 1.0)

        lp = [mpatches.Patch(color=color_map[s], label=s) for s in unique_strategies]
        lp += [mpatches.Patch(color='white', label=''),
               mpatches.Patch(facecolor='white', edgecolor='black', label='Dynamic Model'),
               mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='Static Model')]
        ax2.legend(handles=lp, loc='upper right', title="Legend", fontsize=10)

        # 3 & 4. Line Plots
        self._draw_sparsity_line(ax_line1, 'mAP50', 'mAP@50', names, baseline_params, color_map)
        self._draw_sparsity_line(ax_line2, 'mAP50-95', 'mAP@50-95', names, baseline_params, color_map)

        # 5. Main Table
        columns = ["Group", "Model Name", "Params (M)", "Saved (M)", "GFLOPs", "Reduct.%", "mAP50", "mAP50-95",
                   "F1-Score", "Avg Conf", "Anchors"] if self.groups else ["Model Name", "Params (M)", "Saved (M)",
                                                                           "GFLOPs", "Reduct.%", "mAP50", "mAP50-95",
                                                                           "F1-Score", "Avg Conf", "Anchors"]

        cell_text = []
        last_g = None
        m_f1, m_map = {}, {}
        if self.groups:
            for idx, p in enumerate(self.model_paths):
                g = self.groups[idx]
                f = self.results[p].get('f1_score', 0)
                m = self.results[p].get('mAP50-95', 0)
                if g not in m_f1 or f > m_f1[g]: m_f1[g] = f
                if g not in m_map or m > m_map[g]: m_map[g] = m

        for idx, path in enumerate(self.model_paths):
            data = self.results[path]
            p_m = data['params'] / 1e6
            s_m = (baseline_params - data['params']) / 1e6
            gf = data.get('gflops', 0.0)
            red = (1 - data['params'] / baseline_params) * 100

            row = []
            if self.groups:
                cg = self.groups[idx]
                row.append("" if cg == last_g else cg.replace("Pairing Rate: ", "PR "))
                last_g = cg

            row.extend([data['label'], f"{p_m:.2f}", f"{s_m:.2f}" if s_m > 0 else "-", f"{gf:.3f}",
                        f"-{red:.2f}%" if red > 0.01 else "-", f"{data.get('mAP50', 0.0):.4f}",
                        f"{data['mAP50-95']:.4f}", f"{data.get('f1_score', 0):.4f}", f"{data['avg_conf']:.4f}",
                        f"{data['avg_anchors']:.1f}"])
            cell_text.append(row)

        table = ax_table.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.3)
        table.auto_set_column_width(col=list(range(len(columns))))

        name_col = 1 if self.groups else 0
        m95_col = 7 if self.groups else 6
        f1_col = 8 if self.groups else 7

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e0e0e0')
            elif row > 0:
                p_i = row - 1
                if col == name_col:
                    cell.set_facecolor(mcolors.to_rgba(bar_colors[p_i], alpha=0.25))
                elif self.groups and col == f1_col and abs(
                        self.results[self.model_paths[p_i]].get('f1_score', 0) - m_f1.get(self.groups[p_i], -1)) < 1e-6:
                    cell.get_text().set_color('#27ae60')
                    cell.get_text().set_weight('bold')
                elif self.groups and col == m95_col and abs(
                        self.results[self.model_paths[p_i]].get('mAP50-95', 0) - m_map.get(self.groups[p_i],
                                                                                           -1)) < 1e-6:
                    cell.get_text().set_color('#27ae60')
                    cell.get_text().set_weight('bold')

        # 6. Rank Tables
        for a in (ax_rank1, ax_rank2, ax_rank3): a.axis('off')

        ax_rank1.set_title("Top 5 Overall (F1)", fontweight='bold', pad=10)
        s_f1 = sorted(self.model_paths, key=lambda p: self.results[p].get('f1_score', 0), reverse=True)[:5]
        c1 = [[f"#{i + 1}",
               self.groups[self.model_paths.index(p)].replace("Pairing Rate: ", "PR ") if self.groups else "-",
               self.results[p]['label'], f"{self.results[p].get('f1_score', 0):.4f}"] for i, p in
              enumerate(s_f1)]
        t1 = ax_rank1.table(cellText=c1, colLabels=["Rank", "Group", "Model", "F1"], loc='center', cellLoc='center')
        t1.auto_set_font_size(False)
        t1.set_fontsize(10)
        t1.scale(1, 1.8)
        t1.auto_set_column_width(col=list(range(4)))
        for (r, c), cell in t1.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        ax_rank2.set_title("Best per Group (F1)", fontweight='bold', pad=10)
        c2 = []
        if self.groups:
            for g, ps in m_f1.items():
                if "Baseline" not in g: c2.append([g.replace("Pairing Rate: ", "PR "), self.results[
                                                                                                         max([p for i, p
                                                                                                              in
                                                                                                              enumerate(
                                                                                                                  self.model_paths)
                                                                                                              if
                                                                                                              self.groups[
                                                                                                                  i] == g],
                                                                                                             key=lambda
                                                                                                                 p:
                                                                                                             self.results[
                                                                                                                 p].get(
                                                                                                                 'f1_score',
                                                                                                                 0))][
                                                                                                         'label'],
                                                   f"{ps:.4f}"])
        t2 = ax_rank2.table(cellText=c2 or [["—", "—", "—"]], colLabels=["Group", "Top Model", "F1"], loc='center',
                            cellLoc='center')
        t2.auto_set_font_size(False)
        t2.set_fontsize(10)
        t2.scale(1, 1.8)
        t2.auto_set_column_width(col=list(range(3)))
        for (r, c), cell in t2.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        ax_rank3.set_title("Best per Group (mAP@50-95)", fontweight='bold', pad=10)
        c3 = []
        if self.groups:
            for g, ps in m_map.items():
                if "Baseline" not in g: c3.append([g.replace("Pairing Rate: ", "PR "), self.results[
                                                                                                         max([p for i, p
                                                                                                              in
                                                                                                              enumerate(
                                                                                                                  self.model_paths)
                                                                                                              if
                                                                                                              self.groups[
                                                                                                                  i] == g],
                                                                                                             key=lambda
                                                                                                                 p:
                                                                                                             self.results[
                                                                                                                 p].get(
                                                                                                                 'mAP50-95',
                                                                                                                 0))][
                                                                                                         'label'],
                                                   f"{ps:.4f}"])
        t3 = ax_rank3.table(cellText=c3 or [["—", "—", "—"]], colLabels=["Group", "Top Model", "mAP50-95"],
                            loc='center', cellLoc='center')
        t3.auto_set_font_size(False)
        t3.set_fontsize(10)
        t3.scale(1, 1.8)
        t3.auto_set_column_width(col=list(range(3)))
        for (r, c), cell in t3.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        plt.suptitle(self.report_title, fontsize=24, fontweight='bold', y=0.96)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05)
        self._savefig(fig, "combined_dashboard", "dashboards")

    def _generate_all_plots(self, baseline_params):
        names = [self.results[p]['label'] for p in self.model_paths]
        (base_strategies, color_map, bar_colors, bar_hatches, positions, group_centers, group_names_unique,
         unique_strategies) = self._build_color_and_positions(names)

        print(f"\n{C['bold']}Generating organized plots → {self.save_dir}/{C['res']}")

        self._plot_confidence_boxplot(names, bar_colors, bar_hatches, positions, group_centers, group_names_unique,
                                      color_map, unique_strategies)
        self._plot_map_bar('mAP50', 'mAP@50', names, bar_colors, bar_hatches, positions, group_centers,
                           group_names_unique, color_map, unique_strategies)
        self._plot_map_bar('mAP50-95', 'mAP@50-95', names, bar_colors, bar_hatches, positions, group_centers,
                           group_names_unique, color_map, unique_strategies)
        self._plot_sparsity_vs_accuracy('mAP50', 'mAP@50', names, baseline_params, color_map, bar_colors,
                                        unique_strategies)
        self._plot_sparsity_vs_accuracy('mAP50-95', 'mAP@50-95', names, baseline_params, color_map, bar_colors,
                                        unique_strategies)
        self._plot_ranking_tables(names, baseline_params)
        self._plot_combined_dashboard(baseline_params, names, bar_colors, bar_hatches, positions, group_centers,
                                      group_names_unique, color_map, unique_strategies)

    def cleanup(self):
        if os.path.exists(self.data_yaml):
            os.remove(self.data_yaml)


# ─────────────────────────────────────────────────────────────────────────────
#  EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    MODELS_TO_COMPARE = [
        "weights/yolov8/yolov8m/yolov8m.pt",

        "weights/yolov8/yolov8m/no_repair/0.1/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.1_c2f_true_no_repair.pt",
        "weights/yolov8/yolov8m/data_driven_repair/0.1/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",
        "weights/yolov8/yolov8m/data_free_repair/0.1/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.1_c2f_true_data_free_repair.pt",

        "weights/yolov8/yolov8m/no_repair/0.2/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.2_c2f_true_no_repair.pt",
        "weights/yolov8/yolov8m/data_driven_repair/0.2/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",
        "weights/yolov8/yolov8m/data_free_repair/0.2/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.2_c2f_true_data_free_repair.pt",

        "weights/yolov8/yolov8m/no_repair/0.3/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.3_c2f_true_no_repair.pt",
        "weights/yolov8/yolov8m/data_driven_repair/0.3/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt",
        "weights/yolov8/yolov8m/data_free_repair/0.3/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.3_c2f_true_data_free_repair.pt"
    ]

    # Dynamically extract baseline information
    baseline_filename = os.path.basename(MODELS_TO_COMPARE[0])
    baseline_variant = os.path.splitext(baseline_filename)[0]
    first_folded_path = MODELS_TO_COMPARE[1]
    extracted_layers = os.path.basename(first_folded_path).split('_pr')[0].replace('yolo_', '')
    CUSTOM_LABELS = [
        "Baseline",
        "PR 0.1: No Repair",
        "PR 0.1: Data Driven",
        "PR 0.1: Data Free",
        "PR 0.2: No Repair",
        "PR 0.2: Data Driven",
        "PR 0.2: Data Free",
        "PR 0.3: No Repair",
        "PR 0.3: Data Driven",
        "PR 0.3: Data Free",
    ]

    GROUPS = [
        "Baseline",
        "Pairing Rate: 0.1",
        "Pairing Rate: 0.1",
        "Pairing Rate: 0.1",
        "Pairing Rate: 0.2",
        "Pairing Rate: 0.2",
        "Pairing Rate: 0.2",
        "Pairing Rate: 0.3",
        "Pairing Rate: 0.3",
        "Pairing Rate: 0.3"
    ]

    REPORT_TITLE = f"{baseline_variant} {extracted_layers} Full comparison"
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