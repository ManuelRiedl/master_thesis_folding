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

# This plotting file was made with the help of GEMINI

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
    """Return GFLOPs for a single forward pass at the given image size.
    Returns 0.0 if no FLOPs backend is available."""
    if _FLOPS_BACKEND is None:
        return 0.0

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    try:
        if _FLOPS_BACKEND == "thop":
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            return macs * 2 / 1e9          # MACs → FLOPs → GFLOPs
        else:  # fvcore
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
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

        if _FLOPS_BACKEND:
            print(f"   {C['g']}FLOPs backend: {_FLOPS_BACKEND}{C['res']}")
        else:
            print(f"   {C['y']}No FLOPs backend found. Install 'thop' (pip install thop) to enable GFLOPs.{C['res']}")

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
        safe_path_name = model_path.replace('/', '_').replace('\\', '_').replace('.pt', '')
        dataset_name = os.path.basename(self.image_dir)
        return os.path.join(self.stats_dir, f"{safe_path_name}_{dataset_name}_stats.json")

    def _get_txt_path(self, model_path):
        """Human-readable .txt counterpart of the JSON cache."""
        return self._get_cache_path(model_path).replace('_stats.json', '_stats.txt')

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

    def _save_txt_report(self, model_path, data):
        """Write a pretty, human-readable .txt alongside the JSON cache."""
        txt_path = self._get_txt_path(model_path)
        sep = "=" * 60
        with open(txt_path, 'w') as f:
            f.write(f"{sep}\n")
            f.write(f"  Model Statistics Report\n")
            f.write(f"{sep}\n")
            f.write(f"  Label        : {data['label']}\n")
            f.write(f"  Model Path   : {model_path}\n")
            f.write(f"  Saved at     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{sep}\n\n")
            f.write(f"  ARCHITECTURE\n")
            f.write(f"  {'Parameters (M)':<22}: {data['params'] / 1e6:.4f}\n")
            f.write(f"  {'GFLOPs':<22}: {data.get('gflops', 0.0):.4f}\n\n")
            f.write(f"  ACCURACY\n")
            f.write(f"  {'mAP@50':<22}: {data.get('mAP50', 0.0):.4f}\n")
            f.write(f"  {'mAP@50-95':<22}: {data.get('mAP50-95', 0.0):.4f}\n")
            f.write(f"  {'Precision':<22}: {data.get('precision', 0.0):.4f}\n")
            f.write(f"  {'Recall':<22}: {data.get('recall', 0.0):.4f}\n")
            f.write(f"  {'F1-Score':<22}: {data.get('f1_score', 0.0):.4f}\n\n")
            f.write(f"  CONFIDENCE / DETECTION\n")
            f.write(f"  {'Avg Confidence':<22}: {data.get('avg_conf', 0.0):.4f}\n")
            f.write(f"  {'Avg Active Anchors':<22}: {data.get('avg_anchors', 0.0):.1f}\n")
            f.write(f"\n{sep}\n")
        print(f"   {C['g']}Human-readable stats saved to {txt_path}{C['res']}")

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
                # CRITICAL CACHE FIX: Forcefully update the label in memory
                self.results[path]['label'] = m_label
                continue

            print(f"   {C['dim']}No cache found. Running full validation...{C['res']}")
            model = YOLO(path)
            val_metrics = model.val(data=self.data_yaml, split='val', verbose=False, imgsz=self.imgsz)

            raw_model = model.model.to(self.device).eval()
            confs, counts = [], []
            param_count = sum(p.numel() for p in raw_model.parameters())

            # ── FLOPs ──────────────────────────────────────────────────────
            print(f"   {C['dim']}Computing GFLOPs...{C['res']}")
            gflops = _compute_gflops(raw_model, self.imgsz, self.device)
            print(f"   GFLOPs: {gflops:.3f}")

            # ── Confidence sweep ───────────────────────────────────────────
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

            # Save JSON
            with open(cache_path, 'w') as f:
                json.dump(self.results[path], f, indent=2)
            print(f"   {C['g']}Statistics (JSON) saved to {cache_path}{C['res']}")

            # Save TXT
            self._save_txt_report(path, self.results[path])

            del model
            del raw_model
            torch.cuda.empty_cache()

    # ── console report ────────────────────────────────────────────────────────

    def generate_report(self):
        baseline_path = self.model_paths[0]
        baseline_params = self.results[baseline_path]['params']
        baseline_gflops = self.results[baseline_path].get('gflops', 0.0)

        print("\n" + "=" * 145)
        grp_col = f"{'Group':<18} | " if self.groups else ""
        header = (f"{grp_col}{'Model Name':<25} | {'Params (M)':<10} | {'GFLOPs':<8} | "
                  f"{'Reduct.%':<9} | {'mAP50':<7} | {'mAP50-95':<8} | "
                  f"{'F1-Score':<8} | {'Avg Conf':<8} | {'Anchors'}")
        print(f"{C['bold']}{header}{C['res']}")
        print("-" * 145)

        for idx, path in enumerate(self.model_paths):
            data = self.results[path]
            name = data['label']
            params_millions = data['params'] / 1e6
            gflops = data.get('gflops', 0.0)
            reduction = (1 - (data['params'] / baseline_params)) * 100
            red_str = f"-{reduction:.2f}%" if reduction > 0.01 else "-"
            display_name = name[:23] + ".." if len(name) > 25 else name
            grp_str = f"{self.groups[idx][:17]:<18} | " if self.groups else ""
            print(
                f"{grp_str}{display_name:<25} | {params_millions:<10.2f} | {gflops:<8.3f} | "
                f"{red_str:<9} | {data.get('mAP50', 0.0):.4f}  | {data['mAP50-95']:.4f}   | "
                f"{data['f1_score']:.4f}   | {data['avg_conf']:.4f}   | {data['avg_anchors']:.1f}")
        print("=" * 145)

        self._generate_all_plots(baseline_params, baseline_gflops)

    # ── shared colour / position helpers ─────────────────────────────────────

    def _build_color_and_positions(self, names):
        """Return base_strategies, color_map, bar_colors, bar_hatches, positions, group_centers, group_names_unique."""
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

        # Dynamic intra-group spacing
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

    def _savefig(self, fig, name_suffix):
        """Save figure to plots/ dir and close it."""
        safe_title = "".join(c if c.isalnum() else "_" for c in self.report_title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = os.path.join(self.save_dir, f"{name_suffix}_{safe_title}_{timestamp}.png")
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        print(f"   {C['g']}Saved: {save_name}{C['res']}")
        plt.close(fig)

    def _plot_confidence_boxplot(self, names, bar_colors, bar_hatches, positions,
                                  group_centers, group_names_unique, color_map, unique_strategies):
        fig, ax = plt.subplots(figsize=(14, 6))
        conf_data = [self.results[p]['confs_list'] for p in self.model_paths]

        bplot = ax.boxplot(conf_data, positions=positions, patch_artist=True,
                           medianprops=dict(color='black', linewidth=2.0))
        for patch, color, hatch in zip(bplot['boxes'], bar_colors, bar_hatches):
            patch.set_facecolor(mcolors.to_rgba(color, alpha=0.7))
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            patch.set_hatch(hatch)

        # group separators
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
        self._savefig(fig, "plot_confidence_boxplot")

    def _plot_map_bar(self, metric_key, metric_label, names, bar_colors, bar_hatches, positions,
                       group_centers, group_names_unique, color_map, unique_strategies):
        fig, ax = plt.subplots(figsize=(14, 6))
        values = [self.results[p].get(metric_key, 0.0) for p in self.model_paths]
        rgba_colors = [mcolors.to_rgba(c, alpha=0.85) for c in bar_colors]

        bars = ax.bar(positions, values, color=rgba_colors, edgecolor='black', linewidth=1.2, width=0.8)
        for bar, hatch in zip(bars, bar_hatches):
            bar.set_hatch(hatch)

        # value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha='center', va='bottom', fontsize=8)

        # group separators
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
        self._savefig(fig, f"plot_{metric_key.replace('-', '_').replace('@', '')}_bar")

    def _plot_sparsity_vs_accuracy(self, metric_key, metric_label, names, baseline_params,
                                    color_map, bar_colors, unique_strategies):
        """Sparsity (parameter reduction %) vs. Accuracy line plot — one per metric.
        Style inspired by the reference figure."""
        fig, ax = plt.subplots(figsize=(9, 6))

        # Group models by strategy so each gets its own line
        strategy_paths = {}
        for idx, path in enumerate(self.model_paths):
            name = names[idx]
            if "Baseline" in name or "YOLO" in name:
                strat = "Baseline"
            else:
                strat = name.split(': ')[-1] if ': ' in name else name
            strategy_paths.setdefault(strat, []).append(path)

        palette = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c', '#f39c12']

        for s_idx, (strat, paths) in enumerate(strategy_paths.items()):
            color = color_map.get(strat, palette[s_idx % len(palette)])
            xs, ys = [], []
            for p in paths:
                data = self.results[p]
                sparsity = (1 - data['params'] / baseline_params)
                acc = data.get(metric_key, 0.0)
                xs.append(sparsity)
                ys.append(acc)
            # sort by sparsity so the line is monotone
            pairs = sorted(zip(xs, ys))
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
            ax.plot(xs, ys, marker='o', linewidth=2, markersize=6, color=color, label=strat)

        ax.set_xlabel("Sparsity  (parameter reduction)", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f"Sparsity vs. {metric_label}", fontweight='bold', fontsize=14)
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.suptitle(self.report_title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._savefig(fig, f"plot_sparsity_vs_{metric_key.replace('-', '_').replace('@', '')}")

    def _plot_ranking_tables(self, names, baseline_params):
        """Three small ranking tables as a standalone figure."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        for ax in (ax1, ax2, ax3):
            ax.axis('off')

        # --- Top-5 F1 ---
        ax1.set_title("Top 5 Overall (F1-Score)", fontweight='bold', pad=10)
        sorted_f1 = sorted(self.model_paths, key=lambda p: self.results[p].get('f1_score', 0), reverse=True)[:5]
        cols1 = ["Rank", "Group", "Model", "F1-Score"]
        cells1 = []
        for i, p in enumerate(sorted_f1):
            idx = self.model_paths.index(p)
            grp = self.groups[idx].replace("Pairing Rate: ", "PR ") if self.groups else "-"
            cells1.append([f"#{i + 1}", grp, textwrap.fill(self.results[p]['label'], 15),
                           f"{self.results[p].get('f1_score', 0):.4f}"])
        t1 = ax1.table(cellText=cells1, colLabels=cols1, loc='center', cellLoc='center')
        t1.auto_set_font_size(False); t1.set_fontsize(10); t1.scale(1, 1.8)
        for (r, c), cell in t1.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        # --- Best per group F1 ---
        ax2.set_title("Best per Group (F1-Score)", fontweight='bold', pad=10)
        cells2 = []
        if self.groups:
            group_rankings: dict = {}
            for idx, p in enumerate(self.model_paths):
                g = self.groups[idx].replace("Pairing Rate: ", "PR ")
                group_rankings.setdefault(g, []).append(p)
            for g, paths in group_rankings.items():
                if "Baseline" in g: continue
                best_p = max(paths, key=lambda p: self.results[p].get('f1_score', 0))
                cells2.append([g, textwrap.fill(self.results[best_p]['label'], 15),
                               f"{self.results[best_p].get('f1_score', 0):.4f}"])
        t2 = ax2.table(cellText=cells2 or [["—", "—", "—"]], colLabels=["Group", "Top Model", "F1-Score"],
                       loc='center', cellLoc='center')
        t2.auto_set_font_size(False); t2.set_fontsize(10); t2.scale(1, 1.8)
        for (r, c), cell in t2.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        # --- Best per group mAP ---
        ax3.set_title("Best per Group (mAP@50-95)", fontweight='bold', pad=10)
        cells3 = []
        if self.groups:
            group_rankings2: dict = {}
            for idx, p in enumerate(self.model_paths):
                g = self.groups[idx].replace("Pairing Rate: ", "PR ")
                group_rankings2.setdefault(g, []).append(p)
            for g, paths in group_rankings2.items():
                if "Baseline" in g: continue
                best_p = max(paths, key=lambda p: self.results[p].get('mAP50-95', 0))
                cells3.append([g, textwrap.fill(self.results[best_p]['label'], 15),
                               f"{self.results[best_p].get('mAP50-95', 0):.4f}"])
        t3 = ax3.table(cellText=cells3 or [["—", "—", "—"]], colLabels=["Group", "Top Model", "mAP50-95"],
                       loc='center', cellLoc='center')
        t3.auto_set_font_size(False); t3.set_fontsize(10); t3.scale(1, 1.8)
        for (r, c), cell in t3.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        plt.suptitle(self.report_title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._savefig(fig, "plot_ranking_tables")

    def _plot_combined_dashboard(self, baseline_params, names, bar_colors, bar_hatches,
                                  positions, group_centers, group_names_unique,
                                  color_map, unique_strategies):
        """Original combined overview figure."""
        fig = plt.figure(figsize=(24, 24))
        gs = fig.add_gridspec(3, 6, height_ratios=[3.0, 3.5, 1.0], hspace=0.5, wspace=0.8)

        ax1 = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:])
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')
        ax_rank1 = fig.add_subplot(gs[2, 0:2]); ax_rank1.axis('off')
        ax_rank2 = fig.add_subplot(gs[2, 2:4]); ax_rank2.axis('off')
        ax_rank3 = fig.add_subplot(gs[2, 4:6]); ax_rank3.axis('off')

        conf_data = [self.results[p]['confs_list'] for p in self.model_paths]
        maps_95 = [self.results[p]['mAP50-95'] for p in self.model_paths]

        # group separators helper
        def _add_vsep(ax_):
            for i in range(1, len(self.model_paths)):
                if self.groups and self.groups[i] != self.groups[i - 1]:
                    ax_.axvline((positions[i - 1] + positions[i]) / 2, color='gray', linestyle=':', alpha=0.5)

        # Boxplot
        bplot = ax1.boxplot(conf_data, positions=positions, patch_artist=True,
                            medianprops=dict(color='black', linewidth=2.0))
        for patch, color, hatch in zip(bplot['boxes'], bar_colors, bar_hatches):
            patch.set_facecolor(mcolors.to_rgba(color, alpha=0.7))
            patch.set_edgecolor('black'); patch.set_linewidth(1.2); patch.set_hatch(hatch)
        _add_vsep(ax1)
        ax1.set_title("Confidence Distribution Shift", fontweight='bold')
        ax1.axhline(y=0.25, color='r', linestyle='--')
        ax1.set_xticks(group_centers)
        ax1.set_xticklabels(group_names_unique, rotation=0, fontweight='bold', fontsize=12)

        # mAP50-95 bars
        rgba_bar_colors = [mcolors.to_rgba(c, alpha=0.8) for c in bar_colors]
        bars = ax2.bar(positions, maps_95, color=rgba_bar_colors, edgecolor='black', linewidth=1.2, width=0.8)
        for bar, hatch in zip(bars, bar_hatches): bar.set_hatch(hatch)
        _add_vsep(ax2)
        ax2.set_xticks(group_centers)
        ax2.set_xticklabels(group_names_unique, rotation=0, fontweight='bold', fontsize=12)
        ax2.set_title("mAP@50-95 Accuracy Comparison", fontweight='bold')
        ax2.set_ylim(0, max(maps_95) * 1.2 if maps_95 else 1.0)

        legend_patches = [mpatches.Patch(color=color_map[s], label=s) for s in unique_strategies]
        legend_patches.append(mpatches.Patch(color='white', label=''))
        legend_patches.append(mpatches.Patch(facecolor='white', edgecolor='black', label='Dynamic Model'))
        legend_patches.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='Static Model'))
        ax2.legend(handles=legend_patches, loc='upper right', title="Legend", fontsize=10)

        # Main table (now includes GFLOPs)
        if self.groups:
            columns = ("Group", "Model Name", "Params (M)", "GFLOPs", "Reduct.%",
                       "mAP50", "mAP50-95", "F1-Score", "Avg Conf", "Anchors")
        else:
            columns = ("Model Name", "Params (M)", "GFLOPs", "Reduct.%",
                       "mAP50", "mAP50-95", "F1-Score", "Avg Conf", "Anchors")

        cell_text = []
        last_group_table = None
        max_f1_per_group, max_map_per_group = {}, {}
        if self.groups:
            for idx_g, p in enumerate(self.model_paths):
                g = self.groups[idx_g]
                f1 = self.results[p].get('f1_score', 0)
                ms = self.results[p].get('mAP50-95', 0)
                if g not in max_f1_per_group or f1 > max_f1_per_group[g]: max_f1_per_group[g] = f1
                if g not in max_map_per_group or ms > max_map_per_group[g]: max_map_per_group[g] = ms

        for idx, path in enumerate(self.model_paths):
            data = self.results[path]
            params_m = data['params'] / 1e6
            gflops = data.get('gflops', 0.0)
            reduction = (1 - data['params'] / baseline_params) * 100
            red_str = f"-{reduction:.2f}%" if reduction > 0.01 else "-"
            row = []
            if self.groups:
                current_group = self.groups[idx]
                row.append("" if current_group == last_group_table else textwrap.fill(current_group, width=15))
                last_group_table = current_group
            row.extend([
                textwrap.fill(data['label'], width=28),
                f"{params_m:.2f}",
                f"{gflops:.3f}",
                red_str,
                f"{data.get('mAP50', 0.0):.4f}",
                f"{data['mAP50-95']:.4f}",
                f"{data.get('f1_score', 0):.4f}",
                f"{data['avg_conf']:.4f}",
                f"{data['avg_anchors']:.1f}"
            ])
            cell_text.append(row)

        table = ax_table.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.3)

        model_name_col = 1 if self.groups else 0
        map95_col = 6 if self.groups else 4
        f1_col = 7 if self.groups else 5

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold'); cell.set_facecolor('#e0e0e0')
            elif row > 0:
                path_idx = row - 1
                if col == model_name_col:
                    cell.set_facecolor(mcolors.to_rgba(bar_colors[path_idx], alpha=0.25))
                elif col == f1_col and self.groups:
                    p = self.model_paths[path_idx]; g = self.groups[path_idx]
                    if abs(self.results[p].get('f1_score', 0) - max_f1_per_group.get(g, -1)) < 1e-6:
                        cell.get_text().set_color('#27ae60'); cell.get_text().set_weight('bold')
                elif col == map95_col and self.groups:
                    p = self.model_paths[path_idx]; g = self.groups[path_idx]
                    if abs(self.results[p].get('mAP50-95', 0) - max_map_per_group.get(g, -1)) < 1e-6:
                        cell.get_text().set_color('#27ae60'); cell.get_text().set_weight('bold')

        # Ranking tables (reuse existing logic inline)
        for ax_r in (ax_rank1, ax_rank2, ax_rank3):
            ax_r.axis('off')

        ax_rank1.set_title("Top 5 Overall (F1-Score)", fontweight='bold', pad=10)
        sorted_f1 = sorted(self.model_paths, key=lambda p: self.results[p].get('f1_score', 0), reverse=True)[:5]
        cells1 = []
        for i, p in enumerate(sorted_f1):
            idx_ = self.model_paths.index(p)
            grp_ = self.groups[idx_].replace("Pairing Rate: ", "PR ") if self.groups else "-"
            cells1.append([f"#{i + 1}", grp_, textwrap.fill(self.results[p]['label'], 15),
                           f"{self.results[p].get('f1_score', 0):.4f}"])
        t1 = ax_rank1.table(cellText=cells1, colLabels=["Rank", "Group", "Model", "F1"], loc='center', cellLoc='center')
        t1.auto_set_font_size(False); t1.set_fontsize(10); t1.scale(1, 1.8)
        for (r, c), cell in t1.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        ax_rank2.set_title("Best per Group (F1-Score)", fontweight='bold', pad=10)
        cells2 = []
        if self.groups:
            gr2 = {}
            for idx_, p in enumerate(self.model_paths):
                g_ = self.groups[idx_].replace("Pairing Rate: ", "PR ")
                gr2.setdefault(g_, []).append(p)
            for g_, ps in gr2.items():
                if "Baseline" in g_: continue
                bp = max(ps, key=lambda p: self.results[p].get('f1_score', 0))
                cells2.append([g_, textwrap.fill(self.results[bp]['label'], 15),
                               f"{self.results[bp].get('f1_score', 0):.4f}"])
        t2 = ax_rank2.table(cellText=cells2 or [["—", "—", "—"]], colLabels=["Group", "Top Model", "F1"],
                            loc='center', cellLoc='center')
        t2.auto_set_font_size(False); t2.set_fontsize(10); t2.scale(1, 1.8)
        for (r, c), cell in t2.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        ax_rank3.set_title("Best per Group (mAP@50-95)", fontweight='bold', pad=10)
        cells3 = []
        if self.groups:
            gr3 = {}
            for idx_, p in enumerate(self.model_paths):
                g_ = self.groups[idx_].replace("Pairing Rate: ", "PR ")
                gr3.setdefault(g_, []).append(p)
            for g_, ps in gr3.items():
                if "Baseline" in g_: continue
                bp = max(ps, key=lambda p: self.results[p].get('mAP50-95', 0))
                cells3.append([g_, textwrap.fill(self.results[bp]['label'], 15),
                               f"{self.results[bp].get('mAP50-95', 0):.4f}"])
        t3 = ax_rank3.table(cellText=cells3 or [["—", "—", "—"]], colLabels=["Group", "Top Model", "mAP50-95"],
                            loc='center', cellLoc='center')
        t3.auto_set_font_size(False); t3.set_fontsize(10); t3.scale(1, 1.8)
        for (r, c), cell in t3.get_celld().items():
            if r == 0: cell.set_facecolor('#e0e0e0'); cell.set_text_props(weight='bold')

        plt.suptitle(self.report_title, fontsize=24, fontweight='bold', y=0.96)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05)
        self._savefig(fig, "combined_dashboard")

    # ── master plot dispatcher ────────────────────────────────────────────────

    def _generate_all_plots(self, baseline_params, baseline_gflops):
        names = [self.results[p]['label'] for p in self.model_paths]
        (base_strategies, color_map, bar_colors, bar_hatches,
         positions, group_centers, group_names_unique, unique_strategies) = self._build_color_and_positions(names)

        print(f"\n{C['bold']}Generating plots → {self.save_dir}/{C['res']}")

        # 1. Confidence boxplot
        self._plot_confidence_boxplot(names, bar_colors, bar_hatches, positions,
                                       group_centers, group_names_unique, color_map, unique_strategies)

        # 2. mAP50 bar chart
        self._plot_map_bar('mAP50', 'mAP@50', names, bar_colors, bar_hatches, positions,
                            group_centers, group_names_unique, color_map, unique_strategies)

        # 3. mAP50-95 bar chart
        self._plot_map_bar('mAP50-95', 'mAP@50-95', names, bar_colors, bar_hatches, positions,
                            group_centers, group_names_unique, color_map, unique_strategies)

        # 4. Sparsity vs mAP50
        self._plot_sparsity_vs_accuracy('mAP50', 'mAP@50', names, baseline_params,
                                         color_map, bar_colors, unique_strategies)

        # 5. Sparsity vs mAP50-95
        self._plot_sparsity_vs_accuracy('mAP50-95', 'mAP@50-95', names, baseline_params,
                                         color_map, bar_colors, unique_strategies)

        # 6. Ranking tables
        self._plot_ranking_tables(names, baseline_params)

        # 7. Combined dashboard
        self._plot_combined_dashboard(baseline_params, names, bar_colors, bar_hatches,
                                       positions, group_centers, group_names_unique,
                                       color_map, unique_strategies)

        print(f"\n{C['g']}All plots saved to: {os.path.abspath(self.save_dir)}{C['res']}")

    # ── cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self):
        if os.path.exists(self.data_yaml):
            os.remove(self.data_yaml)


# ─────────────────────────────────────────────────────────────────────────────
#  EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    MODELS_TO_COMPARE = [
        "weights/yolov8m.pt",  # Baseline

        # --- Pairing Rate 0.1 ---
        "weights/no_repair/0.1/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.1_c2f_true_no_repair.pt",
        "weights/data_driven_repair/0.1/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.1_c2f_true_data_driven_repair_calib5000.pt",

        # --- Pairing Rate 0.2 ---
        "weights/no_repair/0.2/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.2_c2f_true_no_repair.pt",
        "weights/data_driven_repair/0.2/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.2_c2f_true_data_driven_repair_calib5000.pt",

        # --- Pairing Rate 0.3 ---
        "weights/no_repair/0.3/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.3_c2f_true_no_repair.pt",
        "weights/data_driven_repair/0.3/c2f_out_fold_true/yolo_conv4_to_conv8_pr0.3_c2f_true_data_driven_repair_calib5000.pt"
    ]

    CUSTOM_LABELS = [
        "Baseline",

        "PR 0.1: No Repair",
        "PR 0.1: Data Driven",

        "PR 0.2: No Repair",
        "PR 0.2: Data Driven",

        "PR 0.3: No Repair",
        "PR 0.3: Data Driven"
    ]

    GROUPS = [
        "Baseline",

        "Pairing Rate: 0.1",
        "Pairing Rate: 0.1",

        "Pairing Rate: 0.2",
        "Pairing Rate: 0.2",

        "Pairing Rate: 0.3",
        "Pairing Rate: 0.3"
    ]

    REPORT_TITLE = "yolo_conv4_to_conv8 Comparative Analysis: Static vs. Data-Driven Folding"
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