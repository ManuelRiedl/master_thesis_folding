import os
import yaml
import torch
from numpy import random
import random
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.models import yolo
import torch.nn as nn
from hkmeans import HKMeans
import utils_new
import cv2
import itertools
import json
from torch.utils.data import Subset, DataLoader


#disable ultralytics logs
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['YOLO_SKIP_CHECK'] = 'True'
#ANSI colours
C = {
    'b': '\033[94m', 'cy': '\033[96m', 'g': '\033[92m', 'y': '\033[93m',
    'r': '\033[91m', 'bold': '\033[1m', 'dim': '\033[2m', 'res': '\033[0m'
}
# We save the clustering matrice U -> For bottleneck layers (we have to use the same matrice) (3.5.3. Residual layers)
u_cache = {}

def save_model(model, yolo_obj, type_name, rate, config_name, num_calib_images=None):
    print(f"\n{C['dim']}Saving folded model into native YOLO dictionary format...{C['res']}")
    ckpt = yolo.ckpt if hasattr(yolo, 'ckpt') else {}
    ckpt['model'] = model.half()
    if hasattr(model, 'names'):
        ckpt['names'] = model.names

    target_dir = os.path.join("weights", str(type_name), str(rate))
    os.makedirs(target_dir, exist_ok=True)

    if num_calib_images is None:
        file_name = f"{config_name}_folded_{type_name}.pt"
    else:
        file_name = f"{config_name}_folded_{type_name}_calib{num_calib_images}.pt"
    save_path = os.path.join(target_dir, file_name)
    torch.save(ckpt, save_path)
    #we save the model as fp16 - but the rest of the code assumes fp32 => inplace operation
    model.float()
    print(f"{C['g']}{C['bold']}Model successfully saved to {save_path}!{C['res']}")



#Loads a batch of the dataset for the data-driven repair (Google GEMINI)
def get_calibration_batch(img_dir, n=32, imgsz=640, device='cpu'):
    if not os.path.exists(img_dir):
        print(f"{C['r']}Warning: Calibration directory not found at {img_dir}. REPAIR will be skipped.{C['res']}")
        return None
    images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:n]
    batch = []
    print(f"   {C['dim']}[Calibration]{C['res']} Loading {len(images)} images for REPAIR...")
    for p in images:
        img = cv2.imread(p)
        img = cv2.resize(img, (imgsz, imgsz))
        img = img.transpose((2, 0, 1))[::-1]
        batch.append(torch.from_numpy(img.copy()).float() / 255.0)

    return torch.stack(batch).to(device)

"""
This function replaces a BN layer with the updated one in memory
"""
def set_module_by_name(model, name, new_module):
    parts = name.split('.')  # eg:  "model.2.cv1.bn" => ["model", "2", "cv1", "bn"]
    # start at the beginning
    parent = model
    # go to the last entry ["bn"] -> Search for the replacement
    for part in parts[:-1]:
        # If part is a number => nn.Sequential or ModuleList (List access) else access the attribute (parent.cv1)
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    # if the last part is a number => we plug in our module in the list
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new_module
    # else we are in a separate module we replace the attribute
    else:
        setattr(parent, parts[-1], new_module)

"""
This function computes the Clustering matrice U by finding similar weights
"""
def cumpute_cluster_matrix_u(conv_L, conv_next, pr):
    print(
        f"   {C['b']}[Step: K-Means]{C['res']} Finding clusters for {C['bold']}{conv_L.out_channels}{C['res']} channels at paring rate of {pr}...")
    # output channels before and after folding
    n_original_channels = conv_L.out_channels
    # k_folded is the number of  "clusters" in the K-Means algorithm
    k_folded = round(n_original_channels * (1 - pr))
    with torch.no_grad():
        # 1. RESHAPE (Flatten) (3.4 - Page 33)
        #    reshape W_L from [C_out, C_in, kernal_H, kernel_W] to [C_out, C_in * kernal_H * kernel_W]
        #    thats because K-Means does not "work" on a 4D matrix (only 2D)
        W_l = conv_L.weight.data.reshape(n_original_channels,
                                         -1)  # -1 => it does figure out the shape on its own 3x3x3 = 27 -> [48,27]
        #    We need now to match the input dimensions of W_l_next to W_L out dimensions
        #    The current shape is: [C_next_out, C_next_in, kernal_H, kernal_W] -> [C_out, C_next_out* kernal_H * kernel_W]]
        #    Since we match the output of L with the input of L+1 we have to reorder the matric before reshaping -> eg we get a [48,864] (96x3x3)
        W_l_next = conv_next.weight.data.permute(1, 0, 2, 3).reshape((n_original_channels, -1))
        # 2. MATRICE A (Algorithm 1.2 - Page 26)
        #     Since we want to minimize the error macros both layers (L, L_next) we have to perform Algorithm 1 (Clustering/K-means) on both
        #     Therefore we have to combine them into one matrice A = [ W_l | W_l_next^T ] (merged by rows - on top of each other)
        A = torch.cat([W_l, W_l_next], dim=1).float().cpu().numpy()

        # 3. Clustering (K-Means) (Algorithm 1/ 3.2 - Page 23/26)
        #     It is a matrice decomposition problem which we can reduce to a K-Means algorithm
        #     simular weight vectors get grouped into one cluster
        #Debug output verbose =1
        #km = KMeans(n_clusters=k_folded, random_state=42, n_init=5,verbose=1)
        km = HKMeans(n_clusters=k_folded, random_state=42, n_init=5,n_jobs=-1)
        km.fit(A)
        labels = km.labels_  # which cluster each channel belongs to
        # 4. Clustering matrice U (Definition 3 - Page 21)
        #     Cluster matrice U => 1 =  assigned cluster
        #     built the clustering matrix U [original channels, folded_channels]
        U = torch.zeros(n_original_channels, k_folded, device=conv_L.weight.device)
        for neuron_index, cluster_idx in enumerate(labels):
            U[neuron_index, cluster_idx] = 1.0
    return U


def get_projection_matrix(U):
    # Mean calculation M-Matrice M = (U^T U)^-1 U^T (Page 23)
    #       (U^T U) is a diagonal matrice -> where each entry is the cluster size (the amount of merged neurons per cluster)
    #        vector which contains the cluster sizes
    device = U.device
    cluster_sizes = torch.sum(U, dim=0)
    #       we take the average of the sum of the weights here -> inverse is 1/cluster size => multiplied by the sum weight value
    M = torch.diag(1.0 / cluster_sizes).to(device) @ U.T
    return M


"""
This function merges weights based on the matrice U for a conv layer followed by a bn layer
The order argument defines if we merge the input weights or adjust the output
"""
def merge_conv_bn(conv_L, bn_L, conv_next, U, order="output", name="Unknown"):
    if U is None:
        return None, None, None

    n_original = U.shape[0]
    n_folded = U.shape[1]
    device = U.device

    with torch.no_grad():
        if order == 'output':
            ctx_color = C['cy'] if '.m.' in name or '.cv' in name else C['b']
            print(f"      {C['dim']}[Debug: Layer]{C['res']} Name: {ctx_color}{name}{C['res']}")
            # 1. Mean calculation M-Matrice M = (U^T U)^-1 U^T (Page 23)
            M = get_projection_matrix(U)
            # 2. Update the values of layer L (Algorithm 1.3) (Page 26)
            original_shape_L = conv_L.weight.data.shape
            W_l_reshaped = conv_L.weight.data.reshape(n_original, -1)
            updated_W_l = M @ W_l_reshaped
            folded_weigh = updated_W_l.reshape(n_folded, *original_shape_L[1:])
            conv_L.weight = nn.Parameter(folded_weigh)
            conv_L.out_channels = n_folded
            print(f"      {C['bold']}{C['b']}[Debug: Conv Output Fold]{C['res']} {n_original} -> {n_folded} channels {C['y']}{list(conv_L.weight.shape)}{C['res']}")

            # 3. BatchNorm layer (Theorem 3.5.3 Page 29)
            new_bn = None
            if bn_L is not None:
                new_bn = nn.BatchNorm2d(n_folded).to(device).to(bn_L.weight.dtype)
                new_bn.weight = nn.Parameter(M @ bn_L.weight.data)
                new_bn.bias = nn.Parameter(M @ bn_L.bias.data)

                # we cant average the stds because it is a squared quantity
                # we would overestimate the true merged variance -> so we need to average before stds (inverse stds)
                inv_stds = 1.0 / torch.sqrt(bn_L.running_var.data + 1e-6)
                new_running_mean_normed = M @ (bn_L.running_mean.data * inv_stds)
                new_inv_stds = M @ inv_stds
                new_running_var = (1.0 / (new_inv_stds + 1e-6)) ** 2
                new_bn.running_mean = new_running_mean_normed * torch.sqrt(new_running_var)
                # Average running_var in std-dev space — prevents overestimation
                new_bn.running_var = new_running_var
                print(f"      {C['b']}[Debug: BN Fold]{C['res']} BN channels updated to a shape of {C['bold']}{n_folded}{C['res']}")
            return conv_L, new_bn, conv_next

        elif order == "input" and conv_next is not None:
            original_shape_next = conv_next.weight.data.shape
            actual_in_channels = original_shape_next[1]
            # This is for the "Block-Diagonal" Expansion -> It is used in C2f Blocks
            # eg: we have: 2 Bottleneck Paths B1, B2
            # n_original: 4 (this means we have identity,split,b1,b2)
            # n_folded: 2 (we fold by 50% so we merge 2 neurons per neuron)
            # Total Input to Concat layer: 8 channels -> Folded to 4
            # U (4x2):          Expanded U (8x4):
            # [ 1, 0 ]               [ U_identiy, 0 ]  => [ 1, 0, 0, 0 ] (Identity)
            # [ 1, 0 ]               [ 0,       U_B ]     [ 1, 0, 0, 0 ]
            # [ 0, 1 ]                                    [ 0, 1, 0, 0 ]
            # [ 0, 1 ]                                    [ 0, 1, 0, 0 ]
            #                                             [ 0, 0, 1, 0 ] (B_B entries)
            #                                             [ 0, 0, 1, 0 ]
            #                                             [ 0, 0, 0, 1 ]
            #                                             [ 0, 0, 0, 1 ]
            num_paths = actual_in_channels // n_original
            if num_paths > 1:
                print(f"      {C['dim']}[Debug: Concat Block]{C['res']} {C['y']}Detected {num_paths} paths. Expanding U diagonally.{C['res']}")
                U_to_use = torch.zeros(actual_in_channels, n_folded * num_paths, device=device)
                for i in range(num_paths):
                    U_to_use[i * n_original:(i + 1) * n_original, i * n_folded:(i + 1) * n_folded] = U
                n_fold_in = n_folded * num_paths
            else:
                U_to_use, n_fold_in = U, n_folded

            # Fold Input Weights (Algorithm 1, Step 3)
            W_flat = conv_next.weight.data.permute(1, 0, 2, 3).contiguous().reshape(actual_in_channels, -1)
            new_W = (U_to_use.T @ W_flat).reshape(n_fold_in, original_shape_next[0], *original_shape_next[2:])
            conv_next.weight = nn.Parameter(new_W.permute(1, 0, 2, 3))
            conv_next.in_channels = n_fold_in
            print(f"      {C['bold']}{C['y']}[Debug: Conv Input Fold]{C['res']} Current Input: {actual_in_channels} -> {n_fold_in} {C['y']}{list(conv_next.weight.shape)}{C['res']}")
            return None, None, conv_next


"""
This function handles c2f layers - Bottleneck layers use the same U matrice
"""
def c2f_layer_folding(c2f_layer, U_input, model, block_name, pairing_rate):
    with torch.no_grad():
        device = c2f_layer.cv1.conv.weight.device
        # The first is the identity path (splitting by 50%)
        cv1 = c2f_layer.cv1.conv
        n_total = cv1.out_channels
        # so we half our input weight matrix
        half = n_total // 2
        #also with respect to our pairing rate -> so eg if we have a input of 96 -> 50% is 48 and our pairing rate is also 50% => So 24 go to the identity and 24 to the Bottleneck layer
        target_half = round(half * (1 - pairing_rate))
        print(f"   {C['cy']}{C['bold']}[C2F Debug] Creating Constrained U: {half}->{target_half} per side{C['res']}")
        # We built the matrice U
        U_new = torch.zeros(n_total, target_half * 2, device=device)

        # We do this in 2 separate ways because if we fold the 96 channels directly into 48 and split it after into 24 - 24
        # If we dont do this it can happen that for example "more" neurons from one side got merged - or neurons from both sides get merged
        # Weight of size [X, 38] cannot be multiplied by input of size [X, 24]
        #Top half (identity path)
        W_top = cv1.weight.data[:half].reshape(half, -1)
        #Successor is cv2.conv so we use permute(1,0,2,3) to get C_in first so each row = one input channel
        W_cv2_identity = c2f_layer.cv2.conv.weight.data.permute(1, 0, 2, 3)[:half].reshape(half, -1)
        #A = [W_l | W_{l+1}]
        A_top = torch.cat([W_top, W_cv2_identity], dim=1).float().cpu().numpy()
        km_top = HKMeans(n_clusters=target_half, random_state=42, n_init=10,n_jobs=-1).fit(A_top)
        for i, lab in enumerate(km_top.labels_):
            U_new[i, lab] = 1.0

        #Bottom half (bottleneck paths)
        W_bot = cv1.weight.data[half:].reshape(half, -1)
        W_bn_input = c2f_layer.m[0].cv1.conv.weight.data.permute(1, 0, 2, 3).reshape(half, -1)
        A_bot = torch.cat([W_bot, W_bn_input], dim=1).float().cpu().numpy()
        km_bot = HKMeans(n_clusters=target_half, random_state=42, n_init=10,n_jobs=-1).fit(A_bot)
        for i, lab in enumerate(km_bot.labels_):
            U_new[i + half, lab + target_half] = 1.0

        #Fold cv1 -> The input layer before the split layer - where we split into the identity path and the bottleneck layers
        bn_cv1_name = f"{block_name}.cv1.bn"
        bn_cv1 = get_module_by_name(model, bn_cv1_name)
        _, bn_f, _ = merge_conv_bn(cv1, bn_cv1, None, U_new, order='output', name=bn_cv1_name)
        set_module_by_name(model, bn_cv1_name, bn_f)

        #Fold the Bottleneck Layers -> We use the bottom "half" of the clustering matrice -> We use the same for all of them.
        U_sliced = U_new[half:, target_half:]
        for i, bottleneck in enumerate(c2f_layer.m):
            conv1_name = f"{block_name}.m.{i}.cv1.conv"
            conv2_name = f"{block_name}.m.{i}.cv2.conv"
            #for cv1 we can compute a new clustering matrice -> since it doesnt go into the concat connection - only cv2 does
            U_cv1 = cumpute_cluster_matrix_u(bottleneck.cv1.conv, bottleneck.cv2.conv, pairing_rate)
            u_cache[conv1_name] = U_cv1
            u_cache[conv2_name] = U_sliced
            #Adjust the input of the first conv layer inside the bottleneck
            merge_conv_bn(None, None, bottleneck.cv1.conv, U_sliced, order='input', name=conv1_name)
            # fold BN1
            bn_b1_name = f"{block_name}.m.{i}.cv1.bn"
            bn_b1 = get_module_by_name(model, bn_b1_name)
            _, b1_f, _ = merge_conv_bn(bottleneck.cv1.conv, bn_b1, None, U_cv1, order='output', name=bn_b1_name)
            set_module_by_name(model, bn_b1_name, b1_f)
            # Adjust the input of the second conv layer
            merge_conv_bn(None, None, bottleneck.cv2.conv, U_cv1, order='input', name=conv2_name)
            # fold BN2
            bn_b2_name = f"{block_name}.m.{i}.cv2.bn"
            bn_b2 = get_module_by_name(model, bn_b2_name)
            _, b2_f, _ = merge_conv_bn(bottleneck.cv2.conv, bn_b2, None, U_sliced, order='output', name=bn_b2_name)
            set_module_by_name(model, bn_b2_name, b2_f)

        # Adjust the input of the last conv Layer in the bottleneck -> The concat layer is handled inside the function
        # We do this here to built a matrice which uses the Identy_clustering and the bottleneck clusterings -> We built a huge diagonal matrice with the diagonals
        # idendity path
        U_top = U_new[:half, :target_half]
        # bottleneck path
        U_bot = U_new[half:, target_half:]
        # how many bottlneck paths we have
        num_bottlenecks = len(c2f_layer.m)
        actual_cv2_in = n_total + half * num_bottlenecks
        total_out_cols = target_half * (2 + num_bottlenecks)
        U_cv2 = torch.zeros(actual_cv2_in, total_out_cols, device=device)
        # First we have the identity path
        U_cv2[:half, :target_half] = U_top
        # Second - the the other half is directly passed to the concat
        U_cv2[half:n_total, target_half:2 * target_half] = U_bot
        # 2..N -> bottleneck adds
        for i in range(num_bottlenecks):
            row_start = n_total + i * half
            col_start = (i + 2) * target_half
            U_cv2[row_start:row_start + half, col_start:col_start + target_half] = U_bot

        merge_conv_bn(None, None, c2f_layer.cv2.conv, U_cv2, order='input', name=f"{block_name}.cv2.conv")
    return U_new


def get_module_by_name(model, name):
    for part in name.split('.'):
        model = model[int(part)] if part.isdigit() else getattr(model, part)
    return model


"""This function fine tunes the folded layers . ONLY this layers get recalibratet (running statistics)"""
def repair_bn_forward_pass(model, loader, device, folding_plan=None, max_samples=1000, verbose=True):
    #get all BN layers from the model
    all_bn_layers = {name: m for name, m in model.named_modules()
              if isinstance(m, nn.BatchNorm2d)}
    if folding_plan is not None:
        #get the actual layers that we folded
        folded_conv_layers = [name for name, cfg in folding_plan.items()
                        if cfg.get("do_folding", False)]
        #mapping of conv layer with associated bn layer (eg: model.4.cv1.conv => model.4.cv1.bn)
        affected_bn_names = set()
        for conv_name in folded_conv_layers:
            if conv_name.endswith(".conv"):
                bn_name = conv_name[:-len(".conv")] + ".bn"
                if bn_name in all_bn_layers:
                    affected_bn_names.add(bn_name)
                else:
                    if verbose:
                        print(f"   {C['y']}[REPAIR] No BN found for {conv_name} → {bn_name} (skipped){C['res']}")
        bn_to_reset = {n: all_bn_layers[n] for n in affected_bn_names}
    else:
        #when no plan is provided => reset all running statistics off all layers
        bn_to_reset = all_bn_layers
    if not bn_to_reset:
        if verbose:
            print(f"   {C['y']}[REPAIR] No BN layers to reset — skipping.{C['res']}")
        return model
    #reset the running statistics in the bn layers
    for bn in bn_to_reset.values():
        bn.momentum = None
        bn.reset_running_stats()
    if verbose:
        print(f"\n{C['bold']}{C['cy']}--- REPAIR: BN Forward-Pass Recalibration ---{C['res']}")
        print(f"   {C['dim']}Resetting {len(bn_to_reset)}/{len(all_bn_layers)} BN layers "
              f"(folded only):{C['res']}")
        for n in sorted(bn_to_reset):
            print(f"      {C['dim']}- {n}{C['res']}")
    #Forward pass => Recalibrate running statistics
    model.train()
    seen = 0
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device=device, dtype=model_dtype)
            try:
                model(images)
            except Exception as e:
                if verbose:
                    print(f"   {C['r']}[REPAIR] Forward pass error: {e}{C['res']}")
                break
            seen += images.shape[0]
            if verbose:
                print(f"   {C['dim']}Samples seen: {seen}/{max_samples}{C['res']}", end="\r")
            if seen >= max_samples:
                break
    model.eval()
    if verbose:
        print(f"\n   {C['g']}REPAIR complete. {len(bn_to_reset)} BN layers recalibrated "
              f"on {seen} samples.{C['res']}")
    return model

def run_folding_experiment(weights_path, config_path, pairing_rate, calib_images, do_repair, calib_ds="coco/labels/train2017"):
    # load yolo weights
    filename = os.path.basename(config_path)
    config_name = os.path.splitext(filename)[0]
    print(f"\n{C['bold']}============================================================{C['res']}")
    print(f"{C['b']}STARTING RUN: Config: {config_name} | PR: {pairing_rate} | Calib N: {calib_images}{C['res']}")
    print(f"{C['bold']}============================================================{C['res']}")
    yolo = YOLO(weights_path)
    model = yolo.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using {next(model.parameters()).device}")

    # load folding configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            folding_plan = yaml.safe_load(f)
    else:
        print(f"{C['r']}No config_folding found. Exit.{C['res']}")
        exit(-1)

    print(f"\n{C['bold']}[BEFORE FOLDING]{C['res']}")
    initial_params = utils_new.count_parameters(model)
    print(f"Total Parameters: {C['bold']}{initial_params:,}{C['res']}")

    #We need this for the folding statistcs later (mainly because of the user output - so we have a clear overview what changed)
    print(f"\n{C['dim']}[Capturing Model Snapshot]{C['res']}")
    shape_snapshot = {
        name: (mod.in_channels if hasattr(mod, 'in_channels') else mod.num_features,
               mod.out_channels if hasattr(mod, 'out_channels') else mod.num_features)
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.BatchNorm2d))
    }
    utils_new.check_layer_shapes(model, folding_plan, shape_snapshot=shape_snapshot, hide_bn=True, internal_name=True,
                                 show_reduction=False)
    #We save the layers we already folded - so we do not fold them twice (beacuse of the C2f block logic)
    processed_layers = set()
    print(f"\n{C['bold']}{C['cy']}--- Starting Model Folding Engine ---{C['res']}")
    for module_name, settings in folding_plan.items():
        if not settings.get('do_folding') or module_name in processed_layers or ".bn" in module_name:
            continue
        print(f"\n{C['bold']}Folding Layer: {C['b']}{module_name}{C['res']}")
        try:
            layer_L = module_name
            conv_L = get_module_by_name(model, layer_L)
            conv_next = None
            conv_next_name = None
            for next_name, next_settings in folding_plan.items():
                if next_settings.get('pre') == layer_L:
                    conv_next = get_module_by_name(model, next_name)
                    conv_next_name = next_name
                    break
            pairing_r = settings.get('pr', pairing_rate)
            ref_mapping = settings.get('consistent_map')

            if ref_mapping and ref_mapping in u_cache:
                print(f"   {C['dim']}[Consistent Map]{C['res']} Inheriting U from {ref_mapping}")
                U_matrix = u_cache[ref_mapping]
            else:
                U_matrix = cumpute_cluster_matrix_u(conv_L, conv_next, pairing_r)
                u_cache[layer_L] = U_matrix

            parts = module_name.split('.')
            c2f_block_name = ".".join(parts[:2])
            c2f_candidate = get_module_by_name(model, c2f_block_name)

            if "C2f" in str(type(c2f_candidate)) and "cv1.conv" in module_name:
                U_final = c2f_layer_folding(c2f_candidate, U_matrix, model, c2f_block_name, pairing_r)
                u_cache[layer_L] = U_final
                for inner_name in folding_plan.keys():
                    if c2f_block_name in inner_name:
                        processed_layers.add(inner_name)
            else:
                bn_name = layer_L.replace(".conv", ".bn")
                bn_L = get_module_by_name(model, bn_name)
                _, bn_folded, _ = merge_conv_bn(conv_L, bn_L, None, U_matrix, order="output", name=bn_name)
                if bn_folded is not None:
                    set_module_by_name(model, bn_name, bn_folded)
                if conv_next is not None:
                    merge_conv_bn(None, None, conv_next, U_matrix, order="input", name=conv_next_name)

            print(f"   {C['g']}Successfully folded {module_name}{C['res']}")

        except Exception as e:
            print(f"   {C['r']}Skipping {module_name} ERROR: {e}{C['res']}")
            continue

    print(f"\n{C['bold']}[AFTER FOLDING]{C['res']}")
    final_params = utils_new.count_parameters(model)
    utils_new.check_layer_shapes(model, folding_plan, shape_snapshot=shape_snapshot, hide_bn=True, internal_name=True,
                                 show_reduction=True)
    reduction = (1 - final_params / initial_params) * 100
    print(f"\n{C['bold']}Total Parameters: {final_params:,} ({C['g']}{reduction:.2f}% reduction{C['res']}{C['bold']}){C['res']}")
    utils_new.test_forward_pass(model, device)
    save_model(model,yolo, "without_repair",pairing_rate,config_name)
    if do_repair:
        #build the data loader for the fine-tuning
        full_dataset = utils_new.COCOImageFolder(
            image_dir=calib_ds,
            imgsz=640,
            max_images=None  # We need the full pool to draw randomly from!
        )

        # 2. Generate 1.000 completely random, unique indices
        # This naturally preserves the rough distribution of the whole dataset
        total_images = len(full_dataset)
        random_indices = random.sample(range(total_images), calib_images)

        # 3. Create a Sub-Dataset with those random 1.000 images
        random_subset = Subset(full_dataset, random_indices)

        # 4. Pass the subset to your DataLoader
        train_loader = DataLoader(
            random_subset,
            batch_size=16,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        repair_bn_forward_pass(model, train_loader, device,folding_plan=folding_plan, max_samples=calib_images)
        save_model(model, yolo,f"forward_pass_repair",pairing_rate,config_name,num_calib_images=calib_images)
    del model
    del yolo
    torch.cuda.empty_cache()


def main():
    WEIGHTS_PATH = "weights/yolov8m.pt"
    CALIB_DS = "coco/images/train2017"
    #if this is None => manual experimen is used
    EXPERIMENTS_FILE = "config_experiments/experiment_1.json"

    # MANUAL PARAMETERS (Used ONLY if EXPERIMENTS_FILE = None !!)
    manual_experiments = [
        {
            "config": "config/yolo_conv5_conv7.yaml",
            "pairing_rates": [0.1],
            "calib_images": [1000],
            "repair": [True]
        }
    ]

    #Use the appropriate config
    if EXPERIMENTS_FILE and os.path.exists(EXPERIMENTS_FILE):
        print(f"{C['dim']}Found {EXPERIMENTS_FILE}. Loading grid search from JSON...{C['res']}")
        with open(EXPERIMENTS_FILE, 'r') as f:
            experiments = json.load(f)
    else:
        print(f"{C['y']}No {EXPERIMENTS_FILE} found! Defaulting to manual parameters in code.{C['res']}")
        experiments = manual_experiments

    print(f"{C['cy']}Queued {len(experiments)} experiment profiles.{C['res']}\n")

    # 2. Execute the Grid Search
    for exp in experiments:
        config_file = exp["config"]

        # itertools.product creates every possible combination of the lists
        # .get() is used here as a safety net in case you forget a key in the JSON
        combinations = itertools.product(
            exp.get("pairing_rates", [0.1]),
            exp.get("calib_images", [1000]),
            exp.get("repair", [True])
        )

        for pr, calib_n, do_rep in combinations:
            # We clear the u_cache so cross-run U-matrices don't leak into each other
            global u_cache
            u_cache = {}

            run_folding_experiment(
                weights_path=WEIGHTS_PATH,
                config_path=config_file,
                pairing_rate=pr,
                calib_images=calib_n,
                do_repair=do_rep,
                calib_ds=CALIB_DS
            )

if __name__ == "__main__":
    main()