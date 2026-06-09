import os
import yaml
import torch
import random
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.models import yolo
import torch.nn as nn
from hkmeans import HKMeans
import utils_new
import cv2
import itertools
import numpy as np
import json
from torch.utils.data import Subset, DataLoader
import copy
import re
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


def fuse_bn_into_conv(conv, bn):
    with torch.no_grad():
        gamma = bn.weight.data.clone()
        beta = bn.bias.data.clone()
        mean = bn.running_mean.data.clone()
        var = bn.running_var.data.clone()
        eps = bn.eps

        scale = gamma / torch.sqrt(var + eps)
        conv.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)

        if conv.bias is not None:
            conv.bias.data = scale * (conv.bias.data - mean) + beta
        else:
            conv.bias = nn.Parameter(scale * (0 - mean) + beta)

        bn.weight.data.fill_(1.0)
        bn.bias.data.fill_(0.0)
        bn.running_mean.data.fill_(0.0)
        bn.running_var.data.fill_(1.0)


def get_average_correlation(U, W_flat):
    n_folded = U.shape[1]
    avg_corr = np.zeros(n_folded)
    U_np = U.cpu().numpy()
    for cluster_idx in range(n_folded):
        members = np.where(U_np[:, cluster_idx] > 0)[0]
        if len(members) <= 1:
            continue
        pairs = [(m, n) for m, n in itertools.product(members, members) if m != n]
        total = 0.0
        for m, n in pairs:
            a = W_flat[m].flatten()
            b = W_flat[n].flatten()
            denom = np.sqrt((a @ a) * (b @ b))
            if denom > 0:
                total += (a @ b) / denom
        avg_corr[cluster_idx] = total / len(pairs)
    return torch.tensor(avg_corr, device="cuda", dtype=torch.float32)


def fuse_all_batchnorms(model):
    pairs = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            conv_name = name.replace(".bn", ".conv")
            try:
                conv_mod = get_module_by_name(model, conv_name)
                if isinstance(conv_mod, nn.Conv2d):
                    pairs.append((conv_name, conv_mod, name, mod))
            except:
                pass
    print(f"   {C['dim']}[Fold-AR] Fusing {len(pairs)} (Conv, BN) pairs...{C['res']}")
    for conv_name, conv_mod, bn_name, bn_mod in pairs:
        fuse_bn_into_conv(conv_mod, bn_mod)
    print(f"   {C['g']}[Fold-AR] All BN layers fused into their convolutions.{C['res']}")


def save_model(model, yolo_obj, repair_mode, pairing_rate, config_base_name, fold_c2f_output, weights_path, num_calib_images=None):
    print(f"\n{C['dim']}Saving folded model into native YOLO dictionary format...{C['res']}")
    ckpt = yolo_obj.ckpt if hasattr(yolo_obj, 'ckpt') else {}
    # we save the model as fp16 - but the rest of the code assumes fp32
    ckpt['model'] = copy.deepcopy(model).half()
    if hasattr(model, 'names'):
        ckpt['names'] = model.names

    # 1. Map the internal repair modes to your clean folder names
    mode_dir_map = {
        "NO_REPAIR": "no_repair",
        "APPROX_REPAIR": "data_free_repair",
        "REPAIR": "data_driven_repair"
    }
    repair_dir = mode_dir_map.get(repair_mode, "unknown_repair")

    # 2. Extract YOLO version and variant from weights_path
    # Example: "weights/yolov8m.pt" -> filename: "yolov8m.pt" -> base: "yolov8m"
    weights_filename = os.path.basename(weights_path)
    model_variant = os.path.splitext(weights_filename)[0]  # e.g., "yolov8m"

    # Extract the family/version (e.g., "yolov8" from "yolov8m" or "yolo11" from "yolo11n")
    # This regex looks for 'yolo' followed by numbers.
    match = re.match(r'(yolo\w*\d+)', model_variant, re.IGNORECASE)
    yolo_version = match.group(1).lower() if match else "unknown_yolo"

    # 3. Build the nested directory structure
    # Format: weights/yolov8/yolov8m/no_repair/0.1/c2f_out_fold_true
    c2f_dir = f"c2f_out_fold_{str(fold_c2f_output).lower()}"
    target_dir = os.path.join("weights", yolo_version, model_variant, repair_dir, str(pairing_rate), c2f_dir)
    os.makedirs(target_dir, exist_ok=True)

    # 4. Build the comprehensive filename
    # Example: yolo_conv4_conv5_pr0.1005_c2f_true_data_free_repair.pt
    file_name = f"{config_base_name}_pr{pairing_rate}_c2f_{str(fold_c2f_output).lower()}_{repair_dir}"

    # Only append the calibration number if it was an actual data-driven repair
    if repair_mode == "REPAIR" and num_calib_images is not None:
        file_name += f"_calib{num_calib_images}"

    file_name += ".pt"

    save_path = os.path.join(target_dir, file_name)
    torch.save(ckpt, save_path)
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
def cumpute_cluster_matrix_u(conv_L, bn_L, conv_next, pr):
    print(
        f"   {C['b']}[Step: K-Means]{C['res']} Finding clusters for {C['bold']}{conv_L.out_channels}{C['res']} channels at paring rate of {pr}...")
    # output channels before and after folding
    n_original_channels = conv_L.out_channels
    # k_folded is the number of  "clusters" in the K-Means algorithm
    k_folded = round(n_original_channels * (1 - pr))
    with torch.no_grad():
        # 1. RESHAPE (Flatten) (3.4 - Page 33)
        #    reshape W_L from [C_out, C_in, kernal_H, kernel_W] to [C_out, C_in * kernal_H * kernel_W]
        W_l = conv_L.weight.data.reshape(n_original_channels, -1)
        # Permute so that next layer's input channels become rows, then flatten the rest.
        if conv_next is not None:
            W_l_next = conv_next.weight.data.permute(1, 0, 2, 3).reshape((n_original_channels, -1))
        else:
            W_l_next = None
        # 2. MATRICE A
        #    [W_l rows | BN γ | BN β | W_{l+1}^T rows]. Including BN γ/β ensures two filters
        #    that have similar spatial weights but very different BN scaling are NOT clustered.
        parts = [W_l]
        if bn_L is not None:
            parts.append(bn_L.weight.data.view(n_original_channels, 1))  # γ (Σ_s)
            parts.append(bn_L.bias.data.view(n_original_channels, 1))    # β
        if W_l_next is not None:
            parts.append(W_l_next)
        A = torch.cat(parts, dim=1).float().cpu().numpy()

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
def merge_conv_bn(conv_L, bn_L, conv_next, U, order="output", name="Unknown", approximate_repair=False):
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
            # Guard: if the conv has a bias (e.g. Detect head's final 1x1 conv has bias=True),
            # fold it with the same M projection. For bias-less convs (BN-followed) this is a no-op.
            if conv_L.bias is not None:
                conv_L.bias = nn.Parameter(M @ conv_L.bias.data)
            print(f"      {C['bold']}{C['b']}[Debug: Conv Output Fold]{C['res']} {n_original} -> {n_folded} channels {C['y']}{list(conv_L.weight.shape)}{C['res']}")

            # 3. BatchNorm layer (Theorem 3.5.3 Page 29)
            new_bn = None
            if bn_L is not None:
                new_bn = nn.BatchNorm2d(n_folded).to(device).to(bn_L.weight.dtype)

                if approximate_repair:
                    #Fold-AR: Data-Free Repair
                    new_bn.running_mean.data.fill_(0.0)
                    new_bn.running_var.data.fill_(1.0)
                    new_bn.weight.data = M @ bn_L.weight.data
                    new_bn.bias.data = M @ bn_L.bias.data

                    W_original = W_l_reshaped.cpu().numpy()
                    avg_corr = get_average_correlation(U, W_original)
                    n_c = torch.sum(U, dim=0).to(device)
                    scale = torch.sqrt(n_c / (1 + (n_c - 1) * avg_corr.to(device)))
                    new_bn.weight.data = new_bn.weight.data * scale
                    print(
                        f"      {C['b']}[Debug: BN Fold]{C['res']} Fold-AR correlation applied to {n_folded} channels")
                else:
                    #Fold-R: Standard Folding
                    """
                    # we cant average the stds because it is a squared quantity
                    # we would overestimate the true merged variance -> so we need to average before stds (inverse stds)
                    inv_stds = 1.0 / torch.sqrt(bn_L.running_var.data + 1e-6)
                    new_running_mean_normed = M @ (bn_L.running_mean.data * inv_stds)
                    new_inv_stds = M @ inv_stds
                    new_running_var = (1.0 / (new_inv_stds + 1e-6)) ** 2
                    new_bn.running_mean = new_running_mean_normed * torch.sqrt(new_running_var)
                    # Average running_var in std-dev space — prevents overestimation
                    new_bn.running_var = new_running_var
                    """
                    USE_ARITHMETIC_BN_MERGE = False  #True for Option B
                    new_bn.weight = nn.Parameter(M @ bn_L.weight.data)
                    new_bn.bias = nn.Parameter(M @ bn_L.bias.data)
                    if USE_ARITHMETIC_BN_MERGE:
                        new_bn.running_mean = M @ bn_L.running_mean.data
                        new_bn.running_var = M @ bn_L.running_var.data
                    else:
                        # we cant average the stds because it is a squared quantity
                        # we would overestimate the true merged variance -> so we need to average before stds (inverse stds)
                        inv_stds = 1.0 / torch.sqrt(bn_L.running_var.data + 1e-6)
                        new_running_mean_normed = M @ (bn_L.running_mean.data * inv_stds)
                        new_inv_stds = M @ inv_stds
                        new_running_var = (1.0 / (new_inv_stds + 1e-6)) ** 2
                        new_bn.running_mean = new_running_mean_normed * torch.sqrt(new_running_var)
                        # Average running_var in std-dev space — prevents overestimation
                        new_bn.running_var = new_running_var
                    print(
                        f"      {C['b']}[Debug: BN Fold]{C['res']} BN channels updated to a shape of {C['bold']}{n_folded}{C['res']}")
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
            assert actual_in_channels % n_original == 0, (
                f"[merge_conv_bn input-fold] consumer '{name}' has in_channels={actual_in_channels} "
                f"which is not a multiple of n_original={n_original}"
            )
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
def c2f_layer_folding(c2f_layer, U_input, model, block_name, pairing_rate,fold_c2f_output=False,approximate_repair=False):
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

        # bn_cv1 applies channel-wise to ALL n_total cv1 outputs; slice into top/bottom halves
        # so each cluster sees its own γ/β alongside the conv rows (matches B2 fix above).
        # get the running statistics of the first cv1 block => So the identity path and the bottleneck paths have them
        bn_cv1 = get_module_by_name(model, f"{block_name}.cv1.bn")
        bn_gamma = bn_cv1.weight.data if bn_cv1 is not None else None
        bn_beta = bn_cv1.bias.data if bn_cv1 is not None else None

        #Top half (identity path)
        W_top = cv1.weight.data[:half].reshape(half, -1)
        #Successor is cv2.conv so we use permute(1,0,2,3) to get C_in first so each row = one input channel
        W_cv2_identity = c2f_layer.cv2.conv.weight.data.permute(1, 0, 2, 3)[:half].reshape(half, -1)
        top_parts = [W_top]
        if bn_gamma is not None:
            top_parts.append(bn_gamma[:half].view(half, 1))
            top_parts.append(bn_beta[:half].view(half, 1))
        top_parts.append(W_cv2_identity)
        A_top = torch.cat(top_parts, dim=1).float().cpu().numpy()
        km_top = HKMeans(n_clusters=target_half, random_state=42, n_init=10,n_jobs=-1).fit(A_top)
        for i, lab in enumerate(km_top.labels_):
            U_new[i, lab] = 1.0

        #Bottom half (bottleneck paths)
        W_bot = cv1.weight.data[half:].reshape(half, -1)
        W_bn_input = c2f_layer.m[0].cv1.conv.weight.data.permute(1, 0, 2, 3).reshape(half, -1)
        bot_parts = [W_bot]
        if bn_gamma is not None:
            bot_parts.append(bn_gamma[half:].view(half, 1))
            bot_parts.append(bn_beta[half:].view(half, 1))
        bot_parts.append(W_bn_input)
        A_bot = torch.cat(bot_parts, dim=1).float().cpu().numpy()
        km_bot = HKMeans(n_clusters=target_half, random_state=42, n_init=10,n_jobs=-1).fit(A_bot)
        for i, lab in enumerate(km_bot.labels_):
            U_new[i + half, lab + target_half] = 1.0

        #Fold cv1 -> The input layer before the split layer - where we split into the identity path and the bottleneck layers
        # (bn_cv1 already fetched above for A_top/A_bot construction)
        bn_cv1_name = f"{block_name}.cv1.bn"
        _, bn_f, _ = merge_conv_bn(cv1, bn_cv1, None, U_new, order='output', name=bn_cv1_name,approximate_repair=approximate_repair)
        set_module_by_name(model, bn_cv1_name, bn_f)

        #Fold the Bottleneck Layers -> We use the bottom "half" of the clustering matrice -> We use the same for all bottleneck cv2 layers
        # (since they have an "add" connection to the contact layer) of them.
        U_sliced = U_new[half:, target_half:]
        for i, bottleneck in enumerate(c2f_layer.m):
            conv1_name = f"{block_name}.m.{i}.cv1.conv"
            conv2_name = f"{block_name}.m.{i}.cv2.conv"
            #for cv1 we can compute a new clustering matrice -> since it doesnt go into the concat connection - only cv2 does
            U_cv1 = cumpute_cluster_matrix_u(bottleneck.cv1.conv, bottleneck.cv1.bn, bottleneck.cv2.conv, pairing_rate)
            u_cache[conv1_name] = U_cv1
            u_cache[conv2_name] = U_sliced
            #Adjust the input of the first conv layer inside the bottleneck
            merge_conv_bn(None, None, bottleneck.cv1.conv, U_sliced, order='input', name=conv1_name,approximate_repair=approximate_repair)
            # fold BN1
            bn_b1_name = f"{block_name}.m.{i}.cv1.bn"
            bn_b1 = get_module_by_name(model, bn_b1_name)
            _, b1_f, _ = merge_conv_bn(bottleneck.cv1.conv, bn_b1, None, U_cv1, order='output', name=bn_b1_name,approximate_repair=approximate_repair)
            set_module_by_name(model, bn_b1_name, b1_f)
            # Adjust the input of the second conv layer
            merge_conv_bn(None, None, bottleneck.cv2.conv, U_cv1, order='input', name=conv2_name,approximate_repair=approximate_repair)
            # fold BN2
            bn_b2_name = f"{block_name}.m.{i}.cv2.bn"
            bn_b2 = get_module_by_name(model, bn_b2_name)
            _, b2_f, _ = merge_conv_bn(bottleneck.cv2.conv, bn_b2, None, U_sliced, order='output', name=bn_b2_name,approximate_repair=approximate_repair)
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

        merge_conv_bn(None, None, c2f_layer.cv2.conv, U_cv2, order='input', name=f"{block_name}.cv2.conv",approximate_repair=approximate_repair)

        if fold_c2f_output:
            print(f"   {C['cy']}{C['bold']}[C2F Debug] Folding the 1x1 output projection layer (cv2){C['res']}")

            # 1. Fetch the BN layer for the cv2 output
            bn_cv2_out_name = f"{block_name}.cv2.bn"
            bn_cv2_out = get_module_by_name(model, bn_cv2_out_name)

            # 2. Compute K-Means U Matrix for cv2 (None for conv_next because the main loop handles downstream)
            U_cv2_output = cumpute_cluster_matrix_u(c2f_layer.cv2.conv, bn_cv2_out, None, pairing_rate)

            # 3. Apply the Output Fold
            _, bn_f_out, _ = merge_conv_bn(c2f_layer.cv2.conv, bn_cv2_out, None, U_cv2_output, order='output',
                                           name=bn_cv2_out_name,approximate_repair=approximate_repair)
            if bn_f_out is not None:
                set_module_by_name(model, bn_cv2_out_name, bn_f_out)

            # 4. Save to u_cache so your new FPN Concat logic can find it!
            u_cache[f"{block_name}.cv2.conv"] = U_cv2_output
        else:
            print(
                f"   {C['cy']}{C['dim']}[C2F Debug] Skipping output fold for {block_name}.cv2.conv (C2F Fold was set to FALSE){C['res']}")

    return U_new


def get_module_by_name(model, name):
    for part in name.split('.'):
        model = model[int(part)] if part.isdigit() else getattr(model, part)
    return model


"""This function recalibrates BN running statistics via a forward pass.

Follows the official REPAIR pattern (knowledge/model-folding-universal/core/repair.py:32-50):
EVERY BatchNorm2d in the model is reset and given momentum=None, then a forward pass
over the calibration loader populates the running stats as a simple average.

The `folding_plan` argument is kept for callsite compatibility but is ignored — partial
resets contaminate downstream activations because the un-reset upstream BNs still update
their running stats in train() mode.
"""
def repair_bn_forward_pass(model, loader, device, folding_plan=None, max_samples=1000, verbose=True):
    if folding_plan is not None and verbose:
        print(f"   {C['dim']}[REPAIR] Note: folding_plan arg is ignored; resetting ALL BN layers (official behavior).{C['res']}")
    #get all BN layers from the model
    bn_to_reset = {name: m for name, m in model.named_modules()
              if isinstance(m, nn.BatchNorm2d)}

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
        print(f"   {C['dim']}Resetting {len(bn_to_reset)} BN layers (ALL):{C['res']}")
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


def run_folding_experiment(weights_path, config_path, pairing_rate, number_calib_images, repair_mode,fold_c2f_output,
                           calib_ds="coco/labels/train2017"):
    # load yolo weights
    filename = os.path.basename(config_path)
    config_base_name = os.path.splitext(os.path.basename(config_path))[0]
    print(f"\n{C['bold']}============================================================{C['res']}")
    print(
        f"{C['b']}STARTING RUN: Config: {config_base_name} | PR: {pairing_rate} | Calib N: {number_calib_images} | C2F Out Fold: {fold_c2f_output}{C['res']}")
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
    approx_repair = (repair_mode == "APPROX_REPAIR")
    if approx_repair:
        print(f"\n{C['bold']}{C['cy']}--- Fold-AR: Fusing BN into Conv (pre-folding) ---{C['res']}")
        fuse_all_batchnorms(model)
    initial_params = utils_new.count_parameters(model)
    print(f"Total Parameters: {C['bold']}{initial_params:,}{C['res']}")

    # We save the layers we already folded - so we do not fold them twice (beacuse of the C2f block logic)
    print(f"\n{C['dim']}[Capturing Model Snapshot]{C['res']}")
    shape_snapshot = {
        name: (mod.in_channels if hasattr(mod, 'in_channels') else mod.num_features,
               mod.out_channels if hasattr(mod, 'out_channels') else mod.num_features)
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.BatchNorm2d))
    }
    utils_new.check_layer_shapes(model, folding_plan, shape_snapshot=shape_snapshot, hide_bn=True, internal_name=True,
                                 show_reduction=False)

    # We save the layers we already folded - and also which consumers (connected output layers) - so we do not fold them twice
    processed_layers = set()
    processed_consumers = set()
    print(f"\n{C['bold']}{C['cy']}--- Starting Model Folding Engine ---{C['res']}")
    for module_name, settings in folding_plan.items():
        if not settings.get('do_folding') or module_name in processed_layers or ".bn" in module_name:
            continue
        print(f"\n{C['bold']}Folding Layer: {C['b']}{module_name}{C['res']}")
        try:
            layer_L = module_name
            conv_L = get_module_by_name(model, layer_L)

            # Collect ALL consumers of this folded layer.
            consumers = []
            for next_name, next_settings in folding_plan.items():
                pre_layer = next_settings.get('pre')
                if pre_layer == layer_L or (isinstance(pre_layer, list) and layer_L in pre_layer):
                    consumers.append((next_name, get_module_by_name(model, next_name), pre_layer))

            layer_pr = settings.get('pr')
            pairing_r = float(layer_pr) if layer_pr is not None else float(pairing_rate)
            ref_mapping = settings.get('consistent_map')

            if ref_mapping and ref_mapping in u_cache:
                print(f"   {C['dim']}[Consistent Map]{C['res']} Inheriting U from {ref_mapping}")
                U_matrix = u_cache[ref_mapping]
            else:
                bn_for_cluster = get_module_by_name(model, layer_L.replace(".conv", ".bn"))
                U_matrix = cumpute_cluster_matrix_u(conv_L, bn_for_cluster, None, pairing_r)
                u_cache[layer_L] = U_matrix

            parts = module_name.split('.')
            c2f_block_name = ".".join(parts[:2])
            c2f_candidate = get_module_by_name(model, c2f_block_name)

            if "C2f" in str(type(c2f_candidate)) and "cv1.conv" in module_name:
                # 1. Fold the C2f block
                U_final = c2f_layer_folding(c2f_candidate, U_matrix, model, c2f_block_name, pairing_r, fold_c2f_output,approx_repair)
                u_cache[layer_L] = U_final
                for inner_name in folding_plan.keys():
                    if c2f_block_name in inner_name:
                        processed_layers.add(inner_name)

                # 2. Sync consumers if C2f output was folded
                c2f_out_name = f"{c2f_block_name}.cv2.conv"
                if c2f_out_name in u_cache:
                    for next_name, next_settings in folding_plan.items():
                        pre_layer = next_settings.get('pre')
                        if pre_layer == c2f_out_name or (isinstance(pre_layer, list) and c2f_out_name in pre_layer):
                            if next_name in processed_consumers: continue

                            cmod = get_module_by_name(model, next_name)
                            # Logic for Block-Diagonal Neck Concat
                            if isinstance(pre_layer, list):
                                all_ready = all(
                                    s not in folding_plan or not folding_plan[s].get('do_folding') or s in u_cache for s
                                    in pre_layer)
                                if not all_ready: continue
                                u_blocks = [
                                    u_cache[s] if s in u_cache else torch.eye(get_module_by_name(model, s).out_channels,
                                                                              device=device) for s in pre_layer]
                                merge_conv_bn(None, None, cmod, torch.block_diag(*u_blocks), order="input",
                                              name=next_name,approximate_repair=approx_repair)
                            else:
                                merge_conv_bn(None, None, cmod, u_cache[c2f_out_name], order="input", name=next_name,approximate_repair=approx_repair)
                            processed_consumers.add(next_name)
            else:
                # Standard backbone folding
                bn_name = layer_L.replace(".conv", ".bn")
                bn_L = get_module_by_name(model, bn_name)
                _, bn_folded, _ = merge_conv_bn(conv_L, bn_L, None, U_matrix, order="output", name=bn_name,approximate_repair=approx_repair)
                if bn_folded is not None:
                    set_module_by_name(model, bn_name, bn_folded)
                for cname, cmod, _ in consumers:
                    merge_conv_bn(None, None, cmod, U_matrix, order="input", name=cname,approximate_repair=approx_repair)

            print(f"   {C['g']}Successfully folded {module_name}{C['res']}")

        except Exception as e:
            print(f"   {C['r']}Skipping {module_name} ERROR: {e}{C['res']}")
            continue

    # Final stats and forward pass
    print(f"\n{C['bold']}[AFTER FOLDING]{C['res']}")
    final_params = utils_new.count_parameters(model)
    utils_new.check_layer_shapes(model, folding_plan, shape_snapshot=shape_snapshot, hide_bn=True, internal_name=True,
                                 show_reduction=True)
    reduction = (1 - final_params / initial_params) * 100
    print(
        f"\n{C['bold']}Total Parameters: {final_params:,} ({C['g']}{reduction:.2f}% reduction{C['res']}{C['bold']}){C['res']}")
    utils_new.test_forward_pass(model, device)



    # Save and Repair
    if repair_mode == "APPROX_REPAIR":
        print(f"\n{C['bold']}{C['g']}Fold-AR complete. No forward pass needed.{C['res']}")
        save_model(model, yolo, repair_mode, pairing_rate, config_base_name,weights_path, fold_c2f_output)

    elif repair_mode == "REPAIR":
        full_dataset = utils_new.COCOImageFolder(image_dir=calib_ds, imgsz=640, max_images=None)
        random_indices = random.sample(range(len(full_dataset)), number_calib_images)
        train_loader = DataLoader(Subset(full_dataset, random_indices), batch_size=16, shuffle=True, num_workers=2,
                                  pin_memory=True)
        repair_bn_forward_pass(model, train_loader, device, folding_plan=folding_plan, max_samples=number_calib_images)
        save_model(model, yolo, repair_mode, pairing_rate, config_base_name, fold_c2f_output,weights_path,
                   num_calib_images=number_calib_images)
    else:
        # save the model by stadard before apply REPAIR
        save_model(model, yolo, "NO_REPAIR", pairing_rate, config_base_name,weights_path, fold_c2f_output)

    del model
    del yolo
    torch.cuda.empty_cache()


def main():
    WEIGHTS_PATH = "weights/yolov8/yolov8m/yolov8m.pt"
    CALIB_DS = "coco/images/train2017"
    #if this is None => manual experimen is used
    EXPERIMENTS_FILE = "config_experiments/experiment_5.json"

    # MANUAL PARAMETERS (Used ONLY if EXPERIMENTS_FILE = None !!)
    manual_experiments = [
        {
            "config": "config_folding/yolo_conv4_to_conv8.json",
            "pairing_rates": [0.100000001, 0.200000001, 0.300000001],
            "calib_images": [5000],
            "repair_mode": ["APPROX_REPAIR"],
            "fold_c2f_output": [True]
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

    #double for loop => Grid Search
    for exp in experiments:
        config_file = exp["config"]
        # itertools => Every combination of that list ("dummy" values to prevent an error)
        combinations = itertools.product(
            exp.get("pairing_rates", [0.1]),
            exp.get("calib_images", [1000]),
            exp.get("repair_mode", ["NO_REPAIR"]),
            exp.get("fold_c2f_output", [False])
        )

        for pr, calib_n, rep_mode, fold_c2f in combinations:
            global u_cache
            u_cache = {}

            run_folding_experiment(
                weights_path=WEIGHTS_PATH,
                config_path=config_file,
                pairing_rate=pr,
                number_calib_images=calib_n,
                repair_mode=rep_mode,
                fold_c2f_output=fold_c2f,
                calib_ds=CALIB_DS
            )

if __name__ == "__main__":
    main()