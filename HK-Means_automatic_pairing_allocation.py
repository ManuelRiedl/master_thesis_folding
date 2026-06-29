import os
import json
import torch
from ultralytics import YOLO
from hkmeans import HKMeans

# ANSI colors for beautiful console output
C = {
    'b': '\033[94m', 'cy': '\033[96m', 'g': '\033[92m', 'y': '\033[93m',
    'r': '\033[91m', 'bold': '\033[1m', 'dim': '\033[2m', 'res': '\033[0m',
    'magenta': '\033[95m'
}


def get_module_by_name(model, name):
    for part in name.split('.'):
        if part.isdigit():
            model = model[int(part)]
        else:
            model = getattr(model, part)
    return model


def extract_A_matrix(W_out, W_in_next):
    n_channels = W_out.shape[0]
    W_l = W_out.reshape(n_channels, -1)
    W_l_next = W_in_next.permute(1, 0, 2, 3).reshape(n_channels, -1)

    A = torch.cat([W_l, W_l_next], dim=1).float().cpu().numpy()
    return A


def  calculate_hkmeans_error(A_matrix, pr):
    n_original = A_matrix.shape[0]

    #how many channels we want to keep
    k_folded = max(1, round(n_original * (1 - pr)))
    #zero devision safeguard
    if k_folded >= n_original:
        return 0.0
    #Run HKMeans
    km = HKMeans(n_clusters=k_folded, random_state=42, n_init=5, n_jobs=-1)
    km.fit(A_matrix)

    #normalized error per channel -> how much information we have lost because of the grouping
    raw_inertia = km.inertia_ if hasattr(km, 'inertia_') else 1.0
    normalized_error = raw_inertia / n_original
    return normalized_error


def get_group_param_count(model, name, data):
    params = 0
    #calc how many params exist in a folding block
    if data["type"] == "isolated" or data["type"] == "standard_conv":
        #normal conv
        block = get_module_by_name(model, name.replace(".conv", ""))
        k_size = block.conv.kernel_size[0] * block.conv.kernel_size[1]
        params += (block.conv.out_channels * block.conv.in_channels * k_size) * 2

    elif data["type"] == "shared_group":
        #c2f block
        c2f = get_module_by_name(model, data["c2f_parent"])
        half_channels = c2f.cv1.conv.out_channels // 2
        mult = 2 + (len(c2f.m) * 2)
        params += (half_channels * 9 * half_channels) * mult

    return params


# =============================================================================
# PHASE-1: Bruteforce HKMeans
# calcs the normalized error (information loss) for different pairing rates of HKmeans.
# =============================================================================
def bruteforce_hkmeans(model, blocks_to_scan, cache_dir, cache_filename):
    print(f"\n{C['bold']}{C['magenta']}=== PHASE 1: LAYER-WISE HK-MEANS Bruteforce ==={C['res']}")
    os.makedirs(cache_dir, exist_ok=True)
    full_cache_path = os.path.join(cache_dir, cache_filename)
    #check if for this configuration the results are already saved
    if os.path.exists(full_cache_path):
        print(f"   {C['g']}[CACHE HIT]{C['res']} Profile found. Loading instantly...")
        with open(full_cache_path, 'r') as f:
            return json.load(f)
    print(f"   {C['y']}[CACHE MISS]{C['res']} Recalculating HKMeans...")

    #brutforce steps -> Calculation of HK-means for each of these pairing rates
    profile_data = {}
    pr_steps = [0.035, 0.065, 0.1, 0.135, 0.165, 0.2, 0.235, 0.265, 0.3, 0.335, 0.365, 0.4, 0.435, 0.465, 0.5]

    with torch.no_grad():
        for idx, block_name in enumerate(blocks_to_scan):
            try:
                block = get_module_by_name(model, block_name)
                block_type_str = str(type(block))
                #standard conv to conv connection - only one consumer
                #the following code is mainly copied from folding_main.py
                if "Conv" in block_type_str and "C2f" not in block_type_str:
                    name = f"{block_name}.conv"
                    print(f"   {C['dim']}Standard Conv: {name} ...{C['res']}", end="\r")
                    if idx + 1 < len(blocks_to_scan):
                        next_block_name = blocks_to_scan[idx + 1]
                        next_block = get_module_by_name(model, next_block_name)
                        #out weights
                        W_out = block.conv.weight.data
                        n_ch = W_out.shape[0]
                        #BN stats
                        try:
                            bn_layer = get_module_by_name(model, f"{block_name}.bn")
                            bn_gamma = bn_layer.weight.data.view(-1, 1)
                            bn_beta = bn_layer.bias.data.view(-1, 1)
                        except Exception:
                            bn_gamma, bn_beta = None, None
                        #consumer weights
                        if "C2f" in str(type(next_block)):
                            W_in_next = next_block.cv1.conv.weight.data
                        else:
                            W_in_next = next_block.conv.weight.data
                        #construction of the constrained A-Matrix
                        parts = [W_out.reshape(n_ch, -1)]
                        if bn_gamma is not None:
                            parts.append(bn_gamma)
                            parts.append(bn_beta)
                        parts.append(W_in_next.permute(1, 0, 2, 3).reshape(n_ch, -1))
                        A = torch.cat(parts, dim=1).float().cpu().numpy()

                        #accross all pairing rates calc hkmeans
                        errors = {}
                        for pr in pr_steps:
                            print(
                                f"   {C['dim']}Block: {name} | Calculating PR: {pr * 100:.1f}% ...{C['res']}".ljust(
                                    80), end="\r")
                            errors[str(pr)] = calculate_hkmeans_error(A, pr)

                        profile_data[name] = {"type": "standard_conv", "errors": errors}
                #second case is the c2f block
                elif "C2f" in block_type_str:
                    cv1 = block.cv1.conv
                    cv1_weight = cv1.weight.data
                    #this is the Split -> Half goes to the bottlenecks and half goes to the Identity path
                    n_total = cv1_weight.shape[0]
                    half = n_total // 2
                    #first c2f-block cv1 BN layer gamma and beta
                    try:
                        bn_cv1 = get_module_by_name(model, f"{block_name}.cv1.bn")
                        bn_gamma = bn_cv1.weight.data if bn_cv1 is not None else None
                        bn_beta = bn_cv1.bias.data if bn_cv1 is not None else None
                    except Exception:
                        bn_gamma, bn_beta = None, None

                    #loop for the bottlenecks inside the c2f-block eg:m.0, m.1 - these all have their own U-matrice
                    # since m.0.cv1 is only connected to m.0.cv2 -> cv1 is isolated and has its own matrice
                    #logic is the same as in the folding_main.py
                    for i, bottleneck in enumerate(block.m):
                        name = f"{block_name}.m.{i}.cv1.conv"
                        print(f"   {C['dim']}Bottleneck Conv: {name} ...{C['res']}", end="\r")

                        W_cv1_bot = bottleneck.cv1.conv.weight.data
                        W_cv2_bot = bottleneck.cv2.conv.weight.data
                        n_bot_ch = W_cv1_bot.shape[0]
                        try:
                            bn_bot = get_module_by_name(model, f"{block_name}.m.{i}.cv1.bn")
                            bot_gamma = bn_bot.weight.data.view(-1, 1)
                            bot_beta = bn_bot.bias.data.view(-1, 1)
                        except Exception:
                            bot_gamma, bot_beta = None, None
                        #matrice eg: (m.0.cv1 , BN, m.0.cv2)
                        iso_parts = [W_cv1_bot.reshape(n_bot_ch, -1)]
                        if bot_gamma is not None:
                            iso_parts.append(bot_gamma)
                            iso_parts.append(bot_beta)
                        iso_parts.append(W_cv2_bot.permute(1, 0, 2, 3).reshape(n_bot_ch, -1))
                        A_iso = torch.cat(iso_parts, dim=1).float().cpu().numpy()
                        errors = {}
                        for pr in pr_steps:
                            print(
                                f"   {C['dim']}Block: {name} | Calculating PR: {pr * 100:.1f}% ...{C['res']}".ljust(
                                    80), end="\r")
                            errors[str(pr)] = calculate_hkmeans_error(A_iso, pr)

                        profile_data[name] = {"type": "isolated", "c2f_parent": block_name, "errors": errors}

                    # Here we have to reuse the matrice from the Identity Split at the top of the c2f block
                    # in the Folding main we calculate the shared U-matrix only by looking at the top cv block before the split
                    name = f"{block_name}_Shared_Group"
                    print(f"   {C['dim']}Shared Group: {name} ...{C['res']}", end="\r")

                    #A_TOP -> Identity Path -> top half of cv1 and top half of cv2 -> This is the connection path
                    W_top = cv1_weight[:half].reshape(half, -1)
                    W_cv2_identity = block.cv2.conv.weight.data.permute(1, 0, 2, 3)[:half].reshape(half, -1)

                    top_parts = [W_top]
                    if bn_gamma is not None:
                        top_parts.append(bn_gamma[:half].view(half, 1))
                        top_parts.append(bn_beta[:half].view(half, 1))
                    top_parts.append(W_cv2_identity)
                    A_top = torch.cat(top_parts, dim=1).float().cpu().numpy()

                    #A_BOT -> Bottleneck Path -> bottom half of cv1 (The half that goes into the bottlenecks)
                    #W_bn_input is the first bottleneck block m.0.cv1 -> We have to use this matrice for the other bottlenecks too
                    W_bot = cv1_weight[half:].reshape(half, -1)
                    W_bn_input = block.m[0].cv1.conv.weight.data.permute(1, 0, 2, 3).reshape(half, -1)

                    bot_parts = [W_bot]
                    if bn_gamma is not None:
                        bot_parts.append(bn_gamma[half:].view(half, 1))
                        bot_parts.append(bn_beta[half:].view(half, 1))
                    bot_parts.append(W_bn_input)
                    A_bot = torch.cat(bot_parts, dim=1).float().cpu().numpy()

                    #Because A_top and A_bot are part of the same layer cv1 they also must be folded by the same percentage.
                    #because of that we average the error -> so we have one "error" score
                    errors = {}
                    for pr in pr_steps:
                        print(
                            f"   {C['dim']}Shared Group: {name} | Calculating PR: {pr * 100:.1f}% ...{C['res']}".ljust(
                                80), end="\r")
                        err_top = calculate_hkmeans_error(A_top, pr)
                        err_bot = calculate_hkmeans_error(A_bot, pr)
                        errors[str(pr)] = (err_top + err_bot) / 2.0

                    profile_data[name] = {"type": "shared_group", "c2f_parent": block_name, "errors": errors}

            except Exception as e:
                print(f"   {C['r']}Error: {block_name}: {e}{C['res']}")

    #save the calculated scores
    print(f"   {C['g']}[SUCCESS]{C['res']} Bruteforce HKmeans complete! Saving to cache.")
    with open(full_cache_path, 'w') as f:
        json.dump(profile_data, f, indent=4)

    return profile_data


# =============================================================================
# PHASE-2:  ALLOCATION
# here we allocate pairing ratio for the individual layers.
# So layers can be folded with the least error
# =============================================================================
def allocate_budget(model, profile_data, target_budget, config_template):
    print(f"\n{C['bold']}{C['cy']}=== PHASE 2: GLOBAL BUDGET ALLOCATION ==={C['res']}")

    #calculate how many parameters are there in total -> so we know how many params we want to save to hit the target_budget
    scoped_total_params = 0
    for name, data in profile_data.items():
        scoped_total_params += get_group_param_count(model, name, data)

    #convert percentage targets (0.2) into absolute parameter targets
    if isinstance(target_budget, str) and target_budget.endswith('%'):
        ratio = float(target_budget.replace('%', '')) / 100.0
        target_reduction_count = int(scoped_total_params * ratio)
    else:
        val = float(target_budget)
        if val <= 1.0:
            target_reduction_count = int(scoped_total_params * val)
        else:
            target_reduction_count = int(val)

    if target_reduction_count > scoped_total_params:
        print(f"   {C['r']}ERROR: Requested reduction exceeds available parameters!{C['res']}")
        return {}, target_reduction_count

    #init small error tolerance
    error_ceiling = 0.0001
    #step size of the error tolerance
    step_ceiling = 0.0005
    final_allocations = {}

    #stepwise raise the error tolerance until the parameter reduction target is met
    while True:
        params_removed = 0
        current_allocations = {}

        #step through the calculated error rates per layer
        for name, data in profile_data.items():
            # check if a layer is not allowed to fold
            is_allowed = True
            if data["type"] == "standard_conv" or data["type"] == "isolated":
                if name in config_template and config_template[name].get("do_folding", True) == False:
                    is_allowed = False
            elif data["type"] == "shared_group":
                cv1_key = f"{data['c2f_parent']}.cv1.conv"
                if cv1_key in config_template and config_template[cv1_key].get("do_folding", True) == False:
                    is_allowed = False

            #find the highest folding ratio that stays UNDER the current error ceiling
            best_pr = 0.0
            if is_allowed:
                for pr_str, error_val in data["errors"].items():
                    if error_val <= error_ceiling:
                        best_pr = float(pr_str)

            current_allocations[name] = best_pr

            #calc how many parameters this folding ratio saves
            if best_pr > 0:
                if data["type"] == "isolated" or data["type"] == "standard_conv":
                    block = get_module_by_name(model, name.replace(".conv", ""))
                    dropped = int(block.conv.out_channels * best_pr)
                    k_size = block.conv.kernel_size[0] * block.conv.kernel_size[1]
                    params_removed += (dropped * block.conv.in_channels * k_size) * 2

                elif data["type"] == "shared_group":
                    c2f = get_module_by_name(model, data["c2f_parent"])
                    half_channels = c2f.cv1.conv.out_channels // 2
                    dropped = int(half_channels * best_pr)
                    mult = 2 + (len(c2f.m) * 2)
                    params_removed += (dropped * 9 * half_channels) * mult

        #stop we hit our target
        if params_removed >= target_reduction_count:
            final_allocations = current_allocations
            break

        #if we did not remove enough params -> raise error ceiling
        error_ceiling += step_ceiling

    print(f"   Locked Error Ceiling at: {error_ceiling:.4f}")
    print(f"   {C['g']}Allocation complete! Removed: {params_removed:,} params{C['res']}")
    return final_allocations, target_reduction_count



def get_sorting_value(dictionary_item):
    name = dictionary_item[0]
    pr = dictionary_item[1]
    return (pr, name)


def print_allocation_summary(allocations, profile_data):
    print(f"\n{C['bold']}{C['b']}=== FINAL FOLDING PLAN SUMMARY ==={C['res']}")
    print(f"{C['dim']}{'-' * 75}{C['res']}")
    print(f"{C['bold']}{'Layer / Group Name'.ljust(45)} | {'Type'.ljust(15)} | {'Assigned PR'}{C['res']}")
    print(f"{C['dim']}{'-' * 75}{C['res']}")
    sorted_allocations = sorted(allocations.items(), key=get_sorting_value, reverse=True)

    for name, pr in sorted_allocations:
        g_type = profile_data[name]["type"]
        color = C['g'] if pr > 0 else C['dim']
        pr_display = f"{pr * 100:.1f}%" if pr > 0 else "0.0%"
        display_name = name.replace("_Shared_Group", " [Shared Skips]")
        print(f"{color}{display_name.ljust(45)} | {g_type.ljust(15)} | {pr_display}{C['res']}")

    print(f"{C['dim']}{'-' * 75}{C['res']}\n")


# =============================================================================
# PHASE-3: Generate configuration
# converts the raw folding rations into a folding plan which can be interpreted by folding_main
# =============================================================================
def generate_json_plan_for_engine(allocations, config_template, anchor_pr=None):
    folding_plan = {}

    folding_plan["__metadata__"] = {
        "anchor_pairing_rate": anchor_pr,
        "is_automatic": True
    }

    for layer_name, layer_config in config_template.items():
        alloc_key = layer_name

        #map internal engine names back to the allocator group names
        if ".m." in layer_name and "cv2" in layer_name:
            block_parent = layer_name.split('.m.')[0]
            alloc_key = f"{block_parent}_Shared_Group"
        elif ".cv1.conv" in layer_name and ".m." not in layer_name:
            block_parent = layer_name.split('.cv1')[0]
            alloc_key = f"{block_parent}_Shared_Group"
        elif ".cv2.conv" in layer_name and ".m." not in layer_name:
            block_parent = layer_name.split('.cv2')[0]
            alloc_key = f"{block_parent}_Shared_Group"

        assigned_pr = allocations.get(alloc_key, 0.0)
        allowed_to_fold = layer_config.get("do_folding", True)

        #only fold if the allocator assigned a ratio > 0 and it is not protected (do_fold = False)
        do_fold = allowed_to_fold and (assigned_pr > 0)

        folding_plan[layer_name] = {
            "pre": layer_config.get("pre"),
            "do_folding": do_fold,
            "pr": round(assigned_pr, 3) if do_fold else 0.0,
            "consistent_map": layer_config.get("consistent_map")
        }

        if "num_channels" in layer_config:
            folding_plan[layer_name]["num_channels"] = layer_config["num_channels"]

    return folding_plan


def generate_auto_plan(real_model, config_filepath, target_reduction, output_dir, anchor_pr,
                       cache_dir="results_save/h_kMeans_results"):
    print(f"\n{C['bold']}{C['b']}================================================={C['res']}")
    print(
        f"{C['bold']}{C['b']} ALLOCATOR RUN: {config_filepath} | Target Reduction: {target_reduction} params{C['res']}")
    print(f"{C['bold']}{C['b']}================================================={C['res']}")

    if not os.path.exists(config_filepath):
        print(f"{C['r']}Error: Cannot find {config_filepath}. Skipping.{C['res']}")
        return None

    with open(config_filepath, 'r') as f:
        config_template = json.load(f)

    #filename for the cached hkmean
    config_filename = os.path.basename(config_filepath)
    config_stem = os.path.splitext(config_filename)[0]
    dynamic_cache_filename = f"cache_h_kmean_{config_stem}.json"

    #get the block which we can scan from the config
    scanned_blocks = set()
    for key in config_template.keys():
        parts = key.split('.')
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            scanned_blocks.add(f"model.{parts[1]}")

    #sort strings numerically by block index
    block_tuples = []
    for block in scanned_blocks:
        block_index = int(block.split('.')[1])
        block_tuples.append((block_index, block))

    #extract the names
    block_tuples.sort()
    backbone_blocks = []
    for item in block_tuples:
        backbone_blocks.append(item[1])

    #1. Step - Run H-Kmeans for different configurations
    profiles = bruteforce_hkmeans(
        model=real_model,
        blocks_to_scan=backbone_blocks,
        cache_dir=cache_dir,
        cache_filename=dynamic_cache_filename
    )

    #2. Step - Allocate pairing rates
    allocations, calculated_count = allocate_budget(
        model=real_model,
        profile_data=profiles,
        target_budget=target_reduction,
        config_template=config_template
    )

    #3. Step - output
    print_allocation_summary(allocations, profiles)
    final_plan = generate_json_plan_for_engine(allocations, config_template, anchor_pr=anchor_pr)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{config_stem}_auto_target_{float(anchor_pr):.1f}_{int(calculated_count)}.json"
    output_plan_file = os.path.join(output_dir, filename)

    with open(output_plan_file, 'w') as f:
        json.dump(final_plan, f, indent=4)

    print(f"{C['g']}Successfully saved engine-compatible plan to {output_plan_file}!\n{C['res']}")
    return output_plan_file



if __name__ == "__main__":
    print(f"{C['cy']}Starting Auto Allocator Batch Execution...{C['res']}")

    #load models
    model_path = "weights/yolov8/yolov8m/yolov8m.pt"
    if os.path.exists(model_path):
        yolo_model = YOLO(model_path).model
        print(f"{C['g']}Successfully loaded model weights from {model_path}{C['res']}")
    else:
        print(f"{C['y']}Error: {model_path} not found.{C['res']}")
        exit(-1)

    #configuration file
    example_config = "config_folding/yolov8_m/yolov8_medium_full_architecture_protected.json"
    output_directory = "config_folding/auto_plan"
    #pairing ratios we want to target -> each of them is a seperate config file eg:yolov8_medium_conv4_to_conv8_0.025.json
    target_ratios = [0.025,0.05, 0.075,0.10,0.125, 0.15, 0.20, 0.225, 0.25]
    if os.path.exists(example_config):
        for pr in target_ratios:
            #0.1 => 10%
            target_budget_str = f"{int(pr * 100)}%"
            print(f"\n{C['bold']}{C['magenta']}================================================={C['res']}")
            print(f"{C['bold']}{C['magenta']} GENERATING PLAN FOR: PR {pr} ({target_budget_str} Reduction) {C['res']}")
            print(f"{C['bold']}{C['magenta']}================================================={C['res']}")

            generate_auto_plan(
                real_model=yolo_model,
                config_filepath=example_config,
                target_reduction=target_budget_str,
                output_dir=output_directory,
                anchor_pr=pr
            )
    else:
        print(
            f"{C['r']}Error: Example config '{example_config}' not found. Please provide a valid template path.{C['res']}")