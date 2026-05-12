
"""This is a forward hook to capture per-channel Mean and Variance (Google GEMINI)"""
class StatsHook:
    def __init__(self, layer_name="Unknown"):
        self.layer_name = layer_name
        self.means = []
        self.vars = []

    def __call__(self, module, inp, out):
        with torch.no_grad():
            # Calculate Mean per-channel across Batch, Height, Width → shape [C]
            m = out.mean(dim=(0, 2, 3)).detach()
            # Calculate Variance per-channel (population variance, unbiased=False)
            v = out.var(dim=(0, 2, 3), unbiased=False).detach()
            self.means.append(m)
            self.vars.append(v)

    def get_stats(self):
        """Averages the captured statistics across all batches."""
        if not self.means:
            return None, None
        final_mean = torch.stack(self.means).mean(dim=0)
        final_var = torch.stack(self.vars).mean(dim=0)
        return final_mean, final_var



"""The data driven repair algorithm -> 2.3 Variance collapse and REPAIR"""
def fold_r_data_driven(model, orig_stats, fold_stats, layer_names):
    print(f"{C['bold']}{C['cy']}--- Data driven REPAIR ---{C['res']}")
    for name in layer_names:
        if name not in orig_stats or name not in fold_stats:
            print(f"   {C['r']}Warning: Missing stats for {name}. Skipping.{C['res']}")
            continue
        #Load the mean and var from the original and the folded model for the current layer
        o_mean, o_var = orig_stats[name]['mean'], orig_stats[name]['var']
        f_mean, f_var = fold_stats[name]['mean'], fold_stats[name]['var']

        #Since we scale each channel in the BN layer independently we cant just devide the original stats by the fold stats -> shape missmatch
        #So we average the merged channels variance also -> And this variance value we use for the scaling
        if o_var.shape[0] != f_var.shape[0]:
            if name in u_cache:
                U = u_cache[name]
                M = get_projection_matrix(U)
                o_mean = M @ o_mean
                o_var = M @ o_var
            else:
                print(f"   {C['r']}Error: Shape mismatch and no U matrix found for {name}. Skipping.{C['res']}")
                continue
        #per-channel scaling factor sqrt(sigma_orig_c / sigma_fold_c)
        raw_s_factor = torch.sqrt(o_var + 1e-6) / torch.sqrt(f_var + 1e-6)
        #I dont know if I should clamp here for numerical stability
        #s_factor = torch.clamp(raw_s_factor, min=0.5, max=2.5)
        s_factor = raw_s_factor
        #get the BN layer
        bn_name = name.replace(".conv", ".bn")
        bn_layer = get_module_by_name(model, bn_name)
        # get the values before applying repair
        restored_var = f_var * (s_factor ** 2)
        restored_var_mean = restored_var.mean().item()
        o_var_mean = o_var.mean().item()
        f_var_mean = f_var.mean().item()
        var_diff = ((f_var_mean - o_var_mean) / (o_var_mean + 1e-9)) * 100
        o_mean_val = o_mean.mean().item()
        f_mean_val = f_mean.mean().item()
        mean_drift = f_mean_val - o_mean_val
        orig_gamma_mean = bn_layer.weight.data.mean().item()
        orig_beta_mean = bn_layer.bias.data.mean().item()

        with torch.no_grad():
            #γ_new = γ_fold * s_c (s_c=sigma_o/sigma_fold) -> We scale by the variance collapse factor
            bn_layer.weight.mul_(s_factor.to(bn_layer.weight.device))

            #β_new =(β_fold - μ_fold) * s_c + μ_orig
            #β_fold = current bias after fold
            #μ_fold = folded mean
            #s_c = scale factor (sigma_o/sigma_fold)
            #μ_orig = original mean => We need that so we dont have 0 as our mean => the next layer expects the old mean
            bn_layer.bias = torch.nn.Parameter(
                (bn_layer.bias - f_mean.to(bn_layer.bias.device)) * s_factor.to(bn_layer.bias.device) +
                o_mean.to(bn_layer.bias.device)
            )

        #values after repair
        new_gamma_mean = bn_layer.weight.data.mean().item()
        new_beta_mean = bn_layer.bias.data.mean().item()
        #in %
        gamma_diff = ((new_gamma_mean - orig_gamma_mean) / (abs(orig_gamma_mean) + 1e-8)) * 100
        #user output
        v_color = C['r'] if var_diff < -20 else (C['y'] if var_diff < -5 else C['g'])
        total_channels = s_factor.numel()

        print(f"\n   {C['bold']}{C['b']}[Layer] {name} ({total_channels} channels){C['res']}")

        print(f"      {C['dim']}├─ 1. Post-Fold Activation State{C['res']}")
        print(
            f"      {C['dim']}│  ├─ Variance Collapse: {o_var_mean:.4f} -> {f_var_mean:.4f} [{v_color}{var_diff:+.1f}%{C['dim']}]{C['res']}")
        print(
            f"      {C['dim']}│  └─ Mean Drift:        {o_mean_val:.4f} -> {f_mean_val:.4f} [{mean_drift:+.4f}]{C['res']}")

        print(f"      {C['dim']}└─ 2. Applied REPAIR Recalibration{C['res']}")
        print(
            f"         ├─ Gamma (Scale):     {orig_gamma_mean:.4f} -> {new_gamma_mean:.4f} [{C['cy']}{gamma_diff:+.1f}%{C['dim']}]{C['res']}")
        print(f"         ├─ Beta (Shift):      {orig_beta_mean:.4f} -> {new_beta_mean:.4f}{C['res']}")
        print(
            f"         └─ Restored Variance: {f_var_mean:.4f} * (Scale)² = {C['g']}{restored_var_mean:.4f}{C['res']} {C['dim']}[Target: {o_var_mean:.4f}]{C['res']}")
    print(f"\n{C['g']}{C['bold']}REPAIR Calibration complete.{C['res']}")

"This is just the debug version of fold_r_data_driven"
def fold_r_data_driven_debug(model, orig_stats, fold_stats, layer_names, max_rows=None):
    print(f"{C['bold']}{C['cy']}--- Data driven REPAIR (DEBUG MODE) ---{C['res']}")

    for name in layer_names:
        if name not in orig_stats or name not in fold_stats:
            print(f"   {C['r']}Warning: Missing stats for {name}. Skipping.{C['res']}")
            continue

        o_mean, o_var = orig_stats[name]['mean'], orig_stats[name]['var']
        f_mean, f_var = fold_stats[name]['mean'], fold_stats[name]['var']

        U = u_cache.get(name)

        if o_var.shape[0] != f_var.shape[0]:
            if U is not None:
                M = get_projection_matrix(U)
                o_mean = M @ o_mean
                o_var = M @ o_var
            else:
                print(f"   {C['r']}Error: Shape mismatch and no U matrix found for {name}. Skipping.{C['res']}")
                continue

        s_factor = torch.sqrt(o_var + 1e-6) / torch.sqrt(f_var + 1e-6)

        bn_name = name.replace(".conv", ".bn")
        bn_layer = get_module_by_name(model, bn_name)

        orig_gamma = bn_layer.weight.data.clone()
        orig_beta = bn_layer.bias.data.clone()

        with torch.no_grad():
            bn_layer.weight.mul_(s_factor.to(bn_layer.weight.device))
            bn_layer.bias = torch.nn.Parameter(
                (bn_layer.bias - f_mean.to(bn_layer.bias.device)) * s_factor.to(bn_layer.bias.device) +
                o_mean.to(bn_layer.bias.device)
            )

        new_gamma = bn_layer.weight.data.clone()
        new_beta = bn_layer.bias.data.clone()

        o_var_mean = o_var.mean().item()
        f_var_mean = f_var.mean().item()
        var_diff = ((f_var_mean - o_var_mean) / (o_var_mean + 1e-9)) * 100

        o_mean_val = o_mean.mean().item()
        f_mean_val = f_mean.mean().item()
        mean_drift = f_mean_val - o_mean_val

        orig_gamma_mean = orig_gamma.mean().item()
        new_gamma_mean = new_gamma.mean().item()
        gamma_diff = ((new_gamma_mean - orig_gamma_mean) / (abs(orig_gamma_mean) + 1e-9)) * 100

        orig_beta_mean = orig_beta.mean().item()
        new_beta_mean = new_beta.mean().item()

        restored_var = f_var * (s_factor ** 2)
        restored_var_mean = restored_var.mean().item()

        total_channels = s_factor.numel()
        v_color = C['r'] if var_diff < -20 else (C['y'] if var_diff < -5 else C['g'])
        print(f"\n   {C['bold']}{C['b']}[Layer] {name} ({total_channels} channels){C['res']}")
        print(f"      {C['dim']}├─ 1. Post-Fold Activation State{C['res']}")
        print(
            f"      {C['dim']}│  ├─ Variance Collapse: {o_var_mean:.4f} -> {f_var_mean:.4f} [{v_color}{var_diff:+.1f}%{C['dim']}]{C['res']}")
        print(
            f"      {C['dim']}│  └─ Mean Drift:        {o_mean_val:.4f} -> {f_mean_val:.4f} [{mean_drift:+.4f}]{C['res']}")
        print(f"      {C['dim']}└─ 2. Applied REPAIR Recalibration{C['res']}")
        print(
            f"         ├─ Gamma (Scale):     {orig_gamma_mean:.4f} -> {new_gamma_mean:.4f} [{C['cy']}{gamma_diff:+.1f}%{C['dim']}]{C['res']}")
        print(f"         ├─ Beta (Shift):      {orig_beta_mean:.4f} -> {new_beta_mean:.4f}{C['res']}")
        print(
            f"         └─ Restored Variance: {f_var_mean:.4f} * (Scale)² = {C['g']}{restored_var_mean:.4f}{C['res']} {C['dim']}[Target: {o_var_mean:.4f}]{C['res']}")

        print(f"\n         {C['bold']}{C['cy']}--- Per-Channel Debug Table ---{C['res']}")
        print(
            f"         {C['dim']}{'Ch':>4} | {'Status':>8} | {'Target Var':>10} -> {'Folded Var':>10} ({'Drop %':>7}) | {'Scale (sc)':>10} | {'New Gamma':>10}{C['res']}")
        print(f"         {C['dim']}" + "-" * 84 + C['res'])

        for i in range(total_channels):
            if max_rows is not None and (max_rows // 2 <= i < total_channels - (max_rows // 2)):
                if i == max_rows // 2:
                    print(
                        f"         {C['dim']} ... | {'...':>8} | {'...':>10}    {'...':>10}          | {'...':>10} | {'...':>10}{C['res']}")
                continue

            if U is not None:
                num_merged = int(torch.sum(U[:, i]).item())
                if num_merged > 1:
                    raw_status = f"Fold({num_merged})"
                    status_str = f"{C['y']}{raw_status:>8}{C['dim']}"
                else:
                    status_str = f"{'Single':>8}"
            else:
                status_str = f"{'Not F.':>8}"

            ov = o_var[i].item()
            fv = f_var[i].item()
            vd = ((fv - ov) / (ov + 1e-9)) * 100
            sf = s_factor[i].item()
            ng = new_gamma[i].item()

            c_vd = C['r'] if vd < -20 else (C['y'] if vd < -5 else C['g'])

            print(
                f"         {C['dim']}{i:4d} | {status_str} | {ov:10.4f} -> {fv:10.4f} ({c_vd}{vd:+6.1f}%{C['dim']}) | {sf:10.4f} | {ng:10.4f}{C['res']}")

    print(f"\n{C['g']}{C['bold']}REPAIR Calibration (DEBUG) complete.{C['res']}")



"""
We attack forward hooks to the BN layers to capture the running statistics (mean, variance)
"""
def capture_statistics(model, calib_batch, layer_names):
    if calib_batch is None:
        return {}
    stats_dict = {}
    hooks = {}
    handles = []
    for name in layer_names:
        try:
            #We want the associated BN layer
            bn_name = name.replace(".conv", ".bn")
            module = get_module_by_name(model, bn_name)
            hook = StatsHook(bn_name)
            handle = module.register_forward_hook(hook)
            hooks[name] = hook
            handles.append(handle)
        except Exception as e:
            print(f"      {C['y']}Warning: Could not attach hook to {name}.bn: {e}{C['res']}")

    print(f"   {C['dim']}[Stats Engine] Running forward pass to capture variances...{C['res']}")
    with torch.no_grad():
        model(calib_batch)

    for name, hook in hooks.items():
        mean, var = hook.get_stats()
        stats_dict[name] = {'mean': mean, 'var': var}

    for handle in handles:
        handle.remove()

    return stats_dict
