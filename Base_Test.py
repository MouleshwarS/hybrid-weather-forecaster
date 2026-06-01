# Importing the required libraries
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import properscoring as ps

def compute_spatial_acc(preds, targets):
    """
    Computes the Spatial Anomaly Correlation Coefficient (ACC) averaged over time.

    Args:
        preds: Numpy array of shape (N_samples, Height, Width)
        targets: Numpy array of shape (N_samples, Height, Width)

    Returns:
        acc_mean: Scalar float (average spatial ACC)
    """
    climatology = np.mean(targets, axis=0) # Shape: (Height, Width)

    # Compute Anomalies
    pred_anom = preds - climatology
    target_anom = targets - climatology

    # Flatten spatial dims: (N, H, W) -> (N, H*W)
    n_samples = preds.shape[0]
    pred_flat = pred_anom.reshape(n_samples, -1)
    target_flat = target_anom.reshape(n_samples, -1)

    acc_list = []
    for i in range(n_samples):
        p = pred_flat[i]
        t = target_flat[i]

        # Pearson Correlation Formula: cov(p, t) / (std(p) * std(t))
        numerator = np.sum(p * t)
        denominator = np.sqrt(np.sum(p**2)) * np.sqrt(np.sum(t**2))

        if denominator != 0:
            acc_list.append(numerator / denominator)
        else:
            acc_list.append(0.0)

    return np.mean(acc_list)

def evaluate_recursive_forecast_tabular(
    bs_model, unet_model, X_test, y_test, target_scaler,
    feature_names, target_var, seq_len, height, width,
    device, max_horizon, batch_size
):
    var_clean = target_var.lower()
    log_vars = ['tp', 'ws', 'precipitation', 'wind']
    target_is_log = any(v in var_clean for v in log_vars)

    target_idx = feature_names.index(target_var)
    num_eval_samples = len(X_test) - max_horizon

    bs_model.eval().to(device)
    unet_model.eval().to(device)

    results = {
        'horizon': [],
        'rmse_before': [], 'rmse_after': [],
        'acc_before': [], 'acc_after': [],
        'crps_before': [], 'crps_after': []
    }

    predictions_by_horizon = {}

    if torch.is_tensor(X_test):
        current_X_recursive = X_test[0:num_eval_samples].clone()
    else:
        current_X_recursive = X_test[0:num_eval_samples].copy()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for h in range(1, max_horizon + 1):
        # Update recursive window
        if h > 1:
            if torch.is_tensor(current_X_recursive):
                current_X_recursive = torch.roll(current_X_recursive, shifts=-1, dims=1)
                current_X_recursive[:, -1, :, :, target_idx] = torch.as_tensor(predictions_by_horizon[h - 1]).squeeze(-1).to(device)
            else:
                current_X_recursive = np.roll(current_X_recursive, shift=-1, axis=1)
                current_X_recursive[:, -1, :, :, target_idx] = predictions_by_horizon[h - 1].squeeze(-1)

        # Baseline input
        model_name = bs_model.__class__.__name__.lower()
        if 'lstm' in model_name or 'gru' in model_name:
            bs_input = current_X_recursive.reshape(num_eval_samples, seq_len, -1)
        else:
            bs_input = current_X_recursive.reshape(num_eval_samples, -1)

        bs_input = torch.as_tensor(bs_input, dtype=torch.float32, device=device)

        with torch.inference_mode():
            hint_flat = bs_model(bs_input)
            hint_tensor = hint_flat.view(-1, 1, height, width)

        # U-Net input
        last_step = current_X_recursive[:, -1]
        if torch.is_tensor(last_step):
            last_step_tensor = last_step.permute(0, 3, 1, 2).float()
        else:
            last_step_tensor = torch.from_numpy(last_step).permute(0, 3, 1, 2).float().to(device)

        unet_input = torch.cat([last_step_tensor, hint_tensor], dim=1)

        # Batched inference
        preds_mu_list, preds_sigma_list = [], []
        temp_loader = DataLoader(TensorDataset(unet_input), batch_size=batch_size, shuffle=False)

        with torch.inference_mode():
            for (batch_x,) in temp_loader:
                output = unet_model(batch_x)
                mu = output[:, 0:1]
                sigma = torch.exp(
                    0.5 * torch.clamp(output[:, 1:2], -10, 2)
                )
                preds_mu_list.append(mu.cpu())
                preds_sigma_list.append(sigma.cpu())

        predictions_by_horizon[h] = (torch.cat(preds_mu_list, dim=0).permute(0, 2, 3, 1).numpy())

        current_sigmas = (torch.cat(preds_sigma_list, dim=0).permute(0, 2, 3, 1).numpy())

        y_true = y_test[h - 1 : num_eval_samples + h - 1]

        # Inverse scaling
        t_inv = target_scaler.inverse_transform(y_true.reshape(-1, 1))
        h_inv = target_scaler.inverse_transform(hint_tensor.permute(0, 2, 3, 1).cpu().numpy().reshape(-1, 1))
        p_inv = target_scaler.inverse_transform(predictions_by_horizon[h].reshape(-1, 1))

        if target_is_log:
            truth_un = np.expm1(t_inv)
            hint_un = np.expm1(h_inv)
            pred_un = np.expm1(p_inv)
        else:
            truth_un, hint_un, pred_un = t_inv, h_inv, p_inv

        results['horizon'].append(h * 6)
        results['rmse_before'].append(root_mean_squared_error(truth_un, hint_un))
        results['rmse_after'].append(root_mean_squared_error(truth_un, pred_un))

        truth_sq = truth_un.reshape(num_eval_samples, height, width)
        hint_sq = hint_un.reshape(num_eval_samples, height, width)
        pred_sq = pred_un.reshape(num_eval_samples, height, width)

        results['acc_before'].append(compute_spatial_acc(hint_sq, truth_sq))
        results['acc_after'].append(compute_spatial_acc(pred_sq, truth_sq))

        target_range = (1.0 / target_scaler.scale_[0] if hasattr(target_scaler, 'scale_')else 1.0)
        sigma_un = current_sigmas.reshape(-1, 1) * target_range

        results['crps_before'].append(mean_absolute_error(truth_un, hint_un))
        results['crps_after'].append(np.mean(ps.crps_gaussian(truth_un.flatten(), pred_un.flatten(), sigma_un.flatten())))

    end_event.record()
    torch.cuda.synchronize()

    return results, (start_event.elapsed_time(end_event) / 1000)

def evaluate_recursive_forecast_store(bs_model, hybrid_model, X_test, y_test, num_eval_samples, 
                               MAX_HORIZON, SEQ_LEN, height, width, target_var, feature_names,
                               device, BATCH_SIZE_EVAL):
    if target_var not in feature_names:
        raise ValueError(f"Target variable '{target_var}' not found.")
    target_idx = feature_names.index(target_var)

    predictions_by_horizon = {}          
    sigmas_by_horizon = {}               
    base_recursive_by_horizon = {}    
    ground_truth_by_horizon = {}
    
    bs_model.to(device).eval()
    hybrid_model.to(device).eval()

    curr_X_hybrid = X_test[0 : num_eval_samples].copy()
    curr_X_base = X_test[0 : num_eval_samples].copy()

    model_name = bs_model.__class__.__name__.lower()
    is_recurrent = 'lstm' in model_name or 'gru' in model_name

    for h in range(1, MAX_HORIZON + 1):
        # --- Recursive Baseline Path ---
        if is_recurrent:
            # Reshape to (Batch, Seq, Features_per_step) -> (N, 4, 3072)
            bs_input_pure = torch.from_numpy(curr_X_base.reshape(num_eval_samples, SEQ_LEN, -1)).float().to(device)
        else:
            # Reshape to (Batch, Seq * Features) -> (N, 12288)
            bs_input_pure = torch.from_numpy(curr_X_base.reshape(num_eval_samples, -1)).float().to(device)
            
        with torch.inference_mode():
            base_only_flat = bs_model(bs_input_pure)
            base_recursive_by_horizon[h] = base_only_flat.view(-1, height, width, 1).cpu().numpy()

        # --- Recursive Hybrid Path ---
        if is_recurrent:
            hybrid_bs_input = torch.from_numpy(curr_X_hybrid.reshape(num_eval_samples, SEQ_LEN, -1)).float().to(device)
        else:
            hybrid_bs_input = torch.from_numpy(curr_X_hybrid.reshape(num_eval_samples, -1)).float().to(device)
            
        with torch.inference_mode():
            # Get the "Hint" for the U-Net
            hint_tensor = bs_model(hybrid_bs_input).view(-1, 1, height, width)

        # Prepare U-Net Input (Last Step of Hybrid Window + Hint)
        last_step_hybrid = torch.from_numpy(curr_X_hybrid[:, -1, :, :, :]).float().to(device).permute(0, 3, 1, 2)
        unet_input = torch.cat([last_step_hybrid, hint_tensor], dim=1)
        
        preds_mu_list, preds_sigma_list = [], []
        temp_loader = DataLoader(TensorDataset(unet_input), batch_size=BATCH_SIZE_EVAL, shuffle=False)
        
        with torch.inference_mode():
            for (batch_x,) in temp_loader:
                output = hybrid_model(batch_x)
                mu = output[:, 0:1, :, :]
                sigma = torch.exp(0.5 * torch.clamp(output[:, 1:2], -10, 2)) 
                preds_mu_list.append(mu.cpu())
                preds_sigma_list.append(sigma.cpu())
                
        pred_mu_tensor = torch.cat(preds_mu_list, dim=0)
        pred_sigma_tensor = torch.cat(preds_sigma_list, dim=0)
        
        predictions_by_horizon[h] = pred_mu_tensor.permute(0, 2, 3, 1).numpy()
        sigmas_by_horizon[h] = pred_sigma_tensor.permute(0, 2, 3, 1).numpy()
        
        ground_truth_by_horizon[h] = y_test[h-1 : num_eval_samples + h-1]

        # --- Shift windows for next lead time ---
        if h < MAX_HORIZON:
            curr_X_hybrid = np.roll(curr_X_hybrid, shift=-1, axis=1)
            curr_X_hybrid[:, -1, :, :, target_idx] = predictions_by_horizon[h].squeeze(-1)
            
            curr_X_base = np.roll(curr_X_base, shift=-1, axis=1)
            curr_X_base[:, -1, :, :, target_idx] = base_recursive_by_horizon[h].squeeze(-1)

    return predictions_by_horizon, base_recursive_by_horizon, ground_truth_by_horizon, sigmas_by_horizon

def plot_forecast_performance(stan_res, resd_res, attn_res, cbam_res, bs_md, var, seq_len, filepath):
    fig, axes = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
    
    model_colors = {
        'baseline':  '#000000', 
        'standard':  '#d62728',
        'residual':  '#2ca02c',
        'attention': '#ff7f0e',
        'cbam':      '#1f77b4'
    }

    metrics = ['rmse', 'acc', 'crps']

    for i, metric in enumerate(metrics):
        ax = axes[i]
        horizon = stan_res['horizon']
        
        ax.plot(horizon, stan_res[f'{metric}_before'], color=model_colors['baseline'], 
                label=f'{bs_md}', linewidth=2)
        
        ax.plot(horizon, stan_res[f'{metric}_after'], color=model_colors['standard'],
                label='Standard U-Net', linewidth=2)

        ax.plot(horizon, resd_res[f'{metric}_after'], color=model_colors['residual'], 
                label='Residual U-Net', linewidth=2)

        ax.plot(horizon, attn_res[f'{metric}_after'], color=model_colors['attention'],
                label='Attention U-Net', linewidth=2)

        ax.plot(horizon, cbam_res[f'{metric}_after'], color=model_colors['cbam'], 
                label='CBAM U-Net', linewidth=2)
        
        if i == 2:
            ax.set_xlabel('Forecast Horizon (Hours)', weight='bold', size=16)
            
        ax.set_ylabel(metric.upper(), weight='bold', size=16)
        ax.set_xticks(horizon)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if metric == 'acc':
            ax.set_ylim(min(stan_res['acc_before']) - 0.05, 1.0)
            
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='best', frameon=True, prop={'size': 14, 'weight': 'bold'})

    fig.suptitle(f'{bs_md} - U-Net Hybrid Metrics | {var} | {seq_len}-Step Inp Seq', 
                 weight='bold', size=20)

    plt.tight_layout(rect=(0, 0, 1, 0.9975))
    plt.savefig(f'{filepath}', bbox_inches='tight')
    plt.show()

def print_comparison_table(bs_md, seq_len, var, stan_res, resd_res, attn_res, cbam_res):
    table_data = []
    
    headers = [
        "Lead Time", "Metric", f"{bs_md}", 
        "Standard\nU-Net", "Residual\nU-Net", "Attention\nU-Net", "CBAM\nU-Net"
    ]

    for i, h in enumerate(stan_res['horizon']):
        for j, metric_name in enumerate(['rmse', 'acc', 'crps']):
            b_val = stan_res[f'{metric_name}_before'][i]
            
            horizon_label = f"{h}h" if j == 1 else ""
            
            row = [
                horizon_label,
                metric_name.upper(),
                f"{b_val:.4f}",
                f"{stan_res[f'{metric_name}_after'][i]:.4f}",
                f"{resd_res[f'{metric_name}_after'][i]:.4f}",
                f"{attn_res[f'{metric_name}_after'][i]:.4f}",
                f"{cbam_res[f'{metric_name}_after'][i]:.4f}"
            ]
            table_data.append(row)
        
        if i < len(stan_res['horizon']) - 1:
            table_data.append(["--------"] * len(headers))

    print(f"{bs_md} - U-Net Hybrid | Recursive Forecast Comparison | {var} | {seq_len}-Step Input Sequence")
    print(tabulate(table_data, headers=headers, tablefmt="simple_grid", 
                   stralign="center", numalign="center"))