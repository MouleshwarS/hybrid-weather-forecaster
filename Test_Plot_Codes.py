# Importing the required libraries
import numpy as np
from sqlalchemy import label
import torch
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns
import properscoring as ps
import scipy.stats as stats
from datetime import timedelta

sns.set_theme(style="whitegrid")

# Base Functions
def predict_baseline(model, x_data):
    '''
    Internal helper to unify inference across different architecture types.
    Handles Scikit-learn, PyTorch MLPs (ElasticNet/NN), and RNNs (LSTM/GRU).

    Args:
        model: Trained model (sklearn or torch.nn.Module).
        x_data (np.ndarray): Input features (Batch, Seq, H, W, C).

    Returns:
        np.ndarray: Predictions in scaled space (Batch, H*W).
    '''
    if hasattr(model, 'predict') and not isinstance(model, torch.nn.Module):
        x_flat = x_data.reshape(len(x_data), -1)
        return model.predict(x_flat)

    # Handle PyTorch models
    model.eval()
    device = next(model.parameters()).device

    with torch.inference_mode():
        batch_size = x_data.shape[0]
        seq_len = x_data.shape[1]

        # Check model type to determine flattening strategy
        # RNNs usually have 'LSTM' or 'GRU' in their class name
        model_class_name = model.__class__.__name__.upper()

        if any(name in model_class_name for name in ['LSTM', 'GRU', 'RNN']):
            # RNNs expect (Batch, Seq, Features_per_step)
            # Features_per_step = H * W * C
            x_reshaped = x_data.reshape(batch_size, seq_len, -1)
        else:
            # MLPs/ElasticNet expect (Batch, Total_Flattened_Features)
            # Total = Seq * H * W * C
            x_reshaped = x_data.reshape(batch_size, -1)

        x_tensor = torch.from_numpy(x_reshaped).float().to(device)
        preds = model(x_tensor)

        return preds.detach().cpu().numpy()


# Spatial Distribution of Metrics (RMSE, ACC, CRPS)
def extract_and_scale_data(h_idx, predictions_by_horizon, base_recursive_by_horizon, 
                            ground_truth_by_horizon, sigmas_by_horizon, 
                            height, width, var_name, target_scaler):
    """Helper function to extract and invert-scale data for a specific horizon."""
    pf_raw = predictions_by_horizon[h_idx]    
    pb_raw = base_recursive_by_horizon[h_idx] 
    truth_raw = ground_truth_by_horizon[h_idx]
    sigmas_raw = sigmas_by_horizon[h_idx]

    is_log_var = any(v in var_name.lower() for v in ['tp', 'ws', 'precipitation', 'wind'])
    target_range = 1.0 / target_scaler.scale_[0] if hasattr(target_scaler, 'scale_') else 1.0

    def safe_inverse(data):
        unscaled = target_scaler.inverse_transform(data.reshape(-1, 1))
        if is_log_var:
            return np.expm1(unscaled)
        return unscaled

    num_samples = len(truth_raw)
    pb_real = safe_inverse(pb_raw).reshape(num_samples, height, width)
    pf_real = safe_inverse(pf_raw).reshape(num_samples, height, width)
    t_real  = safe_inverse(truth_raw).reshape(num_samples, height, width)
    
    sf_real = (sigmas_raw.reshape(num_samples, height, width) * target_range) + 1e-7
    
    return pb_real, pf_real, t_real, sf_real


def plot_grid_metric(grid_data, label, cmap, meta, height, width, 
                      bs_md, unet_type, vn, seq_len, filepath):
    """Helper function to plot an N x 2 grid with separate, row-wise colorbars."""
    num_rows = len(grid_data)
    
    fig, axes = plt.subplots(
        num_rows, 2, 
        figsize=(18, 9 * num_rows), 
        layout='constrained', 
        gridspec_kw={'wspace': 0.02, 'hspace': 0.02}
    )

    if num_rows == 1:
        axes = axes.reshape(1, 2)

    unique_lats = np.linspace(meta['lat_range'][1], meta['lat_range'][0], height)
    unique_lons = np.linspace(meta['lon_range'][0], meta['lon_range'][1], width)

    y_labels = [f'{y:.2f}' for y in unique_lats]
    x_labels = [f'{x:.2f}' for x in unique_lons]

    for i, (data_b, data_f, h_idx) in enumerate(grid_data):
        
        # Calculate min/max bounds strictly for the current ROW (h_idx)
        v_min = min(data_b.min(), data_f.min())
        v_max = max(data_b.max(), data_f.max())
        
        if 'ACC' in label: 
            v_min, v_max = 0, 1

        h_args = {
            'cmap': cmap, 'annot': True, 'fmt': '.3f',
            'annot_kws': {'weight': 'bold', 'size': 13},
            'vmin': v_min, 'vmax': v_max,
            'linecolor': '#000000', 'linewidths': 0.5,
            'xticklabels': x_labels,
            'yticklabels': y_labels,
        }

        # Baseline Plot (Column 0)
        sns.heatmap(data_b, ax=axes[i, 0], cbar=False, **h_args)
        axes[i, 0].set_title(f'T+{h_idx*6}h | {label} | {bs_md} | {vn} | {seq_len}-Step Inp Seq', weight='bold', size=20)

        # Hybrid Plot (Column 1)
        sns.heatmap(data_f, ax=axes[i, 1], cbar=False, **h_args) 
        axes[i, 1].set_title(f'T+{h_idx*6}h | {label} | {unet_type} U-Net | {vn} | {seq_len}-Step Inp Seq', weight='bold', size=20)
        
        for j in range(2):
            axes[i, j].tick_params(axis='both', which='major', labelsize=14)
            
            if j == 0:
                axes[i, j].set_ylabel('Latitude (°N)', weight='bold', size=16)
            else:
                axes[i, j].set_ylabel('')
                axes[i, j].set_yticklabels([])

            if i == num_rows - 1:
                axes[i, j].set_xlabel('Longitude (°E)', weight='bold', size=16)
            else:
                axes[i, j].set_xticklabels([])
                axes[i, j].set_xlabel('')
                axes[i, j].tick_params(axis='x', which='both', bottom=False, size=14)

        mappable = axes[i, 1].collections[0]
        cbar = fig.colorbar(mappable, ax=[axes[i, 0], axes[i, 1]], pad=0.02)
        cbar.set_label(label, weight='bold', size=16)
        cbar.ax.tick_params(labelsize=14)

    plt.savefig(filepath, bbox_inches='tight')
    plt.show()


def plot_rmse_comparison(h_indices, seq_len, predictions_by_horizon, base_recursive_by_horizon, 
                         ground_truth_by_horizon, sigmas_by_horizon, meta, height, width, 
                         bs_md, unet_type, vn, var_name, target_scaler, filepath):
    
    grid_data = []
    for h_idx in h_indices:
        pb, pf, t, _ = extract_and_scale_data(
            h_idx, predictions_by_horizon, base_recursive_by_horizon, 
            ground_truth_by_horizon, sigmas_by_horizon, 
            height, width, var_name, target_scaler
        )
        rmse_base = np.sqrt(np.mean((pb - t)**2, axis=0))
        rmse_final = np.sqrt(np.mean((pf - t)**2, axis=0))
        grid_data.append((rmse_base, rmse_final, h_idx))
        
    plot_grid_metric(grid_data, 'RMSE', 'Oranges', meta, height, width, 
                      bs_md, unet_type, vn, seq_len, filepath)


def plot_acc_comparison(h_indices, seq_len, predictions_by_horizon, base_recursive_by_horizon, 
                        ground_truth_by_horizon, sigmas_by_horizon, meta, height, width, 
                        bs_md, unet_type, vn, var_name, target_scaler, filepath):
    
    def get_acc_grid(p, t):
        climatology = np.mean(t, axis=0) 
        p_anom, t_anom = p - climatology, t - climatology
        num = np.sum(p_anom * t_anom, axis=0)
        den = np.sqrt(np.sum(p_anom**2, axis=0) * np.sum(t_anom**2, axis=0))
        return np.divide(num, den, out=np.zeros_like(num), where=den!=0)

    grid_data = []
    for h_idx in h_indices:
        pb, pf, t, _ = extract_and_scale_data(
            h_idx, predictions_by_horizon, base_recursive_by_horizon, 
            ground_truth_by_horizon, sigmas_by_horizon, 
            height, width, var_name, target_scaler
        )
        acc_base = get_acc_grid(pb, t)
        acc_final = get_acc_grid(pf, t)
        grid_data.append((acc_base, acc_final, h_idx))
        
    plot_grid_metric(grid_data, 'ACC', 'Greens', meta, height, width, 
                      bs_md, unet_type, vn, seq_len, filepath)


def plot_crps_comparison(h_indices, seq_len, predictions_by_horizon, base_recursive_by_horizon, 
                         ground_truth_by_horizon, sigmas_by_horizon, meta, height, width, 
                         bs_md, unet_type, vn, var_name, target_scaler, filepath):
    
    grid_data = []
    for h_idx in h_indices:
        pb, pf, t, sf = extract_and_scale_data(
            h_idx, predictions_by_horizon, base_recursive_by_horizon, 
            ground_truth_by_horizon, sigmas_by_horizon, 
            height, width, var_name, target_scaler
        )
        crps_base = np.mean(np.abs(pb - t), axis=0)
        crps_final = np.mean(ps.crps_gaussian(t, pf, sf), axis=0)
        grid_data.append((crps_base, crps_final, h_idx))
        
    plot_grid_metric(grid_data, 'CRPS', 'Blues', meta, height, width, 
                      bs_md, unet_type, vn, seq_len, filepath)


# Monthly Performance Grids (RMSE, ACC, CRPS)
def plot_monthly_metric_grids_base(metric_name, cmap, predictions_by_horizon, ground_truth_by_horizon, 
                                    X_test, baseline_model, target_scaler, num_eval_samples, max_horizon, 
                                    start_dt, bs_md, unet_type, seq_len, var_name, filepath, 
                                    sigmas_by_horizon=None):
    """
    Core helper function to generate Monthly vs. Lead-time heatmaps for various metrics.
    """
    test_dates = np.array([start_dt + timedelta(hours=int(i + seq_len) * 6) for i in range(num_eval_samples)])
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    years = sorted(list(set(d.year for d in test_dates)))
    horizon_labels = [f'T+{h*6}h' for h in range(1, max_horizon + 1)]

    is_log_var = var_name.lower() in ['tp', 'total precipitation', 'ws', 'wind speed']
    
    if metric_name == 'CRPS':
        target_range = 1.0 / target_scaler.scale_[0] if hasattr(target_scaler, 'scale_') else target_scaler.data_range_[0]

    fig_main, axes = plt.subplots(
        nrows=len(years), 
        ncols=3, 
        figsize=(18, 6 * len(years)), 
        layout='constrained',
        squeeze=False, 
        gridspec_kw={
            'width_ratios': [1, 1, 0.03],
            'wspace': 0.05, 
            'hspace': 0.05
        }
    )

    plt.suptitle(f'Monthly {metric_name} Performance Analysis: {var_name} | {seq_len}-Step Input Sequence', 
                 fontsize=20, weight='bold')

    def calc_acc(p, t):
        climatology = np.mean(t, axis=0) 
        p_anom, t_anom = p - climatology, t - climatology
        num = np.sum(p_anom * t_anom)
        den = np.sqrt(np.sum(p_anom**2) * np.sum(t_anom**2))
        return num / den if den != 0 else 0

    for idx, year in enumerate(years):
        base_mat = np.full((12, max_horizon), np.nan)
        hyb_mat = np.full((12, max_horizon), np.nan)

        for h in range(1, max_horizon + 1):
            p_h, t_h = predictions_by_horizon[h], ground_truth_by_horizon[h]
            s_h = sigmas_by_horizon[h] if sigmas_by_horizon is not None else None

            for m in range(1, 13):
                mask = (np.array([d.month for d in test_dates]) == m) & (np.array([d.year for d in test_dates]) == year)
                idxs = np.where(mask)[0]

                if len(idxs) > 0:
                    hint = predict_baseline(baseline_model, X_test[idxs])

                    # Extract and reverse transform data
                    if is_log_var:
                        p_b = np.expm1(target_scaler.inverse_transform(hint.reshape(-1, 1)))
                        p_f = np.expm1(target_scaler.inverse_transform(p_h[idxs].reshape(-1, 1)))
                        t_r = np.expm1(target_scaler.inverse_transform(t_h[idxs].reshape(-1, 1)))
                    else:
                        p_b = target_scaler.inverse_transform(hint.reshape(-1, 1))
                        p_f = target_scaler.inverse_transform(p_h[idxs].reshape(-1, 1))
                        t_r = target_scaler.inverse_transform(t_h[idxs].reshape(-1, 1))

                    if metric_name == 'RMSE':
                        base_mat[m-1, h-1] = np.sqrt(np.mean((t_r - p_b)**2))
                        hyb_mat[m-1, h-1] = np.sqrt(np.mean((t_r - p_f)**2))
                    
                    elif metric_name == 'ACC':
                        base_mat[m-1, h-1] = calc_acc(p_b, t_r)
                        hyb_mat[m-1, h-1] = calc_acc(p_f, t_r)
                    
                    elif metric_name == 'CRPS':
                        s_f = (s_h[idxs].reshape(-1, 1) * target_range) + 1e-6 # type: ignore
                        base_mat[m-1, h-1] = np.mean(np.abs(t_r - p_b))
                        hyb_mat[m-1, h-1] = np.mean(ps.crps_gaussian(t_r.flatten(), p_f.flatten(), s_f.flatten()))

        ax1, ax2, cbar_ax = axes[idx, 0], axes[idx, 1], axes[idx, 2]

        cfg = {
            'cmap': cmap, 'annot': True, 'fmt': '.3f',
            'annot_kws': {'weight': 'bold', 'size': 13},
            'linecolor': '#000000', 'linewidths': 0.5
        }

        if metric_name == 'ACC':
            cfg['vmin'], cfg['vmax'] = 0, 1
        elif metric_name == 'CRPS':
            cfg['vmin'] = min(np.nanmin(base_mat), np.nanmin(hyb_mat))
            cfg['vmax'] = max(np.nanmax(base_mat), np.nanmax(hyb_mat))

        is_last_row = (idx == len(years) - 1)
        x_ticks = horizon_labels if is_last_row else False

        sns.heatmap(base_mat, ax=ax1, mask=np.isnan(base_mat), cbar=False, 
                    xticklabels=x_ticks, yticklabels=month_names, **cfg)
        ax1.set_title(f'Baseline: {bs_md} ({year})', weight='bold', fontsize=18)
        ax1.set_ylabel('Month', weight='bold', size=16)
        ax1.set_yticklabels(month_names, size=14, rotation=0)
        
        sns.heatmap(hyb_mat, ax=ax2, mask=np.isnan(hyb_mat), cbar=True, cbar_ax=cbar_ax,
                    xticklabels=x_ticks, yticklabels=False, **cfg)
        ax2.set_title(f'Hybrid: {unet_type} U-Net ({year})', weight='bold', fontsize=18)
        ax2.set_ylabel('')

        if is_last_row:
            for ax in [ax1, ax2]:
                ax.set_xlabel('Forecast Horizon', weight='bold', size=16)
                ax.set_xticklabels(horizon_labels, size=14)
        else:
            ax1.set_xlabel('')
            ax2.set_xlabel('')

    plt.savefig(filepath, bbox_inches='tight')
    plt.show()


def plot_rmse_comparison_grids(predictions_by_horizon, ground_truth_by_horizon, X_test, baseline_model, 
                               target_scaler, num_eval_samples, max_horizon, start_dt, 
                               bs_md, unet_type, seq_len, var_name, filepath):
    '''Generates Monthly vs. Lead-time RMSE heatmaps.'''
    plot_monthly_metric_grids_base(
        metric_name='RMSE', cmap='Reds', 
        predictions_by_horizon=predictions_by_horizon, ground_truth_by_horizon=ground_truth_by_horizon, 
        X_test=X_test, baseline_model=baseline_model, target_scaler=target_scaler, 
        num_eval_samples=num_eval_samples, max_horizon=max_horizon, start_dt=start_dt, 
        bs_md=bs_md, unet_type=unet_type, seq_len=seq_len, var_name=var_name, filepath=filepath
    )


def plot_acc_comparison_grids(predictions_by_horizon, ground_truth_by_horizon, X_test, baseline_model, 
                               target_scaler, num_eval_samples, max_horizon, start_dt, 
                               bs_md, unet_type, seq_len, var_name, filepath):
    '''Generates Monthly vs. Lead-time ACC heatmaps.'''
    plot_monthly_metric_grids_base(
        metric_name='ACC', cmap='Greens', 
        predictions_by_horizon=predictions_by_horizon, ground_truth_by_horizon=ground_truth_by_horizon, 
        X_test=X_test, baseline_model=baseline_model, target_scaler=target_scaler, 
        num_eval_samples=num_eval_samples, max_horizon=max_horizon, start_dt=start_dt, 
        bs_md=bs_md, unet_type=unet_type, seq_len=seq_len, var_name=var_name, filepath=filepath
    )


def plot_crps_comparison_grids(predictions_by_horizon, ground_truth_by_horizon, sigmas_by_horizon, X_test, 
                               baseline_model, target_scaler, num_eval_samples, max_horizon, start_dt,
                               bs_md, unet_type, seq_len, var_name, filepath):
    '''Generates Monthly vs. Lead-time CRPS heatmaps.'''
    plot_monthly_metric_grids_base(
        metric_name='CRPS', cmap='Blues', 
        predictions_by_horizon=predictions_by_horizon, ground_truth_by_horizon=ground_truth_by_horizon, 
        X_test=X_test, baseline_model=baseline_model, target_scaler=target_scaler, 
        num_eval_samples=num_eval_samples, max_horizon=max_horizon, start_dt=start_dt, 
        bs_md=bs_md, unet_type=unet_type, seq_len=seq_len, var_name=var_name, filepath=filepath,
        sigmas_by_horizon=sigmas_by_horizon
    )


# Forecast Trajectory and Uncertainty
def plot_forecast_trajectory(sample_start_idx, H, W,
                             predictions_by_horizon, ground_truth_by_horizon, sigmas_by_horizon,
                             X_test, baseline_model, target_scaler,
                             height, width, bs_md, unet_type, seq_len, start_dt,
                             ytick_labels, xtick_labels, units,
                             var_name, filepath):
    '''
    Plots the recursive forecast trajectory.
    '''
    style_map = {
        'Total Precipitation': {'color': '#1f77b4'},
        'Temperature': {'color': '#d62728'},
        'Relative Humidity': {'color': '#2ca02c'},
        'Default': {'color': '#9467bd'}
    }

    steps = 12
    time_axis = np.arange(1, steps + 1) * 6  # 6h, 12h, ..., 72h

    actuals, base_preds, hybrid_preds = [], [], []
    hybrid_sigmas = []

    # Scaling factor for Sigma unscaling
    tr = 1.0 / target_scaler.scale_[0] if hasattr(target_scaler, 'scale_') else target_scaler.data_range_[0]

    var_clean = var_name.lower()
    if any(x in var_clean for x in ['tp', 'precipitation', 'rain']):
        main_color = style_map['Total Precipitation']['color']
    elif any(x in var_clean for x in ['temp', 't2m', 'temperature']):
        main_color = style_map['Temperature']['color']
    elif any(x in var_clean for x in ['rh', 'humidity']):
        main_color = style_map['Relative Humidity']['color']
    else:
        main_color = style_map['Default']['color']

    is_precip = any(x in var_clean for x in ['tp', 'precipitation', 'rain'])
    is_log_var = is_precip or any(x in var_clean for x in ['ws', 'wind'])

    for h in range(1, steps + 1):
        val_t = ground_truth_by_horizon[h][sample_start_idx, H, W].item()
        val_ph = predictions_by_horizon[h][sample_start_idx, H, W].item()
        val_sh = sigmas_by_horizon[h][sample_start_idx, H, W].item()

        curr_X = X_test[sample_start_idx + h - 1: sample_start_idx + h]
        val_pb_scaled = predict_baseline(baseline_model, curr_X)

        if is_log_var:
            t_phys = np.expm1(target_scaler.inverse_transform(np.array([[val_t]]))).item()
            ph_phys = np.expm1(target_scaler.inverse_transform(np.array([[val_ph]]))).item()
            pb_grid_phys = np.expm1(target_scaler.inverse_transform(val_pb_scaled.reshape(-1, 1))).reshape(height, width)
        else:
            t_phys = target_scaler.inverse_transform(np.array([[val_t]]))[0, 0]
            ph_phys = target_scaler.inverse_transform(np.array([[val_ph]]))[0, 0]
            pb_grid_phys = target_scaler.inverse_transform(val_pb_scaled.reshape(-1, 1)).reshape(height, width)

        pb_pixel_phys = pb_grid_phys[H, W].item()

        actuals.append(float(max(0, t_phys)))
        hybrid_preds.append(float(max(0, ph_phys)))
        base_preds.append(float(max(0, pb_pixel_phys)))
        hybrid_sigmas.append(float(val_sh * tr))

    actuals = np.array(actuals)
    hybrid_preds = np.array(hybrid_preds)
    base_preds = np.array(base_preds)
    hybrid_sigmas = np.array(hybrid_sigmas)

    rmse_base = np.sqrt(np.mean((actuals - base_preds) ** 2))
    base_sigmas = np.full_like(base_preds, rmse_base)

    if is_precip:
        plot_actuals = np.cumsum(actuals)
        plot_base = np.cumsum(base_preds)
        plot_hybrid = np.cumsum(hybrid_preds)
        plot_base_err = 1.96 * np.sqrt(np.cumsum(base_sigmas ** 2))
        plot_hybrid_err = 1.96 * np.sqrt(np.cumsum(hybrid_sigmas ** 2))
    else:
        plot_actuals = actuals
        plot_base = base_preds
        plot_hybrid = hybrid_preds
        plot_base_err = 1.96 * base_sigmas
        plot_hybrid_err = 1.96 * hybrid_sigmas

    plt.figure(figsize=(14, 7))

    plt.plot(time_axis, plot_base, color='black', linestyle='-',
         label=f'Stage 1: {bs_md}', linewidth=3, marker='o', markersize=7)
    plt.fill_between(time_axis,
                     (plot_base - plot_base_err),
                     (plot_base + plot_base_err),
                     color='gray', alpha=0.15,
                     label='95% CI (Fixed Spread)')

    plt.plot(time_axis, plot_hybrid, color=main_color,
             label=f'Stage 2: {unet_type} U-Net',
             linewidth=3, marker='s', markersize=7)
    plt.fill_between(time_axis,
                     (plot_hybrid - plot_hybrid_err),
                     (plot_hybrid + plot_hybrid_err),
                     color=main_color, alpha=0.2,
                     label='95% CI (Dynamic Spread)')

    plt.plot(time_axis, plot_actuals, 'k--',
         label='ERA5 Ground Truth',
         linewidth=3, alpha=0.7)

    lat_val, lon_val = ytick_labels[H], xtick_labels[W]
    start_date = start_dt + timedelta(hours=int(sample_start_idx + seq_len) * 6)

    plt.title(
        f'Forecast Trajectory: {lat_val}°N, {lon_val}°E | {bs_md} - {unet_type} U-Net Hybrid | '
        f'Init: {start_date.strftime("%b %d, %Y (%H:%M)")}',
        weight='bold', fontsize=20
    )

    plt.xticks(time_axis, size=14)
    plt.yticks(size=14)
    plt.xlabel('Forecast Horizon (Hours)', weight='bold', fontsize=16)
    plt.ylabel(f'{var_name} ({units})', weight='bold', fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', prop={'weight': 'bold', 'size': 14}, ncol=2)

    plt.tight_layout()
    
    plt.savefig(f'{filepath}', bbox_inches='tight')
    
    plt.show()



# Reliability Comparison
def plot_comparative_reliability_diagrams(predictions_by_horizon, ground_truth_by_horizon, sigmas_by_horizon, X_test, tar_hor, 
                                          baseline_model, target_scaler, bs_md, unet_type, seq_len, vn, var_name, filepath):
    '''
    Plots Z-score distributions for specific horizons: T+6h, T+24h, T+48h, and T+72h.
    Matches Observed Residuals / Predicted Sigma against a Standard Normal PDF.
    '''
    target_horizons = tar_hor if isinstance(tar_hor, list) else [tar_hor]
    num_plots = len(target_horizons)
    
    fig, axes = plt.subplots(
        num_plots, 2, 
        figsize=(16, 4 * num_plots), 
        squeeze=False, 
        sharex=True, 
        sharey=True,
        layout='constrained'
    )

    # Scale factor for unscaling Sigma
    tr = 1.0 / target_scaler.scale_[0] if hasattr(target_scaler, 'scale_') else target_scaler.data_range_[0]

    # Ideal Standard Normal Distribution
    xn = np.linspace(-5, 5, 100)
    pn = stats.norm.pdf(xn, 0, 1)

    is_log_var = var_name.lower() in ['tp', 'total precipitation', 'ws', 'wind speed']

    for plot_idx, h in enumerate(target_horizons):
        t_raw = ground_truth_by_horizon[h]
        p_f_raw = predictions_by_horizon[h]
        s_f_raw = sigmas_by_horizon[h]

        # Baseline Inference
        hint = predict_baseline(baseline_model, X_test[h-1 : len(t_raw) + h - 1])

        # Universal Inverse Transformation
        if is_log_var:
            actual = np.expm1(target_scaler.inverse_transform(t_raw.reshape(-1, 1))).flatten()
            pb_mu = np.expm1(target_scaler.inverse_transform(hint.reshape(-1, 1))).flatten()
            pf_mu = np.expm1(target_scaler.inverse_transform(p_f_raw.reshape(-1, 1))).flatten()
        else:
            actual = target_scaler.inverse_transform(t_raw.reshape(-1, 1)).flatten()
            pb_mu = target_scaler.inverse_transform(hint.reshape(-1, 1)).flatten()
            pf_mu = target_scaler.inverse_transform(p_f_raw.reshape(-1, 1)).flatten()

        sf_std = (s_f_raw.flatten() * tr) + 1e-6

        rmse_b = np.sqrt(np.mean((actual - pb_mu)**2))
        zb = (actual - pb_mu) / (rmse_b + 1e-6)

        zh = (actual - pf_mu) / sf_std

        configs = [(zb, 'coral', f'{bs_md}'), 
                   (zh, 'royalblue', f'{unet_type} U-Net')]

        for i, (z, color, title) in enumerate(configs):
            ax = axes[plot_idx, i]

            z_filt = z[(z > -10) & (z < 10)]

            ax.hist(z_filt, bins=80, density=True, alpha=0.6, color=color, edgecolor='white', label='Observed')
            ax.plot(xn, pn, 'r--', lw=2.5, label=r'$\mathcal{N}(0,1)$', zorder=5)

            ax.set_title(f'T+{h*6}h | {title} | {vn} | {seq_len}-Step Inp Seq', weight='bold', fontsize=20)
            ax.set_xlim([-5, 5])
            ax.set_ylim([0, 0.6])
            ax.grid(True, ls='--', alpha=0.3)
            ax.legend(loc='upper right', prop={'weight': 'bold', 'size': 14})
            
            ax.tick_params(axis='both', which='major', labelsize=12)

            if plot_idx == num_plots - 1:
                ax.set_xlabel('Standardized Residual (Z-Score)', weight='bold', size=16)

            if i == 0: 
                ax.set_ylabel('Density', weight='bold', size=16)

    plt.savefig(f'{filepath}', bbox_inches='tight')
    plt.show()


# Spatial Bias Comparison
def plot_spatial_bias_comparison(seq_len, predictions_by_horizon, ground_truth_by_horizon, X_test, tar_hor, baseline_model, 
                                 target_scaler, height, width, xtick_labels, ytick_labels, bs_md, unet_type, var_name, vn, filepath):
    '''
    Generates side-by-side mean bias maps (Mean Forecast - Mean Observed) for T+6h and T+72h.
    Red (Positive) = Systematic Overestimation | Blue (Negative) = Systematic Underestimation.
    '''
    target_horizons = tar_hor if isinstance(tar_hor, list) else [tar_hor]
    num_rows = len(target_horizons)

    fig, axes = plt.subplots(
        num_rows, 2, 
        figsize=(18, 9 * num_rows), 
        layout='constrained', 
        gridspec_kw={'wspace': 0.02, 'hspace': 0.02}
    )

    is_log_var = var_name.lower() in ['tp', 'total precipitation', 'ws', 'wind speed']

    for row_idx, h_idx in enumerate(target_horizons):
        t_raw = ground_truth_by_horizon[h_idx]
        p_f_raw = predictions_by_horizon[h_idx]
        num_samples = len(t_raw)

        # Baseline Inference
        hint = predict_baseline(baseline_model, X_test[h_idx-1: num_samples + h_idx - 1])

        # Inverse Transformation (Samples, H, W)
        if is_log_var:
            pb = np.expm1(target_scaler.inverse_transform(hint.reshape(-1, 1))).reshape(num_samples, height, width)
            pf = np.expm1(target_scaler.inverse_transform(p_f_raw.reshape(-1, 1))).reshape(num_samples, height, width)
            tr = np.expm1(target_scaler.inverse_transform(t_raw.reshape(-1, 1))).reshape(num_samples, height, width)
        else:
            pb = target_scaler.inverse_transform(hint.reshape(-1, 1)).reshape(num_samples, height, width)
            pf = target_scaler.inverse_transform(p_f_raw.reshape(-1, 1)).reshape(num_samples, height, width)
            tr = target_scaler.inverse_transform(t_raw.reshape(-1, 1)).reshape(num_samples, height, width)

        b_base = np.mean(pb - tr, axis=0)
        b_hyb = np.mean(pf - tr, axis=0)

        mx = max(np.abs(b_base).max(), np.abs(b_hyb).max())

        cfg = {
            'cmap': 'coolwarm',
            'annot': True,
            'fmt': '.3f',
            'center': 0,
            'vmin': -mx,
            'vmax': mx,
            'xticklabels': xtick_labels,
            'yticklabels': ytick_labels,
            'linewidths': 0.5,
            'linecolor': '#000000',
            'annot_kws': {'weight': 'bold', 'size': 13}
        }

        sns.heatmap(b_base, ax=axes[row_idx, 0], cbar=False, **cfg)
        axes[row_idx, 0].set_title(f'T+{h_idx*6}h Bias | {bs_md} | {vn} | {seq_len}-Step Inp Seq', weight='bold', fontsize=20)

        sns.heatmap(b_hyb, ax=axes[row_idx, 1], cbar=False, **cfg)
        axes[row_idx, 1].set_title(f'T+{h_idx*6}h Bias | {unet_type} U-Net | {vn} | {seq_len}-Step Inp Seq', weight='bold', fontsize=20)

        mappable = axes[row_idx, 1].collections[0]
        cbar = fig.colorbar(mappable, ax=[axes[row_idx, 0], axes[row_idx, 1]], pad=0.02)
        cbar.set_label('Mean Bias', weight='bold', size=16)
        cbar.ax.tick_params(labelsize=14)

        for col_idx in range(2):
            axes[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=14)
            
            if row_idx == num_rows - 1:
                axes[row_idx, col_idx].set_xlabel('Longitude (°E)', weight='bold', fontsize=16)
                axes[row_idx, col_idx].set_xticklabels(xtick_labels, rotation=0) 
            else:
                axes[row_idx, col_idx].set_xlabel('')
                axes[row_idx, col_idx].set_xticklabels([])

            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel('Latitude (°N)', weight='bold', fontsize=16)
                axes[row_idx, col_idx].set_yticklabels(ytick_labels, rotation=0)
            else:
                axes[row_idx, col_idx].set_ylabel('')
                axes[row_idx, col_idx].set_yticklabels([])

    plt.savefig(f'{filepath}', bbox_inches='tight')
    plt.show()