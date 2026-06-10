# Importing the required libraries
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid")

# Optimizer and Scheduler Initialization Function
def get_optimizer_and_scheduler(model):
    """Initializes the AdamW optimizer and ReduceLROnPlateau scheduler."""
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1
    )
    return optimizer, scheduler

# Training Loop Function
def train_model(model, train_loader, val_loader, optimizer, scheduler, 
                loss_fn, epochs, device):
    """
    Training Loop for Probabilistic U-Net.
    (Assumes that data already exists on the GPU)
    """
    best_val_loss = float('inf')
    train_losses, val_losses, lr_history = [], [], []
    
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            leave=False
        )

        for step, (features, targets) in enumerate(train_progress):
            optimizer.zero_grad(set_to_none=True)

            prediction = model(features) 

            # Split output into Mean (mu) and Log-Variance
            mu = prediction[:, 0:1]
            log_var = prediction[:, 1:2]

            # Convert log-variance to standard variance with a safety epsilon
            variance = F.softplus(log_var) + 1e-6 

            loss = loss_fn(mu, targets, variance).mean()

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            train_progress.set_postfix(
                loss=f"{total_train_loss / (step + 1):.6f}",
                lr=f"{current_lr:.2e}"
            )

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0

        val_progress = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Val]",
            leave=False
        )

        with torch.inference_mode():
            for step, (f_v, t_v) in enumerate(val_progress):
                f_v = f_v.to(device, non_blocking=True)
                t_v = t_v.to(device, non_blocking=True)

                pred_v = model(f_v)
                mu_v = pred_v[:, 0:1]
                var_v = F.softplus(pred_v[:, 1:2]) + 1e-6

                v_loss = loss_fn(mu_v, t_v, var_v).mean()

                total_val_loss += v_loss.item()

                val_progress.set_postfix(
                    val_loss=f"{total_val_loss / (step + 1):.6f}"
                )

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state_to_save = (
                model._orig_mod.state_dict()
                if hasattr(model, '_orig_mod')
                else model.state_dict()
            )
            best_model_wts = copy.deepcopy(state_to_save)
            print("  --> Saved Best Weights")

        scheduler.step(avg_val_loss)

        if np.isnan(avg_train_loss):
            print("!!! NaN detected in loss. Aborting training. !!!")
            break
    
    model.load_state_dict(best_model_wts)

    return train_losses, val_losses, lr_history

# Training Dynamics Plotting Function
def plot_training_dynamics(train_losses, val_losses, lr_history, model_name, sql):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    num_epochs = len(train_losses)
    epochs_range = np.arange(num_epochs)

    ax1.plot(epochs_range, train_losses, label='Train Loss', linewidth=2, color='red')
    ax1.plot(epochs_range, val_losses, label='Val Loss', linewidth=2, color='green')
    
    ax1.set_xlim(-1.5, num_epochs + 0.5)

    tick_pos = list(range(0, num_epochs, 5))
    if (num_epochs - 1) not in tick_pos:
        tick_pos.append(num_epochs - 1)
        
    ax1.xaxis.set_major_locator(ticker.FixedLocator(tick_pos))
    ax1.xaxis.set_major_formatter(
        ticker.FixedFormatter([str(p + 1) for p in tick_pos])
    )

    ax1.set_xlabel('Epoch', weight='bold', fontsize=14)
    ax1.set_ylabel('Gaussian NLL Loss', weight='bold', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.grid(False)
    
    ax2.plot(epochs_range, lr_history, linestyle='--', alpha=0.6, color='blue', label='Learning Rate')
    ax2.set_ylabel('Learning Rate', weight='bold', fontsize=14)

    best_epoch = np.argmin(val_losses)
    min_val_loss = val_losses[best_epoch]

    ax1.axvline(float(best_epoch), linestyle='-.', alpha=0.5)
    ax1.scatter(float(best_epoch), min_val_loss, s=100, color='black', zorder=5)

    ax1.annotate(
        f'Min Val Loss: {min_val_loss:.5f}',
        xy=(float(best_epoch), min_val_loss),
        xytext=(best_epoch + 1.5, min_val_loss + 0.5),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=12, weight='bold'
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', prop={'weight': 'bold', 'size': 12}, frameon=True)

    plt.title(f'{model_name} Training Dynamics | {sql}-Step Input Sequence', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()