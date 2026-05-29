# Importing the required libraries
import torch
from tqdm import tqdm

import gc
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid")

# Optimizer and Training Functions for Neural Network
def nn_optimizer(nn_model):
    nn_optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=0)
    return nn_optimizer

def nn_train_model(nn_model, nn_train_loader, nn_val_loader, nn_optimizer, 
                   nn_loss_fn, epochs):
    train_losses_nn, val_losses_nn = [], []
    best_val_loss = float('inf')
    
    best_model_wts = copy.deepcopy(nn_model.state_dict())

    for epoch in range(epochs):
        # --- Training Phase ---
        nn_model.train()
        total_train_loss = 0.0
        
        # Fixed the set_postfix logic here for better live tracking
        train_progress = tqdm(enumerate(nn_train_loader), 
                              total=len(nn_train_loader),
                              desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                              leave=False)

        for step, (features, targets) in train_progress:
            nn_optimizer.zero_grad()
            predictions = nn_model(features)
            loss = nn_loss_fn(predictions, targets)
            loss.backward()
            nn_optimizer.step()

            total_train_loss += loss.item()
            train_progress.set_postfix(loss=f"{total_train_loss / (step + 1):.6f}")

        avg_train_loss = total_train_loss / len(nn_train_loader)
        train_losses_nn.append(avg_train_loss)

        # --- Validation Phase ---
        nn_model.eval()
        total_val_loss = 0.0
        
        with torch.inference_mode():
            for features, targets in nn_val_loader:
                predictions = nn_model(features)
                v_loss = nn_loss_fn(predictions, targets)
                total_val_loss += v_loss.item()

        avg_val_loss = total_val_loss / len(nn_val_loader)
        val_losses_nn.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(nn_model.state_dict())
            print("  --> Saved Best Weights")

        gc.collect()
        torch.cuda.empty_cache()

    nn_model.load_state_dict(best_model_wts)
    
    return train_losses_nn, val_losses_nn

# Optimizer and Training Functions for LSTM and GRU
def lg_optimizer(lg_model):
    lg_optimizer = torch.optim.AdamW(lg_model.parameters(), lr=1e-3, weight_decay=0)
    return lg_optimizer

def lg_train_model(lg_model, lg_train_loader, lg_val_loader, lg_optimizer, lg_loss_fn, epochs):
    train_losses_lg, val_losses_lg = [], []
    best_val_loss = float('inf')
    
    best_model_wts = copy.deepcopy(lg_model.state_dict())

    for epoch in range(epochs):
        # --- Training Phase ---
        lg_model.train()
        total_train_loss = 0.0
        
        train_progress = tqdm(enumerate(lg_train_loader), 
                              total=len(lg_train_loader), 
                              desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                              leave=False)

        for step, (features, targets) in train_progress:
            lg_optimizer.zero_grad()
            predictions = lg_model(features)

            loss = lg_loss_fn(predictions, targets)

            loss.backward()
            lg_optimizer.step()

            total_train_loss += loss.item()
            train_progress.set_postfix(loss=f"{total_train_loss / (step + 1):.6f}") 

        avg_train_loss = total_train_loss / len(lg_train_loader)
        train_losses_lg.append(avg_train_loss)

        # --- Validation Phase ---
        lg_model.eval()
        total_val_loss = 0.0
        
        val_progress = tqdm(enumerate(lg_val_loader), 
                            total=len(lg_val_loader), 
                            desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                            leave=False)

        with torch.inference_mode():
            for step, (features, targets) in val_progress:
                predictions = lg_model(features)
                v_loss = lg_loss_fn(predictions, targets)

                total_val_loss += v_loss.item()
                val_progress.set_postfix(val_loss=f"{total_val_loss / (step + 1):.6f}")

        avg_val_loss = total_val_loss / len(lg_val_loader)
        val_losses_lg.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(lg_model.state_dict())
            print("  --> Saved Best Weights")

        gc.collect() 
        torch.cuda.empty_cache() 

    lg_model.load_state_dict(best_model_wts)

    return train_losses_lg, val_losses_lg

# Plotting Function for Training Dynamics
def plot_training_dynamics(train_losses, val_losses, loss_name, model_name, seq_len):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    num_epochs = len(train_losses)
    epochs_range = np.arange(num_epochs)

    ax1.plot(epochs_range, train_losses, label='Train Loss', linewidth=2, color='red')
    ax1.plot(epochs_range, val_losses, label='Val Loss', linewidth=2, color='green')

    ax1.set_ylabel(f'{loss_name}', weight='bold', fontsize=14)
    ax1.set_xlabel('Epoch', weight='bold', fontsize=14)

    ax1.set_xlim(-0.5, num_epochs - 0.5)

    tick_pos = list(range(0, num_epochs, 5))
    if (num_epochs - 1) not in tick_pos:
        tick_pos.append(num_epochs - 1)

    ax1.xaxis.set_major_locator(ticker.FixedLocator(tick_pos))
    ax1.xaxis.set_major_formatter(ticker.FixedFormatter([str(p + 1) for p in tick_pos]))
    ax1.grid(True, linestyle=':', alpha=0.6)

    best_epoch = np.argmin(val_losses)
    min_val_loss = val_losses[best_epoch]

    ax1.axvline(float(best_epoch), linestyle='-.', color='black', alpha=0.3)
    ax1.scatter(float(best_epoch), min_val_loss, s=100, color='black', zorder=5)

    ax1.annotate(
        f'Min Val: {min_val_loss:.6f}',
        xy=(float(best_epoch), min_val_loss),
        xytext=(float(best_epoch) - 3.75, min_val_loss + (max(val_losses) * 0.1)), 
        ha='center',
        arrowprops=dict(arrowstyle='->', connectionstyle="arc3", color='black'),
        fontsize=12, weight='bold'
    )

    ax1.legend(loc='best', prop={'weight': 'bold', 'size': 12}, frameon=True)

    plt.title(f'{model_name} Training Dynamics | {seq_len}-Step Input Sequence', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()