import torch
import torch.nn as nn

# Elastic Net
class PyTorchElasticNet(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.01, l1_ratio=0.5):
        super(PyTorchElasticNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def forward(self, x):
        return self.linear(x)

    def fit(self, X_train, y_train, X_val, y_val, lr=1e-4, epochs=100, batch_size=1024, tol=1e-4):
        device = next(self.parameters()).device
        
        n_samples = X_train.shape[0]
        n_val = X_val.shape[0]

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            # Training Phase
            self.train()
            epoch_train_loss = 0.0
            
            # Shuffling on GPU
            perm = torch.randperm(n_samples, device=device)

            for i in range(0, n_samples, batch_size):
                optimizer.zero_grad()
                
                idx = perm[i:i + batch_size]
                batch_X = X_train[idx]
                batch_y = y_train[idx]

                outputs = self.forward(batch_X)
                mse_loss = criterion(outputs, batch_y)

                # Regularization Penalties
                l1_penalty = torch.norm(self.linear.weight, 1)
                l2_penalty = torch.norm(self.linear.weight, 2)**2

                loss = mse_loss + self.alpha * (
                    self.l1_ratio * l1_penalty +
                    (1 - self.l1_ratio) * 0.5 * l2_penalty
                )

                loss.backward()
                
                optimizer.step()
                epoch_train_loss += loss.item()

            # Validation Phase
            self.eval()
            epoch_val_loss = 0.0
            
            with torch.inference_mode():
                for j in range(0, n_val, batch_size):
                    batch_X_v = X_val[j:j + batch_size]
                    batch_y_v = y_val[j:j + batch_size]
                    
                    val_outputs = self.forward(batch_X_v)
                    epoch_val_loss += criterion(val_outputs, batch_y_v).item()

# Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNet, self).__init__()
        
        layers = []
        in_dim = input_size
        
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

# LSTM    
class LSTMModel(nn.Module):
    def __init__(self, input_features, hidden_sizes, output_size):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_sizes[0]
        self.num_layers = len(hidden_sizes)

        self.lstm = nn.LSTM(
            input_features, 
            self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
    
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        # out: (Batch, Seq_Len, Hidden_Dim)
        out, _ = self.lstm(x)
        
        # Output of the last time step
        last_time_step_out = out[:, -1, :]
        
        # Final linear mapping to grid pixels
        return self.fc(last_time_step_out)

# GRU    
class GRUModel(nn.Module):
    def __init__(self, input_features, hidden_sizes, output_size):
        super(GRUModel, self).__init__()
        
        self.hidden_dim = hidden_sizes[0]
        self.num_layers = len(hidden_sizes)

        self.gru = nn.GRU(
            input_features, 
            self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        
        last_time_step_out = out[:, -1, :]
        
        return self.fc(last_time_step_out)