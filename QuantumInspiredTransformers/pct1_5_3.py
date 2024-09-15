#!/usr/bin/env python
# coding: utf-8

# In[12]:


### ***Environment Setup***


# In[ ]:


# Install necessary libraries
get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
get_ipython().system('pip install -q yfinance')
get_ipython().system('pip install -q scikit-learn')
get_ipython().system('pip install -q optuna')


# In[1]:


import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}:", torch.cuda.get_device_name(i))


# ### ***Imports and Configuration***

# In[91]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import math
import random
import sys
import optuna
import warnings
warnings.filterwarnings('ignore')


# In[92]:


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# In[93]:


# Configuration Parameters
config = {
    'stock': '^GSPC',
    'start_date': '1980-01-01',
    'end_date': '2024-08-10',
    'forecast_horizon': 16,
    'seq_length': 64,
    'batch_size': 128,
    'model_dim': 512,
    'num_heads': 16,
    'num_layers': 2,
    'dropout': 0.4417422120891107,
    'learning_rate': 6.384360487096914e-05,
    'num_epochs': 64,
    'early_stopping_patience': 20,
    'window_size': 10,  # For rolling mean
    'subset_start_date': '2024-06-01',
    'subset_end_date': '2024-08-10',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'max_len': 64,  # Should match seq_length
    'scaling_method': 'standard',  # 'standard' or 'minmax'
}


# In[94]:


print(f"device: {config['device']}")


# ### ***Data Preparation***

# In[95]:


def download_and_preprocess_data(stock, start, end):
    """
    Downloads stock data and extracts only the 'Close' prices.

    Args:
        stock (str): Stock ticker symbol.
        start (str): Start date in 'YYYY-MM-DD'.
        end (str): End date in 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: Dataframe containing only the 'Close' prices.
    """
    try:
        yfd = yf.download(stock, start=start, end=end)
    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)

    # Focus only on 'Close' prices and drop any rows with missing values
    df = yfd[['Close']].dropna()

    return df

df = download_and_preprocess_data(config['stock'], config['start_date'], config['end_date'])
df.head()


# In[115]:


def normalize_data(df, method='standard'):
    """
    Normalizes the 'Close' prices using the specified method.

    Args:
        df (pd.DataFrame): Dataframe containing only 'Close' prices.
        method (str): 'standard' or 'minmax'.

    Returns:
        pd.DataFrame, scaler: Normalized data and the scaler used.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaling_method should be 'standard' or 'minmax'")

    # Normalize the 'Close' prices
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    return scaled_df, scaler

# Apply normalization to your data
scaled_df, scaler = normalize_data(df, method=config['scaling_method'])


# In[97]:


def create_sequences(data, seq_length, forecast_horizon):
    """
    Creates input sequences and targets for time series forecasting.

    Args:
        data (np.ndarray): Normalized data.
        seq_length (int): Length of input sequences.
        forecast_horizon (int): Number of future steps to predict.

    Returns:
        np.ndarray, np.ndarray: Input sequences and targets.
    """
    X = []
    y = []

    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon, 0])  # Assuming 'Close' is at index 3

    X = np.array(X)
    y = np.array(y)

    return X, y

# Selecting relevant features (e.g., all features)
data = scaled_df.values
X, y = create_sequences(data, config['seq_length'], config['forecast_horizon'])
print(f"Input shape: {X.shape}, Target shape: {y.shape}")


# In[98]:


def split_data(X, y, train_size=0.7, val_size=0.15):
    """
    Splits data into training, validation, and test sets sequentially.

    Args:
        X (np.ndarray): Input sequences.
        y (np.ndarray): Targets.
        train_size (float): Proportion of data for training.
        val_size (float): Proportion of data for validation.

    Returns:
        Tuple of tensors: x_train, x_val, x_test, y_train, y_val, y_test
    """
    total_samples = X.shape[0]
    train_end = int(total_samples * train_size)
    val_end = train_end + int(total_samples * val_size)

    x_train = X[:train_end]
    y_train = y[:train_end]
    x_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    x_test = X[val_end:]
    y_test = y[val_end:]

    return x_train, x_val, x_test, y_train, y_val, y_test

x_train, x_val, x_test, y_train, y_val, y_test = split_data(X, y)
print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")


# In[99]:


def to_tensor(x, y):
    """
    Converts numpy arrays to PyTorch tensors.

    Args:
        x (np.ndarray): Input sequences.
        y (np.ndarray): Targets.

    Returns:
        torch.Tensor, torch.Tensor: Input and target tensors.
    """
    X_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

X_train, Y_train = to_tensor(x_train, y_train)
X_val, Y_val = to_tensor(x_val, y_val)
X_test, Y_test = to_tensor(x_test, y_test)


# In[100]:


def create_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size):
    """
    Creates PyTorch DataLoaders for training, validation, and testing.

    Args:
        X_train, Y_train, X_val, Y_val, X_test, Y_test (torch.Tensor): Datasets.
        batch_size (int): Batch size.

    Returns:
        DataLoader, DataLoader, DataLoader: Train, validation, and test loaders.
    """
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = create_dataloaders(
    X_train, Y_train,
    X_val, Y_val,
    X_test, Y_test,
    config['batch_size']
)


# ### ***Model Architecture***

# In[101]:


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings.

    Args:
        d_model (int): The dimension of the embeddings.
        max_len (int): The maximum length of the sequences.
        dropout (float): Dropout rate.
    """
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        if d_model % 2 != 0:
            div_term = torch.exp(torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# In[102]:


class MarketPredictor(nn.Module):
    """
    Transformer-based model for market prediction.

    Args:
        num_features (int): Number of input features.
        seq_length (int): Length of input sequences.
        d_model (int): Dimension of the model.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer encoder layers.
        forecast_horizon (int): Number of future steps to predict.
        dropout (float): Dropout rate.
    """
    def __init__(self, num_features, seq_length=64, d_model=256, nhead=16, num_layers=2, forecast_horizon=16, dropout=0.2):
        super(MarketPredictor, self).__init__()
        self.embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, forecast_horizon)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, forecast_horizon).
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc_out(x)
        return x


# ### ***Training***

# In[103]:


def total_loss(output, target, prev_closing_price):
    """
    Calculates the combined loss.

    Args:
        output (torch.Tensor): Predicted prices.
        target (torch.Tensor): Actual prices.
        prev_closing_price (torch.Tensor): Previous closing prices.

    Returns:
        torch.Tensor: Combined loss.
    """
    mse = nn.MSELoss()

    # Price prediction loss (MSE)
    price_loss = mse(output, target)

    # Percent change calculation
    predicted_pct_change = (output - prev_closing_price) / prev_closing_price
    actual_pct_change = (target - prev_closing_price) / prev_closing_price

    # Percent change loss (MSE)
    pct_change_loss = mse(predicted_pct_change, actual_pct_change)
    
    mse = mean_squared_error(target.detach(), output.detach())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target.detach(), output.detach())

    combined_loss = price_loss + mse + rmse + mae 
        

    return combined_loss


# In[104]:


def calculate_metrics(predictions, targets):
    """
    Calculates evaluation metrics.

    Args:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): Actual values.

    Returns:
        dict: Dictionary containing MSE, RMSE, MAE, and R-squared.
    """
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


# In[105]:


# Initialize model
model = MarketPredictor(
    num_features=X_train.shape[2],
    seq_length=config['seq_length'],
    d_model=config['model_dim'],
    nhead=config['num_heads'],
    num_layers=config['num_layers'],
    forecast_horizon=config['forecast_horizon'],
    dropout=config['dropout']
).to(config['device'])


# In[109]:


# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Define scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Initialize GradScaler for mixed precision
scaler = GradScaler()

# Early Stopping parameters
best_val_loss = float('inf')
epochs_no_improve = 0
early_stopping_patience = config['early_stopping_patience']


# In[110]:


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs, device, early_stopping_patience):
    """
    Trains the model with early stopping and learning rate scheduling.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (function): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (Scheduler): Learning rate scheduler.
        scaler (GradScaler): GradScaler for mixed precision.
        num_epochs (int): Number of epochs.
        device (torch.device): Device to run the model on.
        early_stopping_patience (int): Patience for early stopping.

    Returns:
        nn.Module: Trained model.
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_metrics = {'MSE': 0, 'MAE': 0}

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            prev_closing_price = inputs[:, -1, 0].unsqueeze(-1).to(device)  # Assuming 'Close' is index 3

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets, prev_closing_price)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                prev_closing_price = inputs[:, -1, 0].unsqueeze(-1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets, prev_closing_price)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model


# In[111]:


model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=total_loss,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    num_epochs=config['num_epochs'],
    device=config['device'],
    early_stopping_patience=config['early_stopping_patience']
)


# ### ***Evaluation and Visualization***

# In[113]:


scaler = StandardScaler()


# In[116]:


def evaluate_model(model, test_loader, criterion, device, scaler_close):
    """
    Evaluates the model on the test set and calculates metrics.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        criterion (function): Loss function.
        device (torch.device): Device to run the model on.
        scaler_close (Scaler): Scaler used for inverse transforming 'Close' column.

    Returns:
        dict: Metrics
        np.ndarray: Predicted prices
        np.ndarray: True prices
    """
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            prev_closing_price = inputs[:, -1, 0].unsqueeze(-1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets, prev_closing_price)
            test_loss += loss.item() * inputs.size(0)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

    # Concatenate all predictions and targets
    predicted_outputs = np.concatenate(all_predictions, axis=0)
    true_outputs = np.concatenate(all_targets, axis=0)

    # Reshape to (num_samples * forecast_horizon, 1)
    predicted_flat = predicted_outputs.reshape(-1, 1)
    true_flat = true_outputs.reshape(-1, 1)

    # Inverse transform the 'Close' prices
    inverse_pred = scaler_close.inverse_transform(predicted_flat).reshape(predicted_outputs.shape)
    inverse_true = scaler_close.inverse_transform(true_flat).reshape(true_outputs.shape)

    # Flatten for metric calculations
    inverse_pred_flat = inverse_pred.flatten()
    inverse_true_flat = inverse_true.flatten()

    # Calculate metrics
    metrics = calculate_metrics(inverse_true_flat, inverse_pred_flat)
    print(f"Metrics: {metrics}")

    return metrics, inverse_pred, inverse_true

# Run the evaluation
metrics, predicted_prices, true_prices = evaluate_model(
    model=model,
    test_loader=test_loader,
    criterion=total_loss,
    device=config['device'],
    scaler_close=scaler
)


# In[117]:


# Generate timestamps for plotting (this assumes you have access to the original dataset's timestamps)
test_start_idx = len(scaled_df) - len(predicted_prices)
test_timestamps = scaled_df.index[test_start_idx:]


# In[118]:


def plot_prices(timestamps, true_prices, predicted_prices, forecast_step=0, title='Predicted vs True Prices'):
    """
    Plots predicted and true prices over time for a specific forecast step.

    Args:
        timestamps (pd.DatetimeIndex): Timestamps for the data.
        true_prices (np.ndarray): Actual prices (num_samples, forecast_horizon).
        predicted_prices (np.ndarray): Predicted prices (num_samples, forecast_horizon).
        forecast_step (int): Specific forecast step to plot (0 to forecast_horizon-1).
        title (str): Title of the plot.
    """
    # Ensure that timestamps and prices have the same length
    min_len = min(len(timestamps), true_prices.shape[0], predicted_prices.shape[0])

    # Slice the arrays to have the same length
    timestamps = timestamps[-min_len:]
    true_step = true_prices[-min_len:, forecast_step]
    pred_step = predicted_prices[-min_len:, forecast_step]

    plt.figure(figsize=(14, 8))
    plt.plot(timestamps, true_step, label='True Prices', color='black')
    plt.plot(timestamps, pred_step, label='Predicted Prices', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Prices')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example: Plotting the first forecast step
plot_prices(
    timestamps=test_timestamps,
    true_prices=true_prices,
    predicted_prices=predicted_prices,
    forecast_step=0,
    title='Predicted vs True Prices (Forecast Step 1)'
)


# In[119]:


def plot_horizon_end(timestamps, true_prices, predicted_prices, forecast_step=-1):
    """
    Plots the last price in the forecast horizon.

    Args:
        timestamps (pd.DatetimeIndex): Timestamps.
        true_prices (np.ndarray): Actual prices (num_samples, forecast_horizon).
        predicted_prices (np.ndarray): Predicted prices (num_samples, forecast_horizon).
        forecast_step (int): Specific forecast step to plot (negative indexing supported).
    """

    # Ensure that timestamps and prices have the same length
    min_len = min(len(timestamps), true_prices.shape[0], predicted_prices.shape[0])
    timestamps = timestamps[-min_len:]
    true_prices = true_prices[-min_len:]
    predicted_prices = predicted_prices[-min_len:]

    plt.figure(figsize=(14, 8))
    plt.plot(timestamps, true_prices[:, forecast_step], label='True Prices (Horizon End)', color='black')
    plt.plot(timestamps, predicted_prices[:, forecast_step], label='Predicted Prices (Horizon End)', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Prices')
    plt.title('Predicted vs True Prices Over Time (Horizon End)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plotting the last forecast step
plot_horizon_end(test_timestamps, true_prices, predicted_prices, forecast_step=-1)


# In[120]:


def plot_weekly_resampled(timestamps, true_prices, predicted_prices, forecast_step=0):
    """
    Plots weekly resampled true and predicted prices for a specific forecast step.

    Args:
        timestamps (pd.DatetimeIndex): Timestamps.
        true_prices (np.ndarray): Actual prices (num_samples, forecast_horizon).
        predicted_prices (np.ndarray): Predicted prices (num_samples, forecast_horizon).
        forecast_step (int): Specific forecast step to plot.
    """

    # Ensure that timestamps and prices have the same length
    min_len = min(len(timestamps), true_prices.shape[0], predicted_prices.shape[0])
    timestamps = timestamps[-min_len:]
    true_prices = true_prices[-min_len:]
    predicted_prices = predicted_prices[-min_len:]

    df_plot = pd.DataFrame({
        'True Prices': true_prices[:, forecast_step],
        'Predicted Prices': predicted_prices[:, forecast_step]
    }, index=timestamps)

    df_resampled = df_plot.resample('W').mean()

    plt.figure(figsize=(14, 8))
    plt.plot(df_resampled.index, df_resampled['True Prices'], label='True Prices', color='black')
    plt.plot(df_resampled.index, df_resampled['Predicted Prices'], label='Predicted Prices', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Prices')
    plt.title('Predicted vs True Prices Over Time (Weekly Resampled)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example: Plotting the first forecast step
plot_weekly_resampled(test_timestamps, true_prices, predicted_prices, forecast_step=0)


# In[121]:


def plot_smoothed_data(timestamps, true_prices, predicted_prices, forecast_step=0, window_size=10):
    """
    Plots smoothed true and predicted prices using rolling mean for a specific forecast step.

    Args:
        timestamps (pd.DatetimeIndex): Timestamps.
        true_prices (np.ndarray): Actual prices (num_samples, forecast_horizon).
        predicted_prices (np.ndarray): Predicted prices (num_samples, forecast_horizon).
        forecast_step (int): Specific forecast step to plot.
        window_size (int): Window size for rolling mean.
    """
    # Create DataFrame
    df_smoothed = pd.DataFrame({
        'True Prices Smoothed': pd.Series(true_prices[:, forecast_step]).rolling(window=window_size).mean(),
        'Predicted Prices Smoothed': pd.Series(predicted_prices[:, forecast_step]).rolling(window=window_size).mean()
    }, index=timestamps)

    plt.figure(figsize=(14, 8))
    plt.plot(df_smoothed.index, df_smoothed['True Prices Smoothed'], label='True Prices (Smoothed)', color='blue')
    plt.plot(df_smoothed.index, df_smoothed['Predicted Prices Smoothed'], label='Predicted Prices (Smoothed)', color='red')
    plt.xlabel('Time')
    plt.ylabel('Prices')
    plt.title(f'Predicted vs True Prices Over Time (Smoothed, Forecast Step {forecast_step+1})')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example: Plotting the first forecast step with window size 10
plot_smoothed_data(
    timestamps=test_timestamps,
    true_prices=true_prices,
    predicted_prices=predicted_prices,
    forecast_step=0,
    window_size=10
)


# In[122]:


def plot_raw_prices(timestamps, true_prices, predicted_prices, forecast_step=None):
    """
    Plots raw true and predicted prices over time.

    Args:
        timestamps (pd.DatetimeIndex): Timestamps.
        true_prices (np.ndarray): Actual prices (num_samples, forecast_horizon).
        predicted_prices (np.ndarray): Predicted prices (num_samples, forecast_horizon).
        forecast_step (int or None): Specific forecast step to plot (0 to forecast_horizon-1).
                                     If None, the function will plot the average across all forecast steps.
    """

    # Ensure that timestamps and prices have the same length
    min_len = min(len(timestamps), true_prices.shape[0], predicted_prices.shape[0])
    timestamps = timestamps[-min_len:]
    true_prices = true_prices[-min_len:]
    predicted_prices = predicted_prices[-min_len:]

    if forecast_step is not None:
        # Select the specific forecast step
        true_prices_to_plot = true_prices[:, forecast_step]
        predicted_prices_to_plot = predicted_prices[:, forecast_step]
    else:
        # Average across all forecast steps
        true_prices_to_plot = true_prices.mean(axis=1)
        predicted_prices_to_plot = predicted_prices.mean(axis=1)

    # Print sample values to diagnose the issue
    print("Sample True Prices:", true_prices_to_plot[:5])
    print("Sample Predicted Prices:", predicted_prices_to_plot[:5])

    plt.figure(figsize=(14, 8))
    plt.plot(timestamps, true_prices_to_plot, label='True Prices', color='blue')
    plt.plot(timestamps, predicted_prices_to_plot, label='Predicted Prices', color='red')
    plt.xlabel('Time')
    plt.ylabel('Prices')
    plt.title('Predicted vs True Prices Over Time (Raw)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example: Plotting the raw data for the first forecast step
plot_raw_prices(test_timestamps, true_prices, predicted_prices, forecast_step=0)


# In[123]:


def plot_first_forecast_direction(
    timestamps, true_prices, predicted_prices, start_date, end_date, scaler_close, scaled_df
):
    """
    Plots the actual and predicted market direction (Up/Down) for the first forecasted point
    across all days within a specified date range.

    Args:
        timestamps (pd.DatetimeIndex): Timestamps corresponding to the test set.
        true_prices (np.ndarray): Actual prices (num_samples, forecast_horizon).
        predicted_prices (np.ndarray): Predicted prices (num_samples, forecast_horizon).
        start_date (str): Start date for the subset in 'YYYY-MM-DD'.
        end_date (str): End date for the subset in 'YYYY-MM-DD'.
        scaler_close (Scaler): Scaler used for inverse transforming 'Close' prices.
        scaled_df (pd.DataFrame): The scaled dataframe containing 'Close' prices.
    """
    subset_start = pd.to_datetime(start_date)
    subset_end = pd.to_datetime(end_date)

    # Inverse transform the 'Close' prices to get actual prices
    true_prices_inversed = scaler_close.inverse_transform(true_prices.reshape(-1, 1)).reshape(true_prices.shape)
    predicted_prices_inversed = scaler_close.inverse_transform(predicted_prices.reshape(-1, 1)).reshape(predicted_prices.shape)

    # To get the last closing price before each forecast
    # Assuming that the test set sequences are sequential, we can align them accordingly
    # Extract the 'Close' prices from the scaled_df for the test set
    test_start_idx = len(scaled_df) - len(predicted_prices) - config['forecast_horizon'] + 1
    last_closing_prices = scaled_df['Close'].iloc[test_start_idx : test_start_idx + len(predicted_prices)].values
    last_closing_prices = scaler_close.inverse_transform(last_closing_prices.reshape(-1, 1)).flatten()

    # Compute direction: 1 if price increases, else 0
    binary_predictions = (predicted_prices_inversed[:, 0] > last_closing_prices).astype(int)
    binary_targets = (true_prices_inversed[:, 0] > last_closing_prices).astype(int)

    # Create a DataFrame for easy handling
    df_direction = pd.DataFrame({
        'Timestamp': timestamps,
        'Actual Direction': binary_targets,
        'Predicted Direction': binary_predictions
    })

    # Apply the date filter
    mask = (df_direction['Timestamp'] >= subset_start) & (df_direction['Timestamp'] <= subset_end)
    subset_df = df_direction[mask]

    if subset_df.empty:
        print("No data available for the given date range.")
        return

    # Plotting all forecasted directions within the subset
    plt.figure(figsize=(14, 8))

    # Plot Actual Directions
    plt.scatter(
        subset_df['Timestamp'],
        subset_df['Actual Direction'],
        label='Actual Direction',
        color='blue',
        marker='o',
        alpha=0.6
    )

    # Plot Predicted Directions
    plt.scatter(
        subset_df['Timestamp'],
        subset_df['Predicted Direction'],
        label='Predicted Direction',
        color='orange',
        marker='x',
        alpha=0.6
    )

    # Enhance the plot
    plt.xlabel('Date')
    plt.ylabel('Direction (Up=1 / Down=0)')
    plt.title('Predicted vs Actual Market Direction (First Forecasted Point)')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.yticks([0, 1], ['Down', 'Up'])
    plt.tight_layout()
    plt.show()

    # Optional: Print summary statistics
    total = len(subset_df)
    correct_predictions = (subset_df['Actual Direction'] == subset_df['Predicted Direction']).sum()
    accuracy = correct_predictions / total * 100
    print(f"Total Predictions: {total}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")


# In[124]:


# After evaluation
metrics, predicted_prices, true_prices = evaluate_model(
    model=model,
    test_loader=test_loader,
    criterion=total_loss,
    device=config['device'],
    scaler_close=scaler
)

# Generate timestamps for plotting
test_start_idx = len(scaled_df) - len(predicted_prices)
test_timestamps = scaled_df.index[test_start_idx:]

# Call the corrected plotting function
plot_first_forecast_direction(
    timestamps=test_timestamps,
    true_prices=true_prices,
    predicted_prices=predicted_prices,
    start_date=config['subset_start_date'],
    end_date=config['subset_end_date'],
    scaler_close=scaler,
    scaled_df=df  # Pass the original scaled dataframe
)


# In[ ]:


def plot_percent_changes(true_prices, predicted_prices):
    """
    Plots actual vs predicted percent changes.

    Args:
        true_prices (np.ndarray): Actual prices.
        predicted_prices (np.ndarray): Predicted prices.
    """
    actual_pct_change = (true_prices[1:] - true_prices[:-1]) / true_prices[:-1]
    predicted_pct_change = (predicted_prices[1:] - predicted_prices[:-1]) / predicted_prices[:-1]

    plt.figure(figsize=(14, 8))
    plt.plot(actual_pct_change, label="Actual Percent Change", color='blue')
    plt.plot(predicted_pct_change, label="Predicted Percent Change", color='orange')
    plt.title("Actual vs Predicted Percent Changes")
    plt.xlabel("Time Step")
    plt.ylabel("Percent Change")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print metrics for percent changes
    metrics_pct = calculate_metrics(actual_pct_change, predicted_pct_change)
    print(f"Percent Change Metrics: {metrics_pct}")

plot_percent_changes(true_prices, predicted_prices)


# ### ***Hyperparameter Tuning***

# In[89]:


def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Validation loss.
    """
    # Suggest hyperparameters
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    nhead = trial.suggest_categorical('nhead', [4, 8, 16])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)

    # Initialize model
    model = MarketPredictor(
        num_features=X_train.shape[2],
        seq_length=config['seq_length'],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        forecast_horizon=config['forecast_horizon'],
        dropout=dropout
    ).to(config['device'])

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False)
    scaler_opt = GradScaler()

    # Training parameters
    epochs = 20
    best_val_loss_opt = float('inf')
    epochs_no_improve_opt = 0
    early_stop_patience_opt = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            # Assuming 'Close' is at index 3
            prev_closing_price = inputs[:, -1, 0].unsqueeze(-1).to(config['device'])

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = total_loss(outputs, targets, prev_closing_price)

            scaler_opt.scale(loss).backward()
            scaler_opt.step(optimizer)
            scaler_opt.update()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                prev_closing_price = inputs[:, -1, 0].unsqueeze(-1).to(config['device'])
                outputs = model(inputs)
                loss = total_loss(outputs, targets, prev_closing_price)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss_opt:
            best_val_loss_opt = val_loss
            epochs_no_improve_opt = 0
        else:
            epochs_no_improve_opt += 1
            if epochs_no_improve_opt >= early_stop_patience_opt:
                break

    return best_val_loss_opt


# In[ ]:


# Initialize Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


# In[ ]:




