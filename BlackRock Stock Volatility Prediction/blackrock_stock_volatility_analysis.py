#voladity forecasting of BlackRock stock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from datetime import datetime, timedelta

class EnhancedNeuralNetwork(nn.Module):
    def __init__(self, inputs, hidden_layers, outputs, dropout_rate=0.2):
        super(EnhancedNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(inputs, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers)-1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], outputs))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_layers[0])
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layers[0](x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        for layer in self.layers[1:-1]:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.layers[-1](x)
        return x

class Normalize:
    
    def __init__(self):
        self.minx = None
        self.maxx = None
        
    def normalize(self, x):
        x = np.array(x)
        self.minx = min(x)
        self.maxx = max(x)
        return (x - self.minx) / (self.maxx - self.minx)
    
    def denormalize(self, x):
        return x * (self.maxx - self.minx) + self.minx

class VolatilityAnalyzer:
    def __init__(self, data_path, window_size=150, forecast_horizon=30):
        self.data = pd.read_csv(data_path)[::-1]
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.normalizer = Normalize()
        self.model = None
        self.history = None
        
    def prepare_data(self):
        # Calculate returns
        self.close_prices = self.data['adjClose'].values
        self.returns = self.close_prices[1:] / self.close_prices[:-1] - 1.0
        
        # Calculate historical volatility
        self.volatility = self._build_dataset(self.returns)
        self.normalized_volatility = self.normalizer.normalize(self.volatility)
        
        # Prepare train/test sets
        return self._build_train_test(self.normalized_volatility)
    
    def _build_dataset(self, percent):
        result = []
        for i in range(self.window_size, len(percent)):
            window = percent[i - self.window_size : i]
            # Calculate annualized volatility
            vol = np.std(window) * np.sqrt(252)  # Annualize daily volatility
            result.append(vol)
        return result
    
    def _build_train_test(self, vol, test_size=0.2):
        window = 100
        inputs = []
        outputs = []
        
        for i in range(window, len(vol) - self.forecast_horizon + 1):
            curr_input = vol[i - window : i]
            curr_output = vol[i : i + self.forecast_horizon]
            inputs.append(curr_input)
            outputs.append(curr_output)
        
        # Split into train/test
        split_idx = int(len(inputs) * (1 - test_size))
        train_in = torch.tensor(inputs[:split_idx], dtype=torch.float32)
        train_out = torch.tensor(outputs[:split_idx], dtype=torch.float32)
        test_in = torch.tensor(inputs[split_idx:], dtype=torch.float32)
        test_out = torch.tensor(outputs[split_idx:], dtype=torch.float32)
        
        return (train_in, train_out), (test_in, test_out)
    
    def train_model(self, hidden_layers=[128, 64], learning_rate=0.001, epochs=500, batch_size=32):
        (train_in, train_out), (test_in, test_out) = self.prepare_data()
        
        # Create model
        self.model = EnhancedNeuralNetwork(100, hidden_layers, self.forecast_horizon)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Create data loaders
        train_dataset = TensorDataset(train_in, train_out)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop with history
        self.history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(test_in)
                val_loss = criterion(val_output, test_out)
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss.item())
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss.item():.6f}')
    
    def predict_volatility(self, plot_results=True):
        with torch.no_grad():
            last_window = torch.tensor(self.normalized_volatility[-100:], dtype=torch.float32).unsqueeze(0)
            predicted = self.model(last_window)
        
        predicted_vol = self.normalizer.denormalize(predicted.numpy()[0])
        
        if plot_results:
            self._plot_forecast(predicted_vol)
        
        return predicted_vol
    
    def _plot_forecast(self, predicted_vol):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Volatility Forecast
        historical_vol = self.volatility[-100:]
        hist_x = list(range(len(historical_vol)))
        pred_x = list(range(len(historical_vol), len(historical_vol) + len(predicted_vol)))
        
        ax1.plot(hist_x, historical_vol, color='red', label='Historical')
        ax1.plot(pred_x, predicted_vol, color='limegreen', label='Forecast')
        ax1.set_title("Volatility Forecast for BlackRock")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Annualized Volatility")
        ax1.legend()
        
        # Plot 2: Training History
        ax2.plot(self.history['train_loss'], label='Training Loss')
        ax2.plot(self.history['val_loss'], label='Validation Loss')
        ax2.set_title("Model Training History")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self):
        """Calculate and display model performance metrics"""
        with torch.no_grad():
            _, (test_in, test_out) = self.prepare_data()
            predictions = self.model(test_in)
        
        # Denormalize predictions and actual values
        pred_vol = self.normalizer.denormalize(predictions.numpy())
        actual_vol = self.normalizer.denormalize(test_out.numpy())
        
        # Calculate metrics
        mse = mean_squared_error(actual_vol, pred_vol)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_vol.flatten(), pred_vol.flatten())
        
        print("\nModel Performance Metrics:")
        print(f"Root Mean Square Error: {rmse:.4f}")
        print(f"R-squared Score: {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_vol.flatten(), pred_vol.flatten(), alpha=0.5)
        plt.plot([actual_vol.min(), actual_vol.max()], [actual_vol.min(), actual_vol.max()], 'r--')
        plt.xlabel("Actual Volatility")
        plt.ylabel("Predicted Volatility")
        plt.title("Actual vs Predicted Volatility")
        plt.show()

# Usage example
if __name__ == "__main__":
    analyzer = VolatilityAnalyzer('Volatility_Data.csv')
    analyzer.train_model(epochs=500)
    predicted_volatility = analyzer.predict_volatility()
    analyzer.evaluate_model()