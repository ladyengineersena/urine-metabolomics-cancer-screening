"""
Deep Learning Models for Metabolomics

MLP, 1D-CNN, and TabNet implementations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MetabolomicsDataset(Dataset):
    """PyTorch dataset for metabolomics data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class MLP(nn.Module):
    """Multi-Layer Perceptron for metabolomics classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for metabolite sequences."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        num_filters: list = [64, 128, 256],
        kernel_sizes: list = [3, 5, 7],
        dropout: float = 0.3
    ):
        super(CNN1D, self).__init__()
        
        # Reshape input to (batch, 1, features) for 1D conv
        self.conv_layers = nn.ModuleList()
        prev_channels = 1
        
        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(prev_channels, num_filter, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(num_filter),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout)
                )
            )
            prev_channels = num_filter
        
        # Calculate flattened size (approximate)
        flattened_size = prev_channels * (input_dim // (2 ** len(num_filters)))
        
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepModelTrainer:
    """Trainer for deep learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        return {
            'loss': total_loss / len(train_loader),
            'acc': correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'acc': correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        """Train model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['acc'])
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        self.model.load_state_dict(self.best_model_state)
                        break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train Acc: {train_metrics['acc']:.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val Acc: {val_metrics['acc']:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train Acc: {train_metrics['acc']:.4f}")
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        dataset = MetabolomicsDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Get prediction probabilities."""
        self.model.eval()
        dataset = MetabolomicsDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        probabilities = []
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)

