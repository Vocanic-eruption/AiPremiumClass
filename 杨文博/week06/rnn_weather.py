import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class Config:
    """Centralized configuration management using constants"""
    TIME_STEPS = 5
    HIDDEN_SIZE = 2
    INPUT_SIZE = 2
    OUTPUT_SIZE = 5
    NUM_LAYERS = 2
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 20
    TEST_SIZE = 0.4
    VAL_TEST_SPLIT = 0.5
    RANDOM_STATE = 41
    FILE_PATH = r"D:\work\code\practice\home_work\杨文博\week06\weather_data\Summary of Weather.csv"


class IDataLoader(ABC):
    """Abstract base class for data loading"""

    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pass


class WeatherDataLoader(IDataLoader):
    """Concrete implementation for weather data loading"""

    def __init__(self, file_path: str = 'Summary of Weather.csv'):
        self.file_path = file_path

    def _preprocess_data(self, max_temp: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess time series data into sequences"""
        seq_length = Config.TIME_STEPS * Config.INPUT_SIZE
        n_groups = len(max_temp) - seq_length - Config.OUTPUT_SIZE + 1

        X = np.array([max_temp[i:i + seq_length] for i in range(n_groups)])
        y = np.array([max_temp[i + seq_length:i + seq_length + Config.OUTPUT_SIZE] for i in range(n_groups)])

        X = X.reshape(-1, Config.TIME_STEPS, Config.INPUT_SIZE)
        y = y.reshape(-1, Config.OUTPUT_SIZE)

        return X, y

    def _create_dataloaders(self, X: np.ndarray, y: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders"""
        # Split data
        x_train, x_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp,
            test_size=Config.VAL_TEST_SPLIT,
            random_state=Config.RANDOM_STATE
        )

        # Convert to tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Create datasets and dataloaders
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        return train_loader, val_loader, test_loader

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and preprocess data"""
        weather_data = pd.read_csv(self.file_path, low_memory=False)
        max_temp = weather_data['MaxTemp']
        X, y = self._preprocess_data(max_temp)
        return self._create_dataloaders(X, y)
class IModel(ABC):
    """Abstract base class for models"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class WeatherRNN(IModel, nn.Module):
    """RNN model for weather forecasting"""

    def __init__(self):
        super(WeatherRNN, self).__init__()
        self.hidden_size = Config.HIDDEN_SIZE
        self.num_layers = Config.NUM_LAYERS

        # Define RNN layer
        self.rnn = nn.RNN(
            input_size=Config.INPUT_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            batch_first=True
        )

        # Define fully connected layer
        self.fc = nn.Linear(Config.HIDDEN_SIZE, Config.OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # RNN forward pass
        out, _ = self.rnn(x, h0)

        # Select last time step output
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)
        return out
class ITrainer(ABC):
    """Abstract base class for model training"""

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def validate(self) -> float:
        pass


class RNNTrainer(ITrainer):
    """Concrete implementation for RNN training"""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            writer: SummaryWriter,
            device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            batch_y = batch_y.reshape(-1, Config.OUTPUT_SIZE)

            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def train(self) -> None:
        """Train the model"""
        for epoch in range(Config.EPOCHS):
            train_loss = self._train_epoch(epoch)
            val_loss = self.validate()

            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)

            print(f'Epoch {epoch + 1}/{Config.EPOCHS} - '
                  f'Train Loss: {train_loss:.4f} - '
                  f'Val Loss: {val_loss:.4f}')

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_y = batch_y.reshape(-1, Config.OUTPUT_SIZE)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)
class Experiment:
    """Orchestrates the entire experiment"""

    def __init__(self, data_loader: IDataLoader, model: IModel):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter('runs/weather_rnn_experiment')
        self.data_loader = data_loader
        self.model = model.to(self.device)

    def run(self):
        """Run the complete experiment"""
        # Load data
        train_loader, val_loader, test_loader = self.data_loader.load_data()

        # Train model
        trainer = RNNTrainer(self.model, train_loader, val_loader, self.writer, self.device)
        trainer.train()

        # Clean up
        self.writer.close()
experiment = Experiment(WeatherDataLoader(Config.FILE_PATH), WeatherRNN())
experiment.run()