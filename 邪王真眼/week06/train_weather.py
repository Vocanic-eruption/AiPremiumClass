import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import copy
import uuid
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from 邪王真眼.models.rnn import RNNModel
from 邪王真眼.week06.weather_dataset import WeatherStationDataset

uuid = uuid.uuid4().hex[:8]


def main():
    # hyperparameter
    lr = 0.001
    epochs = 50
    bs = 1024
    weight_decay = 1e-4
    model_type = 'gru' # 'rnn' 'lstm' 'gru' 'birnn'
    label_days = 1
    input_days = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(f'./邪王真眼/week06/result/weather')

    # dataset
    dataset = WeatherStationDataset(
        input_days=input_days,
        label_days=label_days,
        min_station_samples=1000,
        val_ratio=0.3
    )

    baseline_mae = dataset.baseline_mae
    
    train_dataset, val_dataset = dataset.get_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    # model
    model = RNNModel(model_type=model_type, input_size=3, hidden_size=128, num_classes=3, num_layers=3, dropout=0.3).to(device)
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / epochs) ** 0.9
        )
    
    # train
    best_val_mae = float('inf')
    best_model = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        batch = 1

        print(f"\nEpoch {epoch+1}/{epochs}")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            predictions = []
            current_input = inputs.clone()
            
            for _ in range(label_days):
                pred = model(current_input)
                predictions.append(pred.unsqueeze(1))
                
                current_input = torch.cat([
                    current_input[:, 1:, :],
                    pred.unsqueeze(1)
                ], dim=1)
            
            predictions = torch.cat(predictions, dim=1)

            loss = criterion(predictions, labels)
            mae = mae_loss(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mae += mae.item()

            print(f'\rBatch [{batch}/{len(train_loader)}] Loss: {running_loss/batch:.4f} | MAE: {100*(running_mae/batch):.4f}%', end='', flush=True)
            batch += 1

        writer.add_scalar(f'{model_type}_input{input_days}_predict{label_days}_Loss_{uuid}_BaselineMAE_{100*baseline_mae:.4f}%/train', running_loss/batch, epoch)
        writer.add_scalar(f'{model_type}_input{input_days}_predict{label_days}_MAE_{uuid}_BaselineMAE_{100*baseline_mae:.4f}%/train', running_mae/batch, epoch)

        scheduler.step()

        # val
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                predictions = []
                current_input = inputs.clone()
                
                for day in range(label_days):
                    pred = model(current_input)
                    predictions.append(pred.unsqueeze(1))
                    current_input = torch.cat([
                        current_input[:, 1:, :],
                        pred.unsqueeze(1)
                    ], dim=1)
                
                predictions = torch.cat(predictions, dim=1)
                val_loss += criterion(predictions, labels).item()
                val_mae += mae_loss(predictions, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        writer.add_scalar(f'{model_type}_input{input_days}_predict{label_days}_Loss_{uuid}_BaselineMAE_{100*baseline_mae:.4f}%/val', avg_val_loss, epoch)
        writer.add_scalar(f'{model_type}_input{input_days}_predict{label_days}_MAE_{uuid}_BaselineMAE_{100*baseline_mae:.4f}%/val', avg_val_mae, epoch)

        print(f'\n  VAL:   MSE: {avg_val_loss:.4f} | MAE: {100*avg_val_mae:.4f}%')

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
            print(f'  VAL:   best model!')

        model.train()

    # save model
    save_path = f"./邪王真眼/week06/result/weather/weather_{model_type}_input{input_days}_predict{label_days}_MAE_{100*best_val_mae:.4f}%_BaselineMAE_{100*baseline_mae:.4f}%.pth"
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(best_model.state_dict(), save_path)

if __name__ == "__main__":
    main()