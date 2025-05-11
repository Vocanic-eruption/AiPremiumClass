import os
import numpy as np
import pandas as pd

from typing import Tuple, Any, Dict
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class Weather(Dataset):
    def __init__(self, cfg: Dict[str, Any]):
        self.input_days = cfg.get('input_days')
        self.label_days = cfg.get('label_days')
        self.min_station_samples = cfg.get('min_station_samples')
        self.window_size = self.input_days + self.label_days
        self.val_ratio = cfg.get('val_ratio')
        
        self.data_path = os.path.join(cfg.get("data_root"), 'Summary of Weather.csv')
        raw_df = pd.read_csv(self.data_path)
        self.df = self._preprocess_data(raw_df)
        
        self.station_samples = self._generate_station_samples()
        
        total_samples = sum(len(samples) for samples in self.station_samples.values())
        if total_samples < self.min_station_samples:
            raise ValueError(f"初始化失败: 总样本数{total_samples}小于最小阈值{self.min_station_samples}")
        
        self._split_train_val()

        self.baseline_mae = self._calculate_baseline_mae()
        print(f"基准MAE(用最后一个输入时间步的值作为未来所有时间步的预测): {100*self.baseline_mae:.4f}%")
    
    def _calculate_baseline_mae(self):
        total_mae = 0.0
        count = 0
        
        for station, samples in self.station_samples.items():
            for sample in samples:
                last_input = sample[self.input_days-1]
                labels = sample[self.input_days:]
                
                mae = np.mean(np.abs(labels - last_input))
                total_mae += mae
                count += 1
                
        return total_mae / count if count > 0 else 0.0
        
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        dtype_spec = {'STA': str, 'Date': str, 'MaxTemp': float, 'MinTemp': float, 'MeanTemp': float}
        df = pd.read_csv(self.data_path, dtype=dtype_spec, low_memory=False)
        df = df[['STA', 'Date', 'MaxTemp', 'MinTemp', 'MeanTemp']].copy()
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df.sort_values(['STA', 'Date'], inplace=True)
        df = df.reset_index(drop=True)
        
        temp_cols = ['MaxTemp', 'MinTemp', 'MeanTemp']
        scalers = {}
        groups = df.groupby('STA')
        normalized_dfs = []
        
        for station, group in groups:
            scaler = MinMaxScaler()
            group[temp_cols] = scaler.fit_transform(group[temp_cols])
            scalers[station] = scaler
            normalized_dfs.append(group)
        
        df = pd.concat(normalized_dfs, ignore_index=True)
        self.scalers = scalers
        
        df[temp_cols] = df.groupby('STA')[temp_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both'))
        
        return df
    
    def _generate_station_samples(self) -> dict:
        station_samples = {}
        
        for station, group in self.df.groupby('STA'):
            if len(group) < self.window_size:
                continue
                
            samples = []
            temp_data = group[['MaxTemp', 'MinTemp', 'MeanTemp']].values
            
            for i in range(len(group) - self.window_size + 1):
                window = temp_data[i:i+self.window_size]
                samples.append(window)
            
            station_samples[station] = np.array(samples)
        
        return station_samples
    
    def _split_train_val(self):
        all_stations = list(self.station_samples.keys())
        np.random.shuffle(all_stations)
        
        val_size = int(len(all_stations) * self.val_ratio)
        val_stations = set(all_stations[:val_size])
        
        self.train_samples = []
        self.val_samples = []
        
        for station, samples in self.station_samples.items():
            if station in val_stations:
                self.val_samples.extend(samples)
            else:
                self.train_samples.extend(samples)
        
        self.train_samples = np.array(self.train_samples)
        self.val_samples = np.array(self.val_samples)
    
    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset = StationSubset(self.train_samples, self.input_days, self.label_days)
        val_dataset = StationSubset(self.val_samples, self.input_days, self.label_days)
        return train_dataset, val_dataset


class StationSubset(Dataset):
    def __init__(self, samples: np.ndarray, input_days: int, label_days: int):
        self.samples = samples
        self.input_days = input_days
        self.label_days = label_days
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        window = self.samples[idx]
        x = window[:self.input_days]
        y = window[self.input_days:]
        return {'x': x.astype(np.float32), 'label': y.astype(np.float32)}
