"""
Dataset_AirQuality
------------------
Reads one-location air quality CSV files as a regular multivariate time series:
  - index = timestamp (ts_utc)
  - columns = sensor/AQI features

location_key is treated as metadata. It is not pivoted into model channels.
"""
import os
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

AIR_QUALITY_FEATURE_COLS = [
    'pm25', 'pm10', 'no2', 'o3', 'so2', 'co',
    'aod', 'dust', 'uv_index', 'co2',
    'aqi'
]


class Dataset_AirQuality(Dataset):
    """
    Dataset for one-location air_quality.csv files.

    features='M'  -> input and output are all numeric air-quality features
    features='S'  -> input and output are only the target column
    features='MS' -> input is all features, output is only the target column

    Split: 70% train / 10% val / 20% test in chronological order.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='air_quality.csv',
                 target='aqi', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), sep=',')
        df_raw = df_raw.rename(columns={'ts_utc': 'date'})
        df_raw['date'] = pd.to_datetime(df_raw['date'], utc=True).dt.tz_localize(None)

        feature_cols = [c for c in AIR_QUALITY_FEATURE_COLS if c in df_raw.columns]
        if not feature_cols:
            raise ValueError(f"No supported feature columns found in {self.data_path}.")
        if self.target not in df_raw.columns:
            raise ValueError(f"Target column '{self.target}' not found in {self.data_path}.")

        if 'location_key' in df_raw.columns:
            locations = sorted(df_raw['location_key'].dropna().unique())
            if len(locations) > 1:
                raise ValueError(
                    f"{self.data_path} contains multiple locations: {locations}. "
                    "Use one location per file, or split/filter the file before training."
                )
            self.location_key = locations[0] if locations else None
        else:
            self.location_key = None

        ts_data = (
            df_raw[['date'] + feature_cols]
            .sort_values('date')
            .groupby('date', as_index=True)
            .mean(numeric_only=True)
            .sort_index()
            .ffill()
            .bfill()
        )

        if self.features == 'S':
            df_data = ts_data[[self.target]]
            df_target = ts_data[[self.target]]
        elif self.features == 'MS':
            df_data = ts_data[feature_cols]
            df_target = ts_data[[self.target]]
        else:
            df_data = ts_data[feature_cols]
            df_target = ts_data[feature_cols]

        self.variate_names = list(df_data.columns)
        self.target_names = list(df_target.columns)
        self.n_variates = len(self.variate_names)
        self.n_targets = len(self.target_names)

        n = len(ts_data)
        if n < self.seq_len + self.pred_len:
            raise ValueError(
                f"Data has only {n} time steps, "
                f"but at least {self.seq_len + self.pred_len} are required "
                f"(seq_len={self.seq_len} + pred_len={self.pred_len}). "
                "Add more rows or reduce seq_len/pred_len."
            )

        num_train = int(n * 0.7)
        num_val = int(n * 0.1)

        border1s = [
            0,
            max(0, num_train - self.seq_len),
            max(0, num_train + num_val - self.seq_len),
        ]
        border2s = [num_train, num_train + num_val, n]

        b1 = border1s[self.set_type]
        b2 = border2s[self.set_type]

        x_values = df_data.values.astype(np.float32)
        y_values = df_target.values.astype(np.float32)

        train_x = x_values[border1s[0]:border2s[0]]
        train_y = y_values[border1s[0]:border2s[0]]
        if len(train_x) == 0:
            raise ValueError(
                f"No train data available because the file has only {n} time steps. "
                "Add more timestamps to air_quality.csv."
            )

        if self.scale:
            self.scaler_x.fit(train_x)
            self.scaler_y.fit(train_y)
            data_x = self.scaler_x.transform(x_values)
            data_y = self.scaler_y.transform(y_values)
            self.scaler = self.scaler_y
        else:
            data_x = x_values
            data_y = y_values

        ts_slice = ts_data.index[b1:b2]
        self.date_index = ts_slice
        df_stamp = pd.DataFrame({'date': ts_slice})
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop('date', axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq
            ).transpose(1, 0)
        else:
            data_stamp = np.zeros((b2 - b1, 1))

        self.data_x = data_x[b1:b2]
        self.data_y = data_y[b1:b2]
        self.data_stamp = data_stamp

    def make_time_features(self, dates):
        df_stamp = pd.DataFrame({'date': pd.to_datetime(dates)})
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            return df_stamp.drop('date', axis=1).values
        if self.timeenc == 1:
            return time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq
            ).transpose(1, 0)
        return np.zeros((len(df_stamp), 1))

    def get_future_forecast_sample(self):
        if len(self.data_x) < self.seq_len or len(self.data_y) < self.label_len:
            raise ValueError("Not enough data to build the final future forecast sample.")

        offset = to_offset(self.freq)
        last_date = self.date_index[-1]
        future_dates = pd.date_range(
            start=last_date + offset,
            periods=self.pred_len,
            freq=offset,
        )
        label_dates = self.date_index[-self.label_len:]
        decoder_dates = label_dates.append(future_dates)

        return (
            self.data_x[-self.seq_len:],
            self.data_y[-self.label_len:],
            self.data_stamp[-self.seq_len:],
            self.make_time_features(decoder_dates),
            future_dates,
        )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def inverse_transform_x(self, data):
        return self.scaler_x.inverse_transform(data)
