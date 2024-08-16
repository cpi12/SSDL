import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

def read_csv_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    data = {}
    for file in files:
        file_path = os.path.join(folder, file)
        data[file] = pd.read_csv(file_path)
    return data

def handle_nans(df):
    df_filled = df.apply(lambda row: row.fillna(method='ffill'), axis=1)
    df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
    return df_filled

def determine_target_length(data_dict, percentage=0.95):
    max_length = max(len(df) for df in data_dict.values())
    target_length = int(max_length * percentage)
    return target_length

def normalize_length(df, target_length, padding_threshold=0.3):
    current_length = len(df)
    padding_length = target_length - current_length
    padding_ratio = padding_length / target_length

    if padding_ratio > padding_threshold:
        return None

    if padding_length < 0:
        df_normalized = df.iloc[:target_length, :].copy()
    else:
        last_valid_frame = df.iloc[-1:].copy()
        padding = pd.concat([last_valid_frame] * padding_length, ignore_index=True)
        df_normalized = pd.concat([df, padding], ignore_index=True)

    return df_normalized

def calculate_global_stats(data_dict):
    all_data = []
    for df in data_dict.values():
        all_data.append(df.values[:, 1:])  # Assuming first column is time/frame, and others are features

    all_data = np.concatenate(all_data, axis=0)
    global_mean = np.mean(all_data, axis=0)
    global_std = np.std(all_data, axis=0)

    return global_mean, global_std

def normalize_data(df, global_mean, global_std):
    normalized_df = (df.values[:, 1:] - global_mean) / (global_std + 1e-8)
    df.iloc[:, 1:] = normalized_df
    return df

def rotate_skeleton(skeleton, axis='y', angle=15):
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_angle, -sin_angle],
            [0, sin_angle, cos_angle]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis, choose from 'x', 'y', or 'z'")

    skeleton_reshaped = skeleton.reshape(-1, 3)
    rotated_skeleton = skeleton_reshaped.dot(rotation_matrix.T)
    rotated_skeleton = rotated_skeleton.reshape(skeleton.shape)
    
    return rotated_skeleton

class SlidingWindowDataset(Dataset):
    def __init__(self, skeleton_data, sensor1_data, sensor2_data, common_files, window_size, overlap, label_encoder, 
                 skeleton_global_mean=None, skeleton_global_std=None, sensor1_global_mean=None, sensor1_global_std=None, 
                 sensor2_global_mean=None, sensor2_global_std=None, augment=True):
        self.skeleton_data = skeleton_data
        self.sensor1_data = sensor1_data
        self.sensor2_data = sensor2_data
        self.common_files = list(common_files)
        self.window_size = window_size
        self.overlap = overlap
        self.label_encoder = label_encoder
        self.augment = augment
        self.skeleton_global_mean = skeleton_global_mean
        self.skeleton_global_std = skeleton_global_std
        self.sensor1_global_mean = sensor1_global_mean
        self.sensor1_global_std = sensor1_global_std
        self.sensor2_global_mean = sensor2_global_mean
        self.sensor2_global_std = sensor2_global_std

        # Determine if normalization should be applied
        self.normalize = all(val is not None for val in [
            skeleton_global_mean, skeleton_global_std, 
            sensor1_global_mean, sensor1_global_std, 
            sensor2_global_mean, sensor2_global_std
        ])

        # Initialize lists to hold separate data
        self.skeleton_windows, self.sensor1_windows, self.sensor2_windows, self.labels = self._create_windows()

    def normalize_window(self, skeleton_window, sensor1_window, sensor2_window):
        skeleton_window = (skeleton_window - self.skeleton_global_mean) / (self.skeleton_global_std + 1e-8)
        sensor1_window = (sensor1_window - self.sensor1_global_mean) / (self.sensor1_global_std + 1e-8)
        sensor2_window = (sensor2_window - self.sensor2_global_mean) / (self.sensor2_global_std + 1e-8)
        return skeleton_window, sensor1_window, sensor2_window

    def apply_random_augmentation(self, skeleton_window, sensor1_window, sensor2_window):
        augmentations = [
            lambda s, s1, s2: (rotate_skeleton(s, axis='y', angle=np.random.uniform(-10, 10)), s1, s2),
            lambda s, s1, s2: (rotate_skeleton(s, axis='x', angle=np.random.uniform(-10, 10)), s1, s2),
            lambda s, s1, s2: (rotate_skeleton(s, axis='z', angle=np.random.uniform(-10, 10)), s1, s2)
        ]
        # Randomly select one or more augmentations to apply
        chosen_augmentations = random.sample(augmentations, k=np.random.randint(1, len(augmentations) + 1))
        for aug in chosen_augmentations:
            skeleton_window, sensor1_window, sensor2_window = aug(skeleton_window, sensor1_window, sensor2_window)
        return skeleton_window, sensor1_window, sensor2_window

    def _create_windows(self):
        skeleton_windows = []
        sensor1_windows = []
        sensor2_windows = []
        labels = []
        step = self.window_size - self.overlap

        for file in self.common_files:
            skeleton_df = self.skeleton_data[file]
            sensor1_df = self.sensor1_data[file]
            sensor2_df = self.sensor2_data[file]

            activity_code = file.split('A')[1][:2].lstrip('0')
            label = self.label_encoder.transform([[activity_code]])[0]

            num_windows = (len(skeleton_df) - self.window_size) // step + 1
            for i in range(num_windows):
                start = i * step
                end = start + self.window_size
                if end > len(skeleton_df):
                    continue

                skeleton_window = skeleton_df.iloc[start:end, 1:].values
                sensor1_window = sensor1_df.iloc[start:end, -3:].values
                sensor2_window = sensor2_df.iloc[start:end, -3:].values

                if skeleton_window.shape[1] == 97:
                    skeleton_window = skeleton_window[:, 1:]  # Remove the first column to make it 96 features

                # Check for consistency in feature dimensions
                if skeleton_window.shape[1] != 96:
                    print(f"Skipping window with inconsistent features: Expected 96, but got {skeleton_window.shape[1]}")
                    continue
        
                if skeleton_window.shape[0] != self.window_size or sensor1_window.shape[0] != self.window_size or sensor2_window.shape[0] != self.window_size:
                    continue

                # Check for NaNs
                if np.isnan(skeleton_window).sum() > 0.5 * skeleton_window.size or \
                   np.isnan(sensor1_window).sum() > 0.5 * sensor1_window.size or \
                   np.isnan(sensor2_window).sum() > 0.5 * sensor2_window.size:
                    # Skip the window if more than 50% values are NaNs
                    continue

                # Replace any remaining NaNs with 0
                skeleton_window = np.nan_to_num(skeleton_window, nan=0.0)
                sensor1_window = np.nan_to_num(sensor1_window, nan=0.0)
                sensor2_window = np.nan_to_num(sensor2_window, nan=0.0)

                # Normalize the windows using global statistics if available
                if self.normalize:
                    skeleton_window, sensor1_window, sensor2_window = self.normalize_window(skeleton_window, sensor1_window, sensor2_window)

                if self.augment:
                    skeleton_window, sensor1_window, sensor2_window = self.apply_random_augmentation(skeleton_window, sensor1_window, sensor2_window)

                skeleton_windows.append(skeleton_window)
                sensor1_windows.append(sensor1_window)
                sensor2_windows.append(sensor2_window)
                labels.append(label)

        return skeleton_windows, sensor1_windows, sensor2_windows, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        skeleton_window = self.skeleton_windows[idx]
        sensor1_window = self.sensor1_windows[idx]
        sensor2_window = self.sensor2_windows[idx]
        label = self.labels[idx]
        return torch.tensor(skeleton_window, dtype=torch.float32), \
            torch.tensor(sensor1_window, dtype=torch.float32), \
            torch.tensor(sensor2_window, dtype=torch.float32), \
            torch.tensor(label, dtype=torch.float32)
