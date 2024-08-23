import os
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset
import random

def handle_nan_and_scale(data, scaling_method="standard"):
    # Check for entirely NaN columns and handle them
    if np.all(np.isnan(data), axis=0).any():
        print("Warning: Some columns are entirely NaN. Replacing with zeros.")
        data[:, np.all(np.isnan(data), axis=0)] = 0  # Replace entirely NaN columns with 0

    # Replace remaining NaNs with the mean of the respective columns
    col_mean = np.nanmean(data, axis=0)
    nan_mask = np.isnan(data)
    data[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    # Choose scaling method
    if scaling_method == "standard":
        scaler = StandardScaler()
    elif scaling_method == "minmax":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError("Invalid scaling method. Choose 'standard' or 'minmax'.")

    # Apply scaling
    scaled_data = scaler.fit_transform(data)

    return scaled_data

def time_warp(sensor_data, warp_factor=0.2):
    factor = 1 + np.random.uniform(-warp_factor, warp_factor)
    original_length = sensor_data.shape[0]
    new_length = int(original_length * factor)
    return np.interp(
        np.linspace(0, original_length, new_length),
        np.arange(original_length),
        sensor_data
    )

def random_shift(sensor_data, shift_range=5):
    shift = np.random.randint(-shift_range, shift_range)
    return np.roll(sensor_data, shift, axis=0)

def invert_sensor_axis(sensor_data, axis='x'):
    if axis == 'x':
        sensor_data[:, 0] *= -1
    elif axis == 'y':
        sensor_data[:, 1] *= -1
    elif axis == 'z':
        sensor_data[:, 2] *= -1
    return sensor_data

def rotate_sensor_data(sensor_data, axis='y', angle=15):
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

    rotated_data = sensor_data.dot(rotation_matrix.T)
    return rotated_data


def read_csv_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    data = {}
    for file in files:
        file_path = os.path.join(folder, file)
        data[file] = pd.read_csv(file_path)
    return data

def calculate_global_stats(data_dict):
    all_data = []
    for df in data_dict.values():
        all_data.append(df.values[:, 1:])  # Assuming first column is time/frame, and others are features

    all_data = np.concatenate(all_data, axis=0)
    global_mean = np.mean(all_data, axis=0)
    global_std = np.std(all_data, axis=0)

    return global_mean, global_std

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

def scale_skeleton(skeleton, scale_factors=(1.0, 1.0, 1.0)):
    scale_matrix = np.diag(scale_factors)
    skeleton_reshaped = skeleton.reshape(-1, 3)
    scaled_skeleton = skeleton_reshaped.dot(scale_matrix.T)
    return scaled_skeleton.reshape(skeleton.shape)

def translate_skeleton(skeleton, translation=(0.1, 0.1, 0.1)):
    translation = np.array(translation)
    skeleton_reshaped = skeleton.reshape(-1, 3)
    translated_skeleton = skeleton_reshaped + translation
    return translated_skeleton.reshape(skeleton.shape)

def add_sensor_noise(sensor_data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, sensor_data.shape)
    return sensor_data + noise

def add_noise(skeleton, noise_level=0.01):
    noise = np.random.normal(0, noise_level, skeleton.shape)
    return skeleton + noise

def flip_skeleton(skeleton):
    flip_matrix = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    skeleton_reshaped = skeleton.reshape(-1, 3)
    flipped_skeleton = skeleton_reshaped.dot(flip_matrix.T)
    return flipped_skeleton.reshape(skeleton.shape)

def random_joint_dropout(skeleton, dropout_prob=0.1):
    mask = np.random.binomial(1, 1 - dropout_prob, skeleton.shape)
    return skeleton * mask

class SlidingWindowDataset(Dataset):
    def __init__(self, skeleton_data, sensor1_data, sensor2_data, common_files, window_size, overlap, label_encoder, 
                 augment=True, sensor_augment=False, skeleton_augment=True, scaling="minmax"):
        self.skeleton_data = skeleton_data
        self.sensor1_data = sensor1_data
        self.sensor2_data = sensor2_data
        self.common_files = list(common_files)
        self.window_size = window_size
        self.overlap = overlap
        self.label_encoder = label_encoder
        self.augment = augment
        self.scaling = scaling
        self.sensor_augment = sensor_augment
        self.skeleton_augment = skeleton_augment

        # Initialize lists to hold separate data
        self.skeleton_windows, self.sensor1_windows, self.sensor2_windows, self.labels = self._create_windows()

    def _create_windows(self):
        skeleton_windows = []
        sensor1_windows = []
        sensor2_windows = []
        skeleton_masks = []
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

                skeleton_window = skeleton_df.iloc[start:end, :].values
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

                # Create a mask for NaN values in the skeleton data only
                skeleton_window = handle_nan_and_scale(skeleton_window, scaling_method="minmax")
                sensor1_window = handle_nan_and_scale(sensor1_window, scaling_method="minmax")
                sensor2_window = handle_nan_and_scale(sensor2_window, scaling_method="minmax")

                # Apply augmentations if enabled
                if self.augment:
                    if self.skeleton_augment:
                        rotate_skeleton_window = rotate_skeleton(skeleton_window, axis='y', angle=random.uniform(-15, 15))
                        skeleton_windows.append(rotate_skeleton_window)
                        sensor1_windows.append(sensor1_window)
                        sensor2_windows.append(sensor2_window)
                        labels.append(label)
                        rotate_skeleton_window = rotate_skeleton(skeleton_window, axis='x', angle=random.uniform(-15, 15))
                        skeleton_windows.append(rotate_skeleton_window)
                        sensor1_windows.append(sensor1_window)
                        sensor2_windows.append(sensor2_window)
                        labels.append(label)
                        rotate_skeleton_window = rotate_skeleton(skeleton_window, axis='z', angle=random.uniform(-15, 15))
                        skeleton_windows.append(rotate_skeleton_window)
                        sensor1_windows.append(sensor1_window)
                        sensor2_windows.append(sensor2_window)
                        labels.append(label)
                        rotate_skeleton_window = rotate_skeleton(skeleton_window, axis='x', angle=random.uniform(-15, 15))
                        rotate_skeleton_window = rotate_skeleton(rotate_skeleton_window, axis='y', angle=random.uniform(-15, 15))
                        skeleton_windows.append(rotate_skeleton_window)
                        sensor1_windows.append(sensor1_window)
                        sensor2_windows.append(sensor2_window)
                        labels.append(label)
                        rotate_skeleton_window = rotate_skeleton(skeleton_window, axis='x', angle=random.uniform(-15, 15))
                        rotate_skeleton_window = rotate_skeleton(rotate_skeleton_window, axis='z', angle=random.uniform(-15, 15))
                        skeleton_windows.append(rotate_skeleton_window)
                        sensor1_windows.append(sensor1_window)
                        sensor2_windows.append(sensor2_window)
                        labels.append(label)
                        rotate_skeleton_window = rotate_skeleton(skeleton_window, axis='y', angle=random.uniform(-15, 15))
                        rotate_skeleton_window = rotate_skeleton(rotate_skeleton_window, axis='z', angle=random.uniform(-15, 15))
                        skeleton_windows.append(rotate_skeleton_window)
                        sensor1_windows.append(sensor1_window)
                        sensor2_windows.append(sensor2_window)
                        labels.append(label)
                        rotate_skeleton_window = rotate_skeleton(skeleton_window, axis='x', angle=random.uniform(-15, 15))
                        rotate_skeleton_window = rotate_skeleton(rotate_skeleton_window, axis='y', angle=random.uniform(-15, 15))
                        rotate_skeleton_window = rotate_skeleton(rotate_skeleton_window, axis='z', angle=random.uniform(-15, 15))
                        skeleton_windows.append(rotate_skeleton_window)
                        sensor1_windows.append(sensor1_window)
                        sensor2_windows.append(sensor2_window)
                        labels.append(label)
                        noise_skeleton_window = add_noise(skeleton_window, noise_level=0.01)
                        skeleton_windows.append(noise_skeleton_window)
                        sensor1_windows.append(sensor1_window)
                        sensor2_windows.append(sensor2_window)
                        labels.append(label)
                        if random.random() > 0.5:
                            flip_skeleton_window = flip_skeleton(skeleton_window)
                            skeleton_windows.append(flip_skeleton_window)
                            sensor1_windows.append(sensor1_window)
                            sensor2_windows.append(sensor2_window)
                            labels.append(label)
                    if self.sensor_augment:
                        axes = ['x', 'y', 'z']
                        angles = [random.uniform(-15, 15) for _ in range(3)]  # Random angle for each axis

                        for axis1 in axes:
                            for axis2 in axes:
                                # Rotate sensor1 along axis1 and sensor2 along axis2
                                rotate_sensor1_window = rotate_skeleton(sensor1_window, axis=axis1, angle=angles[axes.index(axis1)])
                                rotate_sensor2_window = rotate_skeleton(sensor2_window, axis=axis2, angle=angles[axes.index(axis2)])

                                skeleton_windows.append(skeleton_window)
                                sensor1_windows.append(rotate_sensor1_window)
                                sensor2_windows.append(rotate_sensor2_window)
                                labels.append(label)
                        
                        augmented_sensor1 = add_sensor_noise(sensor1_window, noise_level=0.01)
                        skeleton_windows.append(skeleton_window)
                        sensor1_windows.append(augmented_sensor1)
                        sensor2_windows.append(sensor2_window)
                        labels.append(label)

                        augmented_sensor2 = add_sensor_noise(sensor2_window, noise_level=0.01)
                        skeleton_windows.append(skeleton_window)
                        sensor1_windows.append(sensor1_window)
                        sensor2_windows.append(augmented_sensor2)
                        labels.append(label)

                # Create a mask for NaN values in the skeleton data only
                skeleton_window = handle_nan_and_scale(skeleton_window, scaling_method="minmax")
                sensor1_window = handle_nan_and_scale(sensor1_window, scaling_method="minmax")
                sensor2_window = handle_nan_and_scale(sensor2_window, scaling_method="minmax")
                
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
        return (
            torch.tensor(skeleton_window, dtype=torch.float32),
            torch.tensor(sensor1_window, dtype=torch.float32),
            torch.tensor(sensor2_window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )
