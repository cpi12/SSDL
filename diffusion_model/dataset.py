import os
import pandas as pd
from collections import defaultdict
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def to_one_hot(label, num_classes):
    """Convert a class index to one-hot encoded vector."""
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


def handle_nan_and_scale(data, scaling_method="standard"):
    # Check for entirely NaN columns and handle them
    if np.all(np.isnan(data), axis=0).any():
        # print("Warning: Some columns are entirely NaN. Replacing with zeros.")
        data[:, np.all(np.isnan(data), axis=0)] = 0  # Replace entirely NaN columns with 0

    # Replace remaining NaNs with the mean of the respective columns
    col_mean = np.nanmean(data, axis=0)
    nan_mask = np.isnan(data)
    data[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    # Scaling is removed in this example but you can add it back if necessary
    return data

def adjust_keypoints(skeleton_window, key_joint_indexes, joint_order):
    """
    Adjust the skeleton keypoints to follow the human structure order (head -> neck -> arms -> spine -> legs).
    
    Args:
        skeleton_window (numpy.ndarray): Skeleton data in shape (window_size, num_joints * 3).
        key_joint_indexes (list): List of key joints to use for ordering.
        joint_order (list): The desired order of key joints.
    
    Returns:
        numpy.ndarray: Adjusted skeleton data in shape (window_size, ordered_num_joints * 3).
    """
    adjusted_skeleton = []
    
    for joint_index in joint_order:
        if joint_index in key_joint_indexes:
            start_idx = key_joint_indexes.index(joint_index) * 3  # Find the starting index for (x, y, z) of this joint
            adjusted_skeleton.append(skeleton_window[:, start_idx:start_idx + 3])  # Append (x, y, z) of the joint
    
    return np.hstack(adjusted_skeleton)

def random_joint_dropout(skeleton, dropout_prob=0.1):
    mask = np.random.binomial(1, 1 - dropout_prob, skeleton.shape)
    return skeleton * mask

class SlidingWindowDataset(Dataset):
    def __init__(self, skeleton_data, sensor1_data, sensor2_data, common_files, window_size, overlap, label_encoder, 
                 augment=True, sensor_augment=False, skeleton_augment=True, scaling="minmax"):
        self.overlap = overlap
        self.augment = augment
        self.scaling = scaling
        self.window_size = window_size
        self.sensor1_data = sensor1_data
        self.sensor2_data = sensor2_data
        self.skeleton_data = skeleton_data
        self.label_encoder = label_encoder
        self.sensor_augment = sensor_augment
        self.common_files = list(common_files)
        self.skeleton_augment = skeleton_augment
        self.key_joint_indexes = [0, 2, 3, 5, 6, 7, 12, 13, 14, 18, 19, 20, 22, 23, 24, 26]

        # Joint order based on head -> neck -> arms -> spine -> legs
        self.joint_order = [26, 3, 5, 6, 7, 12, 13, 14, 2, 0, 18, 19, 20, 22, 23, 24]

        # Initialize lists to hold separate data
        self.skeleton_windows, self.sensor1_windows, self.sensor2_windows, self.labels = self._create_windows()

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
            num_classes = len(self.label_encoder.categories_[0]) 
            num_windows = (len(skeleton_df) - self.window_size) // step + 1
            for i in range(num_windows):
                start = i * step
                end = start + self.window_size
                if end > len(skeleton_df):
                    continue
                skeleton_window = skeleton_df.iloc[start:end, :].values
                sensor1_window = sensor1_df.iloc[start:end, -3:].values
                sensor2_window = sensor2_df.iloc[start:end, -3:].values

                if skeleton_window.shape[1] != 96:
                    skeleton_window = skeleton_df.iloc[start:end, 2:].values

                if skeleton_window.shape[1] == 97:
                    skeleton_window = skeleton_window[:, 1:]  # Remove the first column to make it 96 features

                # Check for consistency in feature dimensions
                if skeleton_window.shape[1] != 96:
                    continue

                joint_indices = np.array(self.key_joint_indexes)
                final_indices = np.concatenate([[i * 3, i * 3 + 1, i * 3 + 2] for i in joint_indices])
                skeleton_window = skeleton_window[:, final_indices]

                if skeleton_window.shape[0] != self.window_size or sensor1_window.shape[0] != self.window_size or sensor2_window.shape[0] != self.window_size:
                    continue

                # Apply keypoint adjustment (reorder joints as head -> neck -> arms -> spine -> legs)
                skeleton_window = adjust_keypoints(skeleton_window, self.key_joint_indexes, self.joint_order)

                skeleton_window = handle_nan_and_scale(skeleton_window, scaling_method=self.scaling)
                sensor1_window = handle_nan_and_scale(sensor1_window, scaling_method=self.scaling)
                sensor2_window = handle_nan_and_scale(sensor2_window, scaling_method=self.scaling)

                # Normalize to range [0, 1]
                skeleton_window = self._normalize_to_tensor(skeleton_window)

                # Append original windows after augmentation
                skeleton_windows.append(skeleton_window)
                sensor1_windows.append(sensor1_window)
                sensor2_windows.append(sensor2_window)
                labels.append(label)

        # Perform oversampling to ensure each class has 2000 samples
        num_classes = len(self.label_encoder.categories_[0])  # Get the number of classes from OneHotEncoder
        class_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            lbl_index = lbl.argmax()
            class_indices[lbl_index].append(idx)

        # Proceed with the oversampling process...
        skeleton_windows_oversampled = []
        sensor1_windows_oversampled = []
        sensor2_windows_oversampled = []
        labels_oversampled = []

        for lbl, indices in class_indices.items():
            num_samples = len(indices)
            if num_samples >= 1000:
                selected_indices = random.sample(indices, 1000)
            else:
                additional_samples_needed = 1000 - num_samples
                additional_indices = random.choices(indices, k=additional_samples_needed)
                selected_indices = indices + additional_indices

            for idx in selected_indices:
                skeleton_windows_oversampled.append(skeleton_windows[idx])
                sensor1_windows_oversampled.append(sensor1_windows[idx])
                sensor2_windows_oversampled.append(sensor2_windows[idx])

                # Convert the integer label back to one-hot encoding before appending
                one_hot_label = to_one_hot(lbl, num_classes)  # Convert lbl (int) to one-hot encoded vector
                labels_oversampled.append(one_hot_label)

        # Replace original lists with oversampled data
        self.skeleton_windows = skeleton_windows_oversampled
        self.sensor1_windows = sensor1_windows_oversampled
        self.sensor2_windows = sensor2_windows_oversampled
        self.labels = labels_oversampled

        return self.skeleton_windows, self.sensor1_windows, self.sensor2_windows, self.labels

    def __len__(self):
        return len(self.labels)

    def _normalize_to_tensor(self, data):
        """Converts data to a tensor and scales it between 0 and 1."""
        min_val, max_val = data.min(), data.max()
        normalized_data = (data - min_val) / (max_val - min_val + 1e-6)  # Avoid division by zero
        return torch.tensor(normalized_data, dtype=torch.float32)

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