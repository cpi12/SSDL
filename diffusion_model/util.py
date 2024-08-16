import os
import torch
import numpy as np
import pandas as pd
import math
from torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from .dataset import SlidingWindowDataset, read_csv_files, handle_nans

def add_random_noise(context, noise_std=0.01, noise_fraction=0.2):
    num_samples = context.size(0) 
    num_noisy_samples = int(noise_fraction * num_samples)
    noisy_indices = torch.randperm(num_samples)[:num_noisy_samples]

    noise = torch.randn_like(context[noisy_indices]) * noise_std

    context[noisy_indices] += noise

    return context

def frobenius_norm_loss(predicted, target):
    # Calculate the Frobenius norm loss
    return torch.norm(predicted - target, p='fro')

def prepare_dataset(args):
    if args.dataset_type == 'Own_data':
        skeleton_folder = args.skeleton_folder
        sensor_folder1 = args.sensor_folder1
        sensor_folder2 = args.sensor_folder2

        skeleton_data = read_csv_files(skeleton_folder)
        sensor_data1 = read_csv_files(sensor_folder1)
        sensor_data2 = read_csv_files(sensor_folder2)

        # Find common files across all three directories
        common_files = list(set(skeleton_data.keys()).intersection(set(sensor_data1.keys()), set(sensor_data2.keys())))

        if not common_files:
            raise ValueError("No common files found across the skeleton, sensor1, and sensor2 directories.")

        # Handle NaNs
        # skeleton_data = {file: handle_nans(skeleton_data[file]) for file in common_files}
        # sensor1_data = {file: handle_nans(sensor_data1[file]) for file in common_files}
        # sensor2_data = {file: handle_nans(sensor_data2[file]) for file in common_files}

        # Extract activity codes from file names
        activity_codes = sorted(set(file.split('A')[1][:2].lstrip('0') for file in common_files))

        label_encoder = OneHotEncoder(sparse_output=False)
        label_encoder.fit([[code] for code in activity_codes])

        window_size = args.window_size
        overlap = args.overlap

        # Instantiate the dataset without normalization yet
        dataset = SlidingWindowDataset(
            skeleton_data=skeleton_data,
            sensor1_data=sensor_data1,
            sensor2_data=sensor_data2,
            common_files=common_files,
            window_size=window_size,
            overlap=overlap,
            label_encoder=label_encoder,
            skeleton_global_mean=None,  # Will be calculated later
            skeleton_global_std=None,  # Will be calculated later
            sensor1_global_mean=None,  # Will be calculated later
            sensor1_global_std=None,  # Will be calculated later
            sensor2_global_mean=None,  # Will be calculated later
            sensor2_global_std=None,  # Will be calculated later
            augment=args.augment
        )

        # Calculate global statistics using the entire dataset
        skeleton_train_data = [dataset[i][0].numpy() for i in range(len(dataset))]
        sensor1_train_data = [dataset[i][1].numpy() for i in range(len(dataset))]
        sensor2_train_data = [dataset[i][2].numpy() for i in range(len(dataset))]

        skeleton_global_mean = np.mean(np.concatenate(skeleton_train_data), axis=0)
        skeleton_global_std = np.std(np.concatenate(skeleton_train_data), axis=0)
        sensor1_global_mean = np.mean(np.concatenate(sensor1_train_data), axis=0)
        sensor1_global_std = np.std(np.concatenate(sensor1_train_data), axis=0)
        sensor2_global_mean = np.mean(np.concatenate(sensor2_train_data), axis=0)
        sensor2_global_std = np.std(np.concatenate(sensor2_train_data), axis=0)

        # Apply the global statistics for normalization to the entire dataset
        dataset.skeleton_global_mean = skeleton_global_mean
        dataset.skeleton_global_std = skeleton_global_std
        dataset.sensor1_global_mean = sensor1_global_mean
        dataset.sensor1_global_std = sensor1_global_std
        dataset.sensor2_global_mean = sensor2_global_mean
        dataset.sensor2_global_std = sensor2_global_std
        dataset.normalize = True  # Enable normalization within the dataset

        # Return the full dataset
        return dataset
    else:
        raise ValueError("Only 'Own_data' dataset type is supported. 'UTD_MHAD' is not supported yet.")

def sample_by_t(tensor_to_sample, timesteps, x_shape):
    batch_size = timesteps.shape[0]
    timesteps = timesteps.to(tensor_to_sample.device)
    sampled_tensor = tensor_to_sample.gather(0, timesteps)
    sampled_tensor = torch.reshape(sampled_tensor, (batch_size,) + (1,) * (len(x_shape) - 1))
    return sampled_tensor

def compute_loss(args, model, x0, context, t, epoch, noise=None, device="cpu", diffusion_process=None, angular_loss=False, lip_reg=False):
    if noise is None:
        noise = torch.randn_like(x0)
    
    x0 = x0.to(device)
    context = context.to(device)
    t = t.to(device)
    noise = noise.to(device)

    # Add noise based on the diffusion process
    x_t_subset, _ = diffusion_process.add_noise(x0, t, noise)
    
    # Generate the predicted noise for only the subset of joints
    predicted_noise = model(x_t_subset, context, t).to(device)

    # Compute MSE loss for the subset of joints
    loss = F.mse_loss(noise, predicted_noise)

    # Optional angular loss
    if angular_loss and epoch is not None:
        joint_angles = compute_joint_angles(x0)
        predicted_joint_angles = compute_joint_angles(predicted_noise)
        assert not torch.isnan(predicted_joint_angles).any(), "Predicted joint angles contain NaNs!"
        angular_loss_value = frobenius_norm_loss(predicted_joint_angles, joint_angles)
        total_loss = loss + (0.05 * angular_loss_value)
    else:
        total_loss = loss
    
    if lip_reg and epoch is not None:
        noisy_context = add_random_noise(context, noise_std=0.01, noise_fraction=0.2)
        x_t_noisy, _ = diffusion_process.add_noise(x0, t, noisy_context)
        predicted_noise_lr = model(x_t_noisy, noisy_context, t).to(device)
        lr_loss = F.mse_loss(noise, predicted_noise_lr)
        total_loss = loss + (0.5 * lr_loss)
    else:
        total_loss = loss

    return total_loss


def extract_joint_subset(positions):
    """
    Extract only the subset of key points for computing joint angles based on the joint index mapping.
    The input tensor should remain [batch_size, 90, 96] but only specific columns will be used for angle computation.

    Args:
        positions (torch.Tensor): The full skeleton tensor of shape [batch_size, 90, 96].

    Returns:
        torch.Tensor: Subset tensor containing only the columns corresponding to relevant joints.
    """
    joint_indices = {
        5: [15, 16, 17],  # Left shoulder
        6: [18, 19, 20],  # Left elbow
        7: [21, 22, 23],  # Left wrist
        12: [36, 37, 38],  # Right shoulder
        13: [39, 40, 41],  # Right elbow
        14: [42, 43, 44],  # Right wrist
        18: [54, 55, 56],  # Left hip
        19: [57, 58, 59],  # Left knee
        20: [60, 61, 62],  # Left ankle
        22: [66, 67, 68],  # Right hip
        23: [69, 70, 71],  # Right knee
        24: [72, 73, 74],  # Right ankle
    }
    
    selected_columns = sum(joint_indices.values(), [])
    
    return positions[:, :, selected_columns]

def compute_joint_angles(positions):
    # Indices of the joints of interest for computing angles
    joint_pairs = torch.tensor([
        [5, 6, 7],  # Left shoulder, elbow, wrist
        [12, 13, 14],  # Right shoulder, elbow, wrist
        [18, 19, 20],  # Left hip, knee, ankle
        [22, 23, 24]  # Right hip, knee, ankle
    ], device=positions.device)

    batch_size, num_frames, _ = positions.shape

    # Reshape positions to (batch_size, num_frames, 32, 3)
    # Each joint is represented by 3 values (x, y, z)
    positions = positions.view(batch_size, num_frames, 32, 3)

    # Extract the relevant joint positions based on joint_pairs
    joint_indices = joint_pairs.flatten()  # Flatten to get [5, 6, 7, 12, 13, 14, 18, 19, 20, 22, 23, 24]
    positions_subset = positions[:, :, joint_indices]  # Shape: [batch_size, num_frames, 12, 3]

    # Compute vectors for the joint pairs
    vectors1 = positions_subset[:, :, joint_pairs[:, 1] - joint_indices.min()] - positions_subset[:, :, joint_pairs[:, 0] - joint_indices.min()]
    vectors2 = positions_subset[:, :, joint_pairs[:, 1] - joint_indices.min()] - positions_subset[:, :, joint_pairs[:, 2] - joint_indices.min()]

    # Compute dot product
    dot_product = torch.sum(vectors1 * vectors2, dim=-1)  # Shape: [batch_size, num_frames, 4]

    # Compute norms
    norm1 = torch.norm(vectors1, dim=-1)
    norm2 = torch.norm(vectors2, dim=-1)

    # Avoid division by zero
    denominator = norm1 * norm2
    valid_denominator = denominator != 0

    # Compute cosine of angles where denominator is valid
    cosine_angles = torch.zeros_like(dot_product)
    cosine_angles[valid_denominator] = dot_product[valid_denominator] / denominator[valid_denominator]

    # Clamp values to avoid numerical instability
    cosine_angles = torch.clamp(cosine_angles, -1.0 + 1e-7, 1.0 - 1e-7)

    # Compute angles in radians
    angles = torch.acos(cosine_angles)

    # Set angles to zero where vectors are zero-length
    angles[~valid_denominator] = 0

    return angles  # Shape: [batch_size, num_frames, 4]

def get_alpha(current_epoch, max_alpha=1.0, warmup_epochs=10):
    """
    Returns a gradually increasing alpha value for angular loss.
    
    Args:
        current_epoch (int): The current epoch in training.
        max_alpha (float): The maximum value alpha should reach.
        warmup_epochs (int): The number of epochs over which to gradually increase alpha.
    
    Returns:
        float: The alpha value for the current epoch.
    """
    if current_epoch < warmup_epochs:
        alpha = (max_alpha / warmup_epochs) * current_epoch
    else:
        alpha = max_alpha
    return alpha


def linear_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps)

def cosine_noise_schedule(timesteps, s=0.008):
    steps = np.arange(timesteps + 1) / timesteps
    alphas_cumprod = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas

def quadratic_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = np.linspace(-6, 6, timesteps)
    return beta_start + (beta_end - beta_start) / (1 + np.exp(-betas))

def get_noise_schedule(schedule_type, timesteps, beta_start=0.0001, beta_end=0.02):
    if schedule_type == 'linear':
        return linear_noise_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == 'cosine':
        return cosine_noise_schedule(timesteps)
    elif schedule_type == 'quadratic':
        return quadratic_noise_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == 'sigmoid':
        return sigmoid_noise_schedule(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown noise schedule type: {schedule_type}")

def create_stratified_split(dataset, test_size=0.3, val_size=0.5, random_state=42):
    # Extract labels for stratified splitting
    labels = [label.argmax().item() for _, _, _, label in dataset]  # Assumes labels are one-hot encoded

    # Perform the initial stratified split to get train and (validation + test) indices
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_test_idx = next(stratified_split.split(np.zeros(len(labels)), labels))

    # Perform another stratified split on the (validation + test) indices
    stratified_val_test_split = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    val_idx, test_idx = next(stratified_val_test_split.split(np.zeros(len(val_test_idx)), [labels[i] for i in val_test_idx]))

    # Map back to the original indices
    val_idx = [val_test_idx[i] for i in val_idx]
    test_idx = [val_test_idx[i] for i in test_idx]

    return train_idx, val_idx, test_idx

def calculate_fid(real_activations, generated_activations):
    real_activations = np.concatenate(real_activations, axis=0)
    generated_activations = np.concatenate(generated_activations, axis=0)

    real_activations = real_activations.reshape(real_activations.shape[0], -1)
    generated_activations = generated_activations.reshape(generated_activations.shape[0], -1)

    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_generated = np.mean(generated_activations, axis=0)
    sigma_generated = np.cov(generated_activations, rowvar=False)

    diff = mu_real - mu_generated
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_generated, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm(sigma_real @ sigma_generated + np.eye(sigma_real.shape[0]) * 1e-6)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_generated) - 2 * np.trace(covmean)
    return fid

def get_time_embedding(timestep, dtype):
    half_dim = 320 // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = timestep * emb
    emb = np.concatenate((np.sin(emb), np.cos(emb)))
    return torch.tensor(emb, dtype=dtype)

def get_file_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)

def rescale(tensor, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    tensor = ((tensor - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    if clamp:
        tensor = torch.clamp(tensor, new_min, new_max)
    return tensor

def move_channel(tensor, to="last"):
    if to == "last":
        return tensor.permute(0, 2, 3, 1)
    elif to == "first":
        return tensor.permute(0, 3, 1, 2)

def plot_and_save(values, ylabel, title, filename, output_dir):
    plt.figure()
    plt.plot(values)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

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

def scale_skeleton(skeleton, scale_factor=1.2):
    return skeleton * scale_factor

def flip_skeleton_horizontal(skeleton):
    flipped_skeleton = skeleton.copy()
    flipped_skeleton[:, [3, 4, 5]] = skeleton[:, [6, 7, 8]]
    flipped_skeleton[:, [6, 7, 8]] = skeleton[:, [3, 4, 5]]
    flipped_skeleton[:, ::3] = -flipped_skeleton[:, ::3]
    return flipped_skeleton

def get_keypoint_groups():
    """
    Define keypoint groups based on their logical regions (e.g., arms, legs, and anchors).

    Returns:
        dict: A dictionary where the keys are group names and the values are lists of keypoint indices.
    """
    return {
        "left_arm": [0, 1, 2],  # Shoulder, elbow, wrist
        "right_arm": [3, 4, 5],  # Shoulder, elbow, wrist
        "left_leg": [6, 7, 8],  # Hip, knee, ankle
        "right_leg": [9, 10, 11],  # Hip, knee, ankle
        "anchors": [0, 3, 6, 9]  # Hips and shoulders (these are key anchor points for stability)
    }

