import os
import math
import torch
import imageio
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from torch.nn import functional as F
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from .dataset import SlidingWindowDataset, read_csv_files

def visualize_skeleton(positions, save_path='skeleton_animation.gif'):
    connections = [
        # Spine
        (0, 1),  # Pelvis -> Spine Chest
        (1, 2),  # Spine Chest -> Neck
        (2, 15), # Neck -> Head

        # Left Arm
        (2, 3),  # Neck -> Left Shoulder
        (3, 4),  # Left Shoulder -> Left Elbow
        (4, 5),  # Left Elbow -> Left Wrist

        # Right Arm
        (2, 6),  # Neck -> Right Shoulder
        (6, 7),  # Right Shoulder -> Right Elbow
        (7, 8),  # Right Elbow -> Right Wrist

        # Left Leg
        (0, 9),  # Pelvis -> Left Hip
        (9, 10), # Left Hip -> Left Knee
        (10, 11), # Left Knee -> Left Ankle

        # Right Leg
        (0, 12),  # Pelvis -> Right Hip
        (12, 13), # Right Hip -> Right Knee
        (13, 14)  # Right Knee -> Right Ankle
    ]
    
    frames = []
    sample_idx = 0

    # Loop through all 90 frames
    for frame_idx in range(90):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Remove background and axes
        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()
        
        for joint1, joint2 in connections:
            joint1_coords = positions[sample_idx, frame_idx, joint1*3:(joint1*3)+3]
            joint2_coords = positions[sample_idx, frame_idx, joint2*3:(joint2*3)+3]

            if len(joint1_coords) < 3 or len(joint2_coords) < 3:
                continue

            xs = [joint1_coords[0], joint2_coords[0]]
            ys = [joint1_coords[1], joint2_coords[1]]
            zs = [joint1_coords[2], joint2_coords[2]]

            # Plot the bones as dark blue lines
            ax.plot(xs, ys, zs, marker='o', color='darkblue')

            # Plot the joints as red dots
            ax.scatter(joint1_coords[0], joint1_coords[1], joint1_coords[2], color='red', s=50)  # Joint 1
            ax.scatter(joint2_coords[0], joint2_coords[1], joint2_coords[2], color='red', s=50)  # Joint 2

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Set a rotated view angle for better depth perception
        ax.view_init(elev=-90, azim=-90)  # Adjust azimuth and elevation for better 3D perception

        # Capture the frame
        plt.tight_layout()
        fig.canvas.draw()

        # Convert to a numpy array and add to frames list
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        # Close the figure to save memory
        plt.close(fig)

    # Save the frames as a GIF with a duration of 0.2 seconds per frame (5 fps)
    imageio.mimsave(save_path, frames, duration=0.2)  # Adjust duration for frame speed
    print(f'GIF saved as {save_path}')

# def visualize_skeleton(positions, frame_range=(30, 50), save_path='skeleton_animation.gif'):

#     connections = [
#         # Spine
#         (0, 1),  # Pelvis -> Spine Chest
#         (1, 2),  # Spine Chest -> Neck
#         (2, 15), # Neck -> Head

#         # Left Arm
#         (2, 3),  # Neck -> Left Shoulder
#         (3, 4),  # Left Shoulder -> Left Elbow
#         (4, 5),  # Left Elbow -> Left Wrist

#         # Right Arm
#         (2, 6),  # Neck -> Right Shoulder
#         (6, 7),  # Right Shoulder -> Right Elbow
#         (7, 8),  # Right Elbow -> Right Wrist

#         # Left Leg
#         (0, 9),  # Pelvis -> Left Hip
#         (9, 10), # Left Hip -> Left Knee
#         (10, 11), # Left Knee -> Left Ankle

#         # Right Leg
#         (0, 12),  # Pelvis -> Right Hip
#         (12, 13), # Right Hip -> Right Knee
#         (13, 14)  # Right Knee -> Right Ankle
#     ]
    
#     # Define colors for each body part
#     color_map = {
#         "spine": "magenta",
#         "left_arm": "blue",
#         "right_arm": "red",
#         "left_leg": "purple",
#         "right_leg": "green",
#         "head": "orange"
#     }

#     frames = []
#     sample_idx = 0

#     for frame_idx in range(frame_range[0], frame_range[1] + 1):
#         fig = plt.figure(figsize=(8, 8))
#         ax = fig.add_subplot(111, projection='3d')

#         for joint1, joint2 in connections:
#             # Correct slicing to exclude the padded column
#             joint1_coords = positions[sample_idx, frame_idx, joint1*3:(joint1*3)+3]
#             joint2_coords = positions[sample_idx, frame_idx, joint2*3:(joint2*3)+3]

#             # Skip if the extracted coordinates are invalid or empty
#             if len(joint1_coords) < 3 or len(joint2_coords) < 3:
#                 continue

#             xs = [joint1_coords[0], joint2_coords[0]]
#             ys = [joint1_coords[1], joint2_coords[1]]
#             zs = [joint1_coords[2], joint2_coords[2]]

#             # Determine the color based on the joint index
#             if (joint1, joint2) in [(0, 1), (1, 2), (2, 3)]:
#                 color = color_map["spine"]
#             elif joint1 in [4, 5, 6]:  # Left arm
#                 color = color_map["left_arm"]
#             elif joint1 in [7, 8, 9]:  # Right arm
#                 color = color_map["right_arm"]
#             elif joint1 in [10, 11, 12]:  # Left leg
#                 color = color_map["left_leg"]
#             elif joint1 in [13, 14, 15]:  # Right leg
#                 color = color_map["right_leg"]
#             else:  # Head
#                 color = color_map["head"]

#             ax.plot(xs, ys, zs, marker='o', color=color)

#         # Set labels and title
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.set_title(f'Frame {frame_idx}')
#         ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

#         # Set a rotated view angle for better depth perception
#         ax.view_init(elev=-90, azim=-90)  # Adjust azimuth and elevation for better 3D perception

#         # Capture the frame
#         plt.tight_layout()
#         fig.canvas.draw()

#         # Convert to a numpy array and add to frames list
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         frames.append(image)

#         # Close the figure to save memory
#         plt.close(fig)

#     # Save the frames as a GIF
#     imageio.mimsave(save_path, frames, duration=10)  # Adjust duration for frame speed
#     print(f'GIF saved as {save_path}')

# def visualize_skeleton_comparison(x0, x0_pred, frame_range=(45, 50), save_path='skeleton_comparison_animation.gif'):
#     # Define joint connections
#     joint_connections = [
#         (0, 1), (1, 2), (2, 3),  # Left arm: Neck -> Left Shoulder -> Left Elbow -> Left Wrist
#         (0, 4), (4, 5), (5, 6),  # Right arm: Neck -> Right Shoulder -> Right Elbow -> Right Wrist
#         (7, 8), (8, 9),  # Left leg: Hip Left -> Knee Left -> Ankle Left
#         (10, 11), (11, 12)  # Right leg: Hip Right -> Knee Right -> Ankle Right
#     ]

#     frames = []
#     sample_idx = 0

#     for frame_idx in range(frame_range[0], frame_range[1] + 1):
#         fig, axs = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})

#         for ax, positions, title in zip(axs, [x0, x0_pred], ['Original x0', 'Predicted x0_pred']):
#             for joint1, joint2 in joint_connections:
#                 # Get joint coordinates
#                 joint1_coords = positions[sample_idx, frame_idx, joint1*3:(joint1*3)+3]
#                 joint2_coords = positions[sample_idx, frame_idx, joint2*3:(joint2*3)+3]

#                 # Plot the bone
#                 xs = [joint1_coords[0], joint2_coords[0]]  # X coordinates
#                 ys = [joint1_coords[1], joint2_coords[1]]  # Y coordinates
#                 zs = [joint1_coords[2], joint2_coords[2]]  # Z coordinates

#                 ax.plot(xs, ys, zs, marker='o', color='blue' if joint1 < 7 else 'green')

#             # Set labels and title
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')
#             ax.set_zlabel('Z')
#             ax.set_title(title)
#             ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

#             # Set a rotated view angle for better depth perception
#             ax.view_init(elev=-90, azim=-90)  # Adjust azimuth and elevation for better 3D perception

#         # Capture the frame
#         plt.tight_layout()
#         fig.canvas.draw()

#         # Convert to a numpy array and add to frames list
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         frames.append(image)

#         # Close the figure to save memory
#         plt.close(fig)

#     # Save the frames as a GIF
#     imageio.mimsave(save_path, frames, duration=0.5)  # Adjust duration for frame speed
#     print(f'GIF saved as {save_path}')

def custom_sensor_loss(predictions, labels):
    """
    Custom loss function for the sensor model.

    Args:
        predictions (torch.Tensor): The model output, typically of shape (batch_size, num_classes).
        labels (torch.Tensor): The ground truth labels, typically one-hot encoded with shape (batch_size, num_classes).

    Returns:
        torch.Tensor: The computed loss.
    """
    # Standard CrossEntropyLoss
    loss = torch.nn.CrossEntropyLoss()(predictions, labels.argmax(dim=1))
    
    # Additional regularization or penalties can be added here if needed
    # Example: penalty = compute_custom_penalty(predictions)
    # loss += penalty

    return loss

def compute_loss(
    args, model, x0, context, t, mask=None, noise=None, device="cpu",
    diffusion_process=None, angular_loss=False, lip_reg=False, epoch=None
):
    """
    Custom loss function for the diffusion model using a mix of DDIM and DDPM.

    Args:
        model (nn.Module): The diffusion model.
        x0 (torch.Tensor): The original clean input (e.g., skeleton positions).
        context (torch.Tensor): The context or conditioning input (e.g., sensor model output).
        t (torch.Tensor): The time step indices.
        mask (torch.Tensor, optional): A mask tensor indicating valid or invalid positions in the data.
        noise (torch.Tensor, optional): The noise to be added.
        device (str): The device to run the model on.
        diffusion_process (DiffusionProcess): The diffusion process object.
        angular_loss (bool): Whether to include angular loss in the computation.
        lip_reg (bool): Whether to include Lipschitz regularization.
        epoch (int, optional): The current training epoch, used for adjusting angular loss weight.

    Returns:
        torch.Tensor: The computed total loss.
    """
    if noise is None:
        noise = torch.randn_like(x0)

    # Ensure all tensors are on the correct device
    x0 = (x0 - x0.mean()) / x0.std()
    x0 = x0.to(device)
    context = context.to(device)
    t = t.to(device)
    noise = noise.to(device)

    # Add noise to x0 to get xt
    xt, _ = diffusion_process.add_noise(x0, t)
    # Toggle between predicting noise (DDPM) and predicting clean data (DDIM) using ddim_scale
    if diffusion_process.ddim_scale == 1.0:
        # Predict noise (DDPM approach)
        predicted_noise = model(xt, context, t).to(device)
        mse_loss = F.mse_loss(predicted_noise, noise)
    else:
        # Predict clean data (DDIM approach)
        x0_pred = model(xt, context, t).to(device)
        x0_pred_originail = x0_pred
        mse_loss = F.mse_loss(x0_pred, x0)

        if epoch is not None:
        # For the 0th epoch, save both the original and predicted skeleton
            if epoch == 0:
                visualize_skeleton(
                    x0.cpu().detach().numpy(),
                    save_path=f'./gif_tl/original_skeleton_animation_{epoch}.gif'
                )
                visualize_skeleton(
                    x0_pred.cpu().detach().numpy(),
                    save_path=f'./gif_tl/skeleton_animation_epoch_{epoch}.gif'
                )
            # For the first 400 epochs, save visualizations every 30 epochs
            elif epoch <= 2500 and epoch % 250 == 0:
                visualize_skeleton(
                    x0.cpu().detach().numpy(),
                    save_path=f'./gif_tl/original_skeleton_animation_{epoch}.gif'
                )
                visualize_skeleton(
                    x0_pred.cpu().detach().numpy(),
                    save_path=f'./gif_tl/skeleton_animation_epoch_{epoch}.gif'
                )
            # For the remaining epochs, save visualizations every 500 epochs
            elif epoch == 3000:
                visualize_skeleton(
                    x0.cpu().detach().numpy(),
                    save_path=f'./gif_tl/original_skeleton_animation_{epoch}.gif'
                )
                visualize_skeleton(
                    x0_pred.cpu().detach().numpy(),
                    save_path=f'./gif_tl/skeleton_animation_epoch_{epoch}.gif'
                )

    total_loss = mse_loss

    # Optional angular loss
    if angular_loss:
        if diffusion_process.ddim_scale == 1.0:
            # Angular loss based on predicted noise
            predicted_joint_angles = compute_joint_angles(predicted_noise)
        else:
            # Angular loss based on predicted clean data
            predicted_joint_angles = compute_joint_angles(x0_pred)
            joint_angles = compute_joint_angles(x0)
            angular_loss_value = F.mse_loss(predicted_joint_angles, joint_angles)
            # difference = joint_angles - predicted_joint_angles

            # # Compute the Frobenius norm squared (||.||_F^2)
            # frobenius_norm_squared = torch.norm(difference, p='fro') ** 1/2

            # # Compute the final loss with the 1/2 scaling factor
            # angular_loss_value = 0.5 * frobenius_norm_squared
        total_loss += 0.5 * angular_loss_value

    # Optional Lipschitz regularization (LipReg)
    if lip_reg:
        noisy_context = add_random_noise(context, noise_std=0.01, noise_fraction=0.2)
        xt_noisy, _ = diffusion_process.add_noise(x0, t, noisy_context)
        if diffusion_process.ddim_scale == 1.0:
            predicted_noise_lr = model(xt_noisy, noisy_context, t).to(device)
            lip_reg_loss = F.mse_loss(predicted_noise_lr, noise)
        else:
            x0_pred_lr = model(xt_noisy, noisy_context, t).to(device)
            lip_reg_loss = F.mse_loss(x0_pred_lr, x0_pred_originail)
        total_loss += 0.5 * lip_reg_loss

    return total_loss


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

def min_max_scale(data, data_min, data_max, feature_range=(0, 1)):
    data_min = np.array(data_min)
    data_max = np.array(data_max)
    
    scale = (feature_range[1] - feature_range[0]) / (data_max - data_min + 1e-8)
    min_range = feature_range[0]

    return scale * (data - data_min) + min_range

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

        # Ensure consistent column sizes (96 columns for skeleton data)
        for file in common_files:
            if skeleton_data[file].shape[1] == 97:
                # If there's an extra column, drop it (assumed to be the first column)
                skeleton_data[file] = skeleton_data[file].iloc[:, 1:]  # Drop the first column

        # Extract activity codes from file names
        activity_codes = sorted(set(file.split('A')[1][:2].lstrip('0') for file in common_files))

        label_encoder = OneHotEncoder(sparse_output=False)
        label_encoder.fit([[code] for code in activity_codes])

        window_size = args.window_size
        overlap = args.overlap

        # Instantiate the dataset with Min-Max scaling applied at each window level
        dataset = SlidingWindowDataset(
            skeleton_data=skeleton_data,
            sensor1_data=sensor_data1,
            sensor2_data=sensor_data2,
            common_files=common_files,
            window_size=window_size,
            overlap=overlap,
            label_encoder=label_encoder,
            augment=args.augment,
            scaling="minmax"  # Use Min-Max scaling for each window segment
        )

        return dataset
    else:
        raise ValueError("Only 'Own_data' dataset type is supported. 'UTD_MHAD' is not supported yet.")


def sample_by_t(tensor_to_sample, timesteps, x_shape):
    batch_size = timesteps.shape[0]
    timesteps = timesteps.to(tensor_to_sample.device)
    sampled_tensor = tensor_to_sample.gather(0, timesteps)
    sampled_tensor = torch.reshape(sampled_tensor, (batch_size,) + (1,) * (len(x_shape) - 1))
    return sampled_tensor

# def compute_loss(args, model, x0, context, t, epoch, noise=None, device="cpu", diffusion_process=None, angular_loss=False, lip_reg=False):
#     if noise is None:
#         noise = torch.randn_like(x0)
    
#     x0 = x0.to(device)
#     context = context.to(device)
#     t = t.to(device)
#     noise = noise.to(device)

#     # Add noise based on the diffusion process
#     x_t_subset, _ = diffusion_process.add_noise(x0, t)
    
#     # Generate the predicted noise for only the subset of joints
#     predicted_noise = model(x_t_subset, context, t).to(device)

#     # Compute MSE loss for the subset of joints
#     loss = torch.mean(F.mse_loss(noise, predicted_noise))

#     # Optional angular loss
#     if angular_loss and epoch is not None:
#         joint_angles = compute_joint_angles(x0)
#         predicted_joint_angles = compute_joint_angles(predicted_noise)
#         assert not torch.isnan(predicted_joint_angles).any(), "Predicted joint angles contain NaNs!"
#         angular_loss_value = frobenius_norm_loss(predicted_joint_angles, joint_angles)
#         total_loss = loss + (0.05 * angular_loss_value)
#     else:
#         total_loss = loss
    
#     if lip_reg and epoch is not None:
#         noisy_context = add_random_noise(context, noise_std=0.01, noise_fraction=0.2)
#         x_t_noisy, _ = diffusion_process.add_noise(x0, t, noisy_context)
#         predicted_noise_lr = model(x_t_noisy, noisy_context, t).to(device)
#         lr_loss = F.mse_loss(noise, predicted_noise_lr)
#         total_loss = loss + (0.5 * lr_loss)
#     else:
#         total_loss = loss

#     return total_loss


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
        [3, 4, 5],  # Left shoulder, elbow, wrist
        [6, 7, 8],  # Right shoulder, elbow, wrist
        [9, 10, 11],  # Left hip, knee, ankle
        [12, 13, 14]  # Right hip, knee, ankle
    ], device=positions.device)
    
    # joint_pairs = torch.tensor([
    #     [4, 5, 6],   # Left shoulder, elbow, wrist
    #     [7, 8, 9],   # Right shoulder, elbow, wrist
    #     [10, 11, 12],  # Left hip, knee, ankle
    #     [13, 14, 15]  # Right hip, knee, ankle
    # ], device=positions.device)


    # joint_pairs = torch.tensor([
    #     [0, 1, 2],  # Left shoulder, elbow, wrist
    #     [3, 4, 5],  # Right shoulder, elbow, wrist
    #     [6, 7, 8],  # Left hip, knee, ankle
    #     [9, 10, 11]  # Right hip, knee, ankle
    # ], device=positions.device)

    batch_size, num_frames, _ = positions.shape

    # Account for potential padding: remove extra columns
    # Calculate how many joints we have based on columns
    num_joints = (_ // 3)

    # If extra padding columns exist, slice them off
    positions = positions[:, :, :num_joints * 3]

    # Reshape positions to (batch_size, num_frames, num_joints, 3)
    positions = positions.view(batch_size, num_frames, num_joints, 3)

    # Process in smaller chunks if the tensor is large
    chunk_size = 100  # Adjust based on memory capacity
    angles = []

    for i in range(0, num_frames, chunk_size):
        positions_chunk = positions[:, i:i + chunk_size]

        # Compute vectors for the joint pairs
        vectors1 = positions_chunk[:, :, joint_pairs[:, 1]] - positions_chunk[:, :, joint_pairs[:, 0]]
        vectors2 = positions_chunk[:, :, joint_pairs[:, 1]] - positions_chunk[:, :, joint_pairs[:, 2]]

        # Compute dot product
        dot_product = torch.sum(vectors1 * vectors2, dim=-1)

        # Compute norms
        norm1 = torch.norm(vectors1, dim=-1)
        norm2 = torch.norm(vectors2, dim=-1)

        # Avoid division by zero
        denominator = norm1 * norm2
        valid_denominator = denominator != 0

        # Compute cosine of angles where denominator is valid
        cosine_angles = torch.zeros_like(dot_product)
        epsilon = 1e-6
        denominator = torch.clamp(denominator, min=epsilon)
        cosine_angles[valid_denominator] = dot_product[valid_denominator] / denominator[valid_denominator]

        # Clamp values to avoid numerical instability
        cosine_angles = torch.clamp(cosine_angles, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute angles in radians
        chunk_angles = torch.acos(cosine_angles)
        chunk_angles[~valid_denominator] = 0

        angles.append(chunk_angles)

    # Concatenate all angle chunks
    return torch.cat(angles, dim=1)

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

