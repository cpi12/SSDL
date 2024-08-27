import torch
import os
from .encoder import Encoder1D
from .model import Diffusion1D
from .decoder import Decoder1D
from .sensor_model import CombinedLSTMClassifier
import torch
import torch.nn as nn

# def get_edge_index(device='cpu'):
#     # Define the connections for the filtered key joints
#     connections = [
#         (0, 1), (1, 2), (2, 3),  # Left arm: Neck -> Left Shoulder -> Left Elbow -> Left Wrist
#         (0, 4), (4, 5), (5, 6),  # Right arm: Neck -> Right Shoulder -> Right Elbow -> Right Wrist
#         (7, 8), (8, 9),  # Left leg: Hip Left -> Knee Left -> Ankle Left
#         (10, 11), (11, 12)  # Right leg: Hip Right -> Knee Right -> Ankle Right
#     ]
#     edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()
#     return edge_index.to(device)

def get_edge_index(device='cpu'):
    # Updated connections based on the filtered key_joint_indexes
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
    
    # Converting the connections to a PyTorch edge index tensor
    edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()
    return edge_index.to(device)


def load_encoder(device):
    """
    Load the Encoder1D model onto the specified device.
    Args:
        device (torch.device): Device to load the model on.
    Returns:
        Encoder1D: Loaded encoder model.
    """
    model = Encoder1D().to(device)
    return model

# def load_diffusion(device):
#     edge_index = get_edge_index(device)  # Ensure edge_index is moved to the correct device
#     model = Diffusion1D(edge_index=edge_index, device=device).to(device)
#     return model


def load_diffusion(device):
    """
    Load the Diffusion1D model onto the specified device.
    Args:
        device (torch.device): Device to load the model on.
    Returns:
        Diffusion1D: Loaded diffusion model.
    """
    edge_index = get_edge_index(device)
    model = Diffusion1D(edge_index=edge_index).to(device)
    return model

def load_decoder(device):
    """
    Load the Decoder1D model onto the specified device.
    Args:
        device (torch.device): Device to load the model on.
    Returns:
        Decoder1D: Loaded decoder model.
    """
    model = Decoder1D().to(device)
    return model

def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def load_sensor_model(args, device):
    model = CombinedLSTMClassifier(
        sensor_input_size=3, 
        hidden_size=256, 
        num_layers=8, 
        num_classes=12, 
        conv_channels=16, 
        kernel_size=3, 
        dropout=0.5, 
        num_heads=4
    ).to(device)

    model.apply(initialize_weights)

    if not args.train_sensor_model:
        sensor_model_path = os.path.join(args.output_dir, "sensor_model", "best_sensor_model.pth")
        if os.path.exists(sensor_model_path):
            checkpoint = torch.load(sensor_model_path, map_location=device)
            # Handle the 'module.' prefix if necessary
            if 'module.' in next(iter(checkpoint.keys())):
                checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
            model.eval()  # Set model to evaluation mode
            print(f"Loaded pre-trained sensor model from {sensor_model_path}")
        else:
            raise FileNotFoundError(f"No pre-trained sensor model found at {sensor_model_path}")
    
    return model



