import torch
import os
from .encoder import Encoder1D
from .model import Diffusion1D
from .decoder import Decoder1D
from .sensor_model import CombinedLSTMClassifier

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

def load_diffusion(device):
    """
    Load the Diffusion1D model onto the specified device.
    Args:
        device (torch.device): Device to load the model on.
    Returns:
        Diffusion1D: Loaded diffusion model.
    """
    model = Diffusion1D().to(device)
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

def load_sensor_model(args, device):
    model = CombinedLSTMClassifier(
        sensor_input_size=3, 
        hidden_size=64, 
        num_layers=2, 
        num_classes=12, 
        conv_channels=16, 
        kernel_size=3, 
        dropout=0.5, 
        num_heads=4
    ).to(device)

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



