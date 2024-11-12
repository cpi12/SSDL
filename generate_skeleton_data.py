import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from diffusion_model.model_loader import load_sensor_model, load_diffusion
from diffusion_model.skeleton_model import SkeletonLSTMModel
from diffusion_model.diffusion import DiffusionProcess, Scheduler
from dataset import handle_nan_and_scale  # Assuming the handle_nan_and_scale function is in dataset.py

# Preprocess accelerometer data with overlapping windows
def preprocess_accelerometer_data_with_overlap(raw_accelerometer_data, window_size=90, overlap=45):
    step = window_size - overlap
    windows = []

    for start in range(0, len(raw_accelerometer_data) - window_size + 1, step):
        window = raw_accelerometer_data[start:start + window_size]
        window = handle_nan_and_scale(window, scaling_method="minmax")
        windows.append(window)

    return torch.tensor(windows, dtype=torch.float32)  # Return a tensor with all windows

# Generate skeleton data with sliding window preprocessing
def generate_skeleton(sensor_model, diffusion_model, raw_accelerometer_data, device, window_size=90, overlap=45):
    sensor_model.to(device)
    diffusion_model.to(device)

    # Preprocess the accelerometer data with overlapping windows
    input_data_windows = preprocess_accelerometer_data_with_overlap(raw_accelerometer_data, window_size, overlap).to(device)

    # Initialize a list to hold the generated skeleton data
    all_skeleton_data = []

    # Pass each window through the sensor model and generate skeleton data
    with torch.no_grad():
        for input_data in input_data_windows:
            input_data = input_data.unsqueeze(0)  # Add batch dimension
            context, _ = sensor_model(input_data, return_attn_output=True)

            # Setup diffusion process
            scheduler = Scheduler(sched_type='cosine', T=1000, step=1, device=device)
            diffusion_process = DiffusionProcess(scheduler=scheduler, device=device, ddim_scale=0.0)

            # Generate skeleton data for each window
            skeleton_data = diffusion_process.sample(diffusion_model, context)
            all_skeleton_data.append(skeleton_data.cpu().numpy())

    return np.concatenate(all_skeleton_data, axis=0)  # Concatenate results to form the full output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to your pre-trained model weights
    sensor_model_path = "./results/sensor_model/best_sensor_model.pth"
    diffusion_model_path = "./results/diffusion_model/best_diffusion_model.pth"

    # Load models
    sensor_model, diffusion_model = load_models(sensor_model_path, diffusion_model_path, device)

    # Sample raw accelerometer data (Replace this with actual input data)
    raw_accelerometer_data = np.random.rand(1000, 3)  # Dummy data, replace with actual input

    # Generate skeleton data
    skeleton_output = generate_skeleton(sensor_model, diffusion_model, raw_accelerometer_data, device)

    print("Generated Skeleton Data:")
    print(skeleton_output)
