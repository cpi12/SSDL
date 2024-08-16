import os
import shutil
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from adabelief_pytorch import AdaBelief
from diffusion_model.pipeline import generate
from diffusion_model.model_loader import load_sensor_model, load_diffusion
from diffusion_model.diffusion import DiffusionProcess
from diffusion_model.util import (
    get_alpha,
    prepare_dataset,
    compute_loss,
)

def ensure_dir(path, rank):
    # Only rank 0 should handle directory creation and deletion
    if rank == 0:
        if os.path.exists(path):
            shutil.rmtree(path)  # Remove the existing directory and its contents
        os.makedirs(path)  # Create a fresh directory
    # Synchronize all processes to ensure they wait until rank 0 finishes setup
    dist.barrier()

def setup(rank, world_size, seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup():
    dist.destroy_process_group()

def train_sensor_model(rank, args, device, train_dataset, train_loader, val_loader):
    print("Training Sensor model")
    sensor_model = load_sensor_model(args, device)
    sensor_model = DDP(sensor_model, device_ids=[rank], find_unused_parameters=True)

    sensor_optimizer = torch.optim.Adam(
        sensor_model.parameters(),
        lr=args.sensor_lr,
        betas=(0.9, 0.98)
    )

    sensor_model_save_dir = os.path.join(args.output_dir, "sensor_model")
    ensure_dir(sensor_model_save_dir, rank)  # Pass the rank to handle directory creation

    sensor_log_dir = os.path.join(sensor_model_save_dir, "sensor_logs")
    ensure_dir(sensor_log_dir, rank)  # Pass the rank to handle directory creation
    
    if rank == 0:
        writer = SummaryWriter(log_dir=sensor_log_dir)

    best_loss = float('inf')

    for epoch in range(args.sensor_epoch):
        sensor_model.train()
        epoch_train_loss = 0.0

        for _, sensor1, sensor2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.sensor_epoch} (Training)"):
            sensor1, sensor2, labels = sensor1.to(device), sensor2.to(device), labels.to(device)

            sensor_optimizer.zero_grad()
            output = sensor_model(sensor1, sensor2)
            loss = torch.nn.CrossEntropyLoss()(output, labels.argmax(dim=1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sensor_model.parameters(), max_norm=1.0)
            sensor_optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation phase
        sensor_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for _, sensor1, sensor2, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.sensor_epoch} (Validation)"):
                sensor1, sensor2, labels = sensor1.to(device), sensor2.to(device), labels.to(device)
                output = sensor_model(sensor1, sensor2)
                loss = torch.nn.CrossEntropyLoss()(output, labels.argmax(dim=1))

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.sensor_epoch}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(sensor_model.state_dict(), os.path.join(sensor_model_save_dir, "best_sensor_model.pth"))
                print(f"Saved best sensor model with Validation Loss: {best_loss}")


def train_diffusion_model(rank, args, device, train_dataset, train_loader, val_loader):
    print("Training Diffusion model")
    sensor_model = load_sensor_model(args, device)
    diffusion_model = load_diffusion(device)

    sensor_model = DDP(sensor_model, device_ids=[rank], find_unused_parameters=True)
    diffusion_model = DDP(diffusion_model, device_ids=[rank], find_unused_parameters=True)

    diffusion_optimizer = torch.optim.AdamW(
        diffusion_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    output_dir = os.path.join(args.output_dir, "diffusion_model")
    ensure_dir(output_dir)

    writer = SummaryWriter(log_dir=output_dir)

    best_loss = float('inf')

    diffusion_process = DiffusionProcess(timesteps=args.timesteps, schedule_type='linear')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(diffusion_optimizer, T_0=10, T_mult=2)

    for epoch in range(args.epochs):
        diffusion_model.train()
        sensor_model.train()
        epoch_train_loss = 0.0

        for skeleton, sensor1, sensor2, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Training)"):
            skeleton, sensor1, sensor2 = skeleton.to(device), sensor1.to(device), sensor2.to(device)

            t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
            context = sensor_model(sensor1, sensor2, return_attn_output=True)

            loss = compute_loss(args, diffusion_model, skeleton, context, t, epoch, device=device, diffusion_process=diffusion_process, angular_loss=args.angular_loss)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            diffusion_optimizer.step()
            diffusion_optimizer.zero_grad()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation phase
        diffusion_model.eval()
        sensor_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for skeleton, sensor1, sensor2, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Validation)"):
                skeleton, sensor1, sensor2 = skeleton.to(device), sensor1.to(device), sensor2.to(device)
                t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
                loss = compute_loss(args, diffusion_model, skeleton, sensor_model(sensor1, sensor2, return_attn_output=True), t, epoch, device=device, diffusion_process=diffusion_process, angular_loss=args.angular_loss)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        scheduler.step(epoch)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(diffusion_model.state_dict(), os.path.join(output_dir, f"best_diffusion_model.pth"))
                print(f"Saved best diffusion model with Validation Loss: {best_loss}")

def main(rank, args):
    setup(rank, args.world_size, seed=42)
    device = torch.device(f'cuda:{rank}')

    # Prepare the full dataset
    dataset = prepare_dataset(args)

    # Perform the train/validation/test split
    total_indices = list(range(len(dataset)))
    train_idx, val_test_idx = train_test_split(total_indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=42)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Use DistributedSampler for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, drop_last=True)

    if args.train_sensor_model:
        train_sensor_model(rank, args, device, train_dataset, train_loader, val_loader)
    else:
        train_diffusion_model(rank, args, device, train_dataset, train_loader, val_loader)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training for Diffusion and Sensor Models")
    parser.add_argument("--skeleton_folder", type=str, default="./Own_Data/Labelled_Student_data/Skeleton_Data", help="Path to the skeleton data folder")
    parser.add_argument("--sensor_folder1", type=str, default="./Own_Data/Labelled_Student_data/Accelerometer_Data/Meta_wrist", help="Path to the first sensor data folder")
    parser.add_argument("--sensor_folder2", type=str, default="./Own_Data/Labelled_Student_data/Accelerometer_Data/Meta_hip", help="Path to the second sensor data folder")
    parser.add_argument("--sensor_model_path", type=str, default="./models/sensor_model.pth", help="Path to the pre-trained sensor model")
    parser.add_argument("--window_size", type=int, default=90, help="Window size for the sliding window dataset")
    parser.add_argument("--overlap", type=int, default=85, help="Overlap for the sliding window dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--sensor_lr", type=float, default=1e-3, help="Weight decay for sensor regularization")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate for scheduler")
    parser.add_argument("--sensor_epoch", type=int, default=100, help="Number of epochs to train the sensor model")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs to train the diffusion model")
    parser.add_argument("--step_size", type=int, default=20, help="Step size for weight decay")
    parser.add_argument("--world_size", type=int, default=8, help="Number of GPUs to use for training")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the trained model")
    parser.add_argument("--dataset_type", type=str, default="Own_data", help="Dataset type")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of timesteps for the diffusion process")
    parser.add_argument("--angular_loss", type=eval, choices=[True, False], default=False, help="Whether to use angular loss during training")
    parser.add_argument("--train_sensor_model", type=eval, choices=[True, False], default=False, help="Set to True to train the sensor model; set to False to train the diffusion model")
    parser.add_argument("--augment",type=eval, choices=[True, False], default=False, help="Flag to determine whether data augmentation needed to be done or not")
    args = parser.parse_args()

    mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
