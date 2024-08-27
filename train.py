import os
import torch
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from diffusion_model.diffusion import DiffusionProcess, Scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusion_model.model_loader import load_sensor_model, load_diffusion
from diffusion_model.util import (
    prepare_dataset,
    compute_loss,
)

def ensure_dir(path, rank):
    if rank == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
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

def train_sensor_model(rank, args, device, train_loader, val_loader):
    print("Training Sensor model")
    sensor_model = load_sensor_model(args, device)
    sensor_model = DDP(sensor_model, device_ids=[rank], find_unused_parameters=True)

    sensor_optimizer = torch.optim.Adam(
        sensor_model.parameters(),
        lr=args.sensor_lr,
        betas=(0.9, 0.98)
    )

    sensor_model_save_dir = os.path.join(args.output_dir, "sensor_model")
    ensure_dir(sensor_model_save_dir, rank)

    sensor_log_dir = os.path.join(sensor_model_save_dir, "sensor_logs")
    ensure_dir(sensor_log_dir, rank)
    
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

def train_diffusion_model(rank, args, device, train_loader, val_loader):
    print("Training Diffusion model")
    sensor_model = load_sensor_model(args, device)
    diffusion_model = load_diffusion(device)

    sensor_model = DDP(sensor_model, device_ids=[rank], find_unused_parameters=True)
    diffusion_model = DDP(diffusion_model, device_ids=[rank], find_unused_parameters=True)

    diffusion_optimizer = optim.Adam(
        diffusion_model.parameters(),
        lr=1e-5,
        eps=1e-8,
        betas=(0.9, 0.98)
    )

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(diffusion_optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Setup output directories
    diffusion_model_save_dir = os.path.join(args.output_dir, "diffusion_model")
    ensure_dir(diffusion_model_save_dir, rank)

    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=diffusion_model_save_dir)

    best_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()

    # Initialize the diffusion process with the scheduler
    diffusion_process = DiffusionProcess(
        scheduler=Scheduler(sched_type='cosine', T=args.timesteps, step=1, device=device),
        device=device,
        ddim_scale=args.ddim_scale
    )

    for epoch in range(args.epochs):
        diffusion_model.train()
        sensor_model.train()
        epoch_train_loss = 0.0

        for skeleton, sensor1, sensor2, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Training)"):
            skeleton, sensor1, sensor2, mask = skeleton.to(device), sensor1.to(device), sensor2.to(device), mask.to(device)

            t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
            context = sensor_model(sensor1, sensor2, return_attn_output=True)
            diffusion_optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Enable mixed precision
                loss = compute_loss(
                    args=args,
                    model=diffusion_model,
                    x0=skeleton,
                    context=context,
                    t=t,
                    mask=mask,
                    device=device,
                    diffusion_process=diffusion_process,
                    angular_loss=args.angular_loss,
                    epoch=epoch
                )

            scaler.scale(loss).backward()

            # Gradient Clipping (optional)
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)

            scaler.step(diffusion_optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        diffusion_model.eval()
        sensor_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for skeleton, sensor1, sensor2, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Validation)"):
                skeleton, sensor1, sensor2, mask = skeleton.to(device), sensor1.to(device), sensor2.to(device), mask.to(device)
                t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()

                with torch.cuda.amp.autocast():  # Enable mixed precision
                    loss = compute_loss(
                        args=args,
                        model=diffusion_model,
                        x0=skeleton,
                        context=sensor_model(sensor1, sensor2, return_attn_output=True),
                        t=t,
                        mask=mask,
                        device=device,
                        diffusion_process=diffusion_process,
                        angular_loss=args.angular_loss,
                        epoch=epoch
                    )
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        # scheduler.step(avg_val_loss)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(diffusion_model.state_dict(), os.path.join(diffusion_model_save_dir, f"best_diffusion_model.pth"))
                print(f"Saved best diffusion model with Validation Loss: {best_loss}")

def train_diffusion_model_with_transfer(rank, args, device, train_loader, val_loader):
    print("Training Diffusion model with Transfer Learning")
    sensor_model = load_sensor_model(args, device)
    diffusion_model = load_diffusion(device)

    # Load the pre-trained model weights and adjust the state dict
    checkpoint = torch.load('results/diffusion_model/best_diffusion_model.pth')
    state_dict = checkpoint
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = state_dict[key]

    # Load the modified state dict
    diffusion_model.load_state_dict(new_state_dict, strict=False)

    # Freeze the encoder layers until the cross-attention
    for name, param in diffusion_model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
        elif 'cross_attn' in name:
            break  # Stop freezing when cross-attention is reached

    sensor_model = DDP(sensor_model, device_ids=[rank], find_unused_parameters=True)
    diffusion_model = DDP(diffusion_model, device_ids=[rank], find_unused_parameters=True)

    diffusion_optimizer = optim.Adam(
        diffusion_model.parameters(),
        lr=1e-5,
        eps=1e-8,
        betas=(0.9, 0.98)
    )

    # Training loop remains mostly the same
    best_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()
    diffusion_model_save_dir = os.path.join(args.output_dir, "diffusion_model_transfer")
    ensure_dir(diffusion_model_save_dir, rank)
    writer = SummaryWriter(log_dir=diffusion_model_save_dir) if rank == 0 else None

    diffusion_process = DiffusionProcess(
        scheduler=Scheduler(sched_type='cosine', T=args.timesteps, step=1, device=device),
        device=device,
        ddim_scale=args.ddim_scale
    )

    for epoch in range(args.epochs):
        diffusion_model.train()
        sensor_model.train()
        epoch_train_loss = 0.0

        for skeleton, sensor1, sensor2, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Training)"):
            skeleton, sensor1, sensor2, mask = skeleton.to(device), sensor1.to(device), sensor2.to(device), mask.to(device)
            t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
            context = sensor_model(sensor1, sensor2, return_attn_output=True)
            diffusion_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = compute_loss(
                    args=args,
                    model=diffusion_model,
                    x0=skeleton,
                    context=context,
                    t=t,
                    mask=mask,
                    device=device,
                    diffusion_process=diffusion_process,
                    angular_loss=args.angular_loss,
                    epoch=epoch
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, diffusion_model.parameters()), max_norm=1.0)
            scaler.step(diffusion_optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        diffusion_model.eval()
        sensor_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for skeleton, sensor1, sensor2, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Validation)"):
                skeleton, sensor1, sensor2, mask = skeleton.to(device), sensor1.to(device), sensor2.to(device), mask.to(device)
                t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()

                with torch.cuda.amp.autocast():
                    loss = compute_loss(
                        args=args,
                        model=diffusion_model,
                        x0=skeleton,
                        context=sensor_model(sensor1, sensor2, return_attn_output=True),
                        t=t,
                        mask=mask,
                        device=device,
                        diffusion_process=diffusion_process,
                        angular_loss=args.angular_loss,
                        epoch=epoch
                    )
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(diffusion_model.state_dict(), os.path.join(diffusion_model_save_dir, f"best_diffusion_model_transfer.pth"))
                print(f"Saved best diffusion model with Validation Loss: {best_loss}")

def main(rank, args):
    setup(rank, args.world_size, seed=42)
    device = torch.device(f'cuda:{rank}')

    # Prepare the full dataset
    dataset = prepare_dataset(args)

    # Extract labels for stratified splitting
    labels = [dataset[i][3] for i in range(len(dataset))]

    # Simplify labels if they are multi-dimensional (e.g., one-hot encoded)
    if isinstance(labels[0], (list, torch.Tensor, np.ndarray)):
        labels = [torch.argmax(torch.tensor(label)).item() if isinstance(label, torch.Tensor) else np.argmax(label) for label in labels]

    # Perform the stratified split
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_test_idx = next(stratified_split.split(range(len(dataset)), labels))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_test_idx)

    # Use DistributedSampler for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, drop_last=True)

    if args.train_sensor_model:
        train_sensor_model(rank, args, device, train_loader, val_loader)
    elif args.transfer_learn:
        train_diffusion_model_with_transfer(rank, args, device, train_loader, val_loader)
    else:
        train_diffusion_model(rank, args, device, train_loader, val_loader)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training for Diffusion and Sensor Models")
    parser.add_argument("--skeleton_folder", type=str, default="./Own_Data/Labelled_Student_data/Skeleton_Data", help="Path to the skeleton data folder")
    parser.add_argument("--sensor_folder1", type=str, default="./Own_Data/Labelled_Student_data/Accelerometer_Data/Meta_wrist", help="Path to the first sensor data folder")
    parser.add_argument("--sensor_folder2", type=str, default="./Own_Data/Labelled_Student_data/Accelerometer_Data/Meta_hip", help="Path to the second sensor data folder")
    parser.add_argument("--sensor_model_path", type=str, default="./models/sensor_model.pth", help="Path to the pre-trained sensor model")
    parser.add_argument("--window_size", type=int, default=90, help="Window size for the sliding window dataset")
    parser.add_argument("--overlap", type=int, default=70, help="Overlap for the sliding window dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--sensor_lr", type=float, default=1e-3, help="Weight decay for sensor regularization")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate for scheduler")
    parser.add_argument("--sensor_epoch", type=int, default=500, help="Number of epochs to train the sensor model")
    parser.add_argument("--epochs", type=int, default=3000, help="Number of epochs to train the diffusion model")
    parser.add_argument("--step_size", type=int, default=20, help="Step size for weight decay")
    parser.add_argument("--world_size", type=int, default=8, help="Number of GPUs to use for training")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the trained model")
    parser.add_argument("--dataset_type", type=str, default="Own_data", help="Dataset type")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of timesteps for the diffusion process")
    parser.add_argument("--angular_loss", type=eval, choices=[True, False], default=False, help="Whether to use angular loss during training")
    parser.add_argument("--train_sensor_model", type=eval, choices=[True, False], default=False, help="Set to True to train the sensor model; set to False to train the diffusion model")
    parser.add_argument("--augment",type=eval, choices=[True, False], default=True, help="Flag to determine whether data augmentation needed to be done or not")
    parser.add_argument("--lip_reg",type=eval, choices=[True, False], default=True, help="Flag to determine whether to inlcude LR or not")
    parser.add_argument("--predict_noise",type=eval, choices=[True, False], default=False, help="Flag to determine whether to inlcude LR or not")
    parser.add_argument('--ddim_scale', type=float, default=0.5, help='Scale factor for DDIM (0 for pure DDIM, 1 for pure DDPM)')
    parser.add_argument('--transfer_learn', type=eval, choices=[True, False], default=False, help='Set to True for transfer learning')

    args = parser.parse_args()

    mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
