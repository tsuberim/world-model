import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import yt_dlp
import os
import argparse
from sklearn.model_selection import train_test_split
from model import create_unet
import wandb
from tqdm import tqdm

class VideoFrameDataset(Dataset):
    def __init__(self, frame_sequences, target_frames):
        self.frame_sequences = frame_sequences
        self.target_frames = target_frames
    
    def __len__(self):
        return len(self.frame_sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.frame_sequences[idx]), torch.FloatTensor(self.target_frames[idx])

def download_youtube_video(url, output_dir="videos"):
    """Download YouTube video"""
    # Extract video hash from URL
    video_hash = url.split('v=')[1].split('&')[0]
    output_path = os.path.join(output_dir, f"{video_hash}.mp4")
    
    # Create videos directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading video from {url}...")
    ydl_opts = {
        'format': 'mp4[height<=480]',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Video downloaded to {output_path}")
    return output_path


def video_to_hsv_frames(video_path, frame_width=96, frame_height=64, target_fps=6):
    """Convert video to HSV frames at target FPS"""
    # Create frames directory if it doesn't exist
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate cache filename based on video path and parameters
    video_hash = os.path.basename(video_path).split('.')[0]
    cache_filename = f"{frames_dir}/{video_hash}_{frame_width}x{frame_height}_{target_fps}fps.npy"
    
    # Check if cached frames exist
    if os.path.exists(cache_filename):
        print(f"Loading cached frames from {cache_filename}")
        frames = np.load(cache_filename)
        print(f"Loaded {len(frames)} cached frames")
        return frames
    
    print("Converting video to HSV frames...")
    
    # Check if video file exists
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
    except Exception as e:
        print(f"Error: {e}")
        return []

    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle cases where FPS is 0 (e.g., some image sequences)
    if original_fps == 0:
        print("Warning: Original FPS is 0. Cannot perform subsampling. Processing all frames.")
        original_fps = target_fps # Assume target FPS to avoid division by zero

    print(f"Original video: {total_frames} frames at {original_fps:.2f} FPS")
    print(f"Target FPS: {target_fps}")
    
    if original_fps < target_fps:
        print(f"Warning: Original video FPS ({original_fps:.2f}) is lower than target FPS ({target_fps}). Frame duplication will occur.")

    frames = []
    frame_count = 0
    selected_frame_count = 0
    
    # Create progress bar for frame extraction
    pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate the timestamp of the current frame in the original video
        current_time = frame_count / original_fps
        
        # Calculate the timestamp of the next frame we *want* in the target video
        next_target_time = selected_frame_count / target_fps
        
        # If the original frame's time is at or after the time we need for our
        # target video, we select this frame.
        if current_time >= next_target_time:
            # Resize frame
            resized_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            
            # Convert BGR to HSV
            frame_hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
            
            # Normalize to [0, 1]
            frame_hsv = frame_hsv.astype(np.float32) / 255.0
            
            # Transpose to (C, H, W) format
            frame_hsv = frame_hsv.transpose(2, 0, 1)
            
            frames.append(frame_hsv)
            selected_frame_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Convert to numpy array and save
    frames = np.array(frames)
    print(f"Saving {len(frames)} frames to {cache_filename}")
    np.save(cache_filename, frames)
    print(f"Successfully extracted and cached {len(frames)} frames to match {target_fps} FPS.")
    return frames


def create_sequences(frames, sequence_length=4):
    """Create sequences of frames for training"""
    sequences = []
    targets = []
    
    for i in range(len(frames) - sequence_length):
        # Get 4 consecutive frames
        sequence = frames[i:i+sequence_length]
        # Target is the 4th frame
        target = frames[i+sequence_length-1]
        
        # Concatenate sequence frames along channel dimension
        sequence_concatenated = np.concatenate(sequence, axis=0)
        
        sequences.append(sequence_concatenated)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def hsv_to_rgb(hsv_frame):
    """Convert HSV frame to RGB for wandb"""
    # Transpose from (C, H, W) to (H, W, C)
    hsv_frame = hsv_frame.transpose(1, 2, 0)
    
    # Denormalize from [0, 1] to [0, 255]
    hsv_frame = (hsv_frame * 255).astype(np.uint8)
    
    # Convert HSV to RGB
    rgb_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
    return rgb_frame


def train_single_epoch(model, train_loader, val_loader, optimizer, device='cuda'):
    """Train the model for a single epoch on one video"""
    model = model.to(device)
    criterion = nn.MSELoss()
    
    # Training
    model.train()
    train_loss = 0.0
    train_pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (sequences, targets) in enumerate(train_pbar):
        sequences, targets = sequences.to(device), targets.to(device)
        
        optimizer.zero_grad()
        loss, outputs = compute_loss(model, sequences, targets, criterion)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Validation
    model.eval()
    val_loss = 0.0
    sample_predictions = []
    sample_targets = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        for sequences, targets in val_pbar:
            sequences, targets = sequences.to(device), targets.to(device)
            loss, outputs = compute_loss(model, sequences, targets, criterion)
            val_loss += loss.item()
            
            # Collect sample predictions for visualization
            if len(sample_predictions) < 4:  # Save first 4 samples
                for i in range(min(4 - len(sample_predictions), len(outputs))):
                    sample_predictions.append(outputs[i].cpu().numpy())
                    sample_targets.append(targets[i].cpu().numpy())
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f'Video training complete: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_loss, val_loss, (sample_predictions, sample_targets) if sample_predictions else None

def compute_loss(model, sequences, targets, criterion, mixing_rate = 0.85):
    loss = 0.0
    
    first_and_second_frames = sequences[:, :6, :, :]
    second_and_third_frames = sequences[:, 3:9, :, :]
    third_and_fourth_frames = sequences[:, 6:, :, :]

    third_frame = sequences[:, 6:9, :, :]
    fourth_frame = sequences[:, 9:, :, :]

    # predict 3rd frame using first and second frames
    first_and_second_frames_output = model(first_and_second_frames)
    loss += criterion(first_and_second_frames_output, third_frame)

    # predict 4th frame using second and third frames (mixed with 3rd frame prediction)
    mixed_middle_frame = (1-mixing_rate)*second_and_third_frames[:, 3:, :, :] + (mixing_rate)*first_and_second_frames_output
    middle_input_frames = torch.cat([second_and_third_frames[:, :3, :, :], mixed_middle_frame], dim=1)
    second_and_third_frames_output = model(middle_input_frames)
    loss += criterion(second_and_third_frames_output, fourth_frame)

    # predict 5th frame using 3rd and 4th frames (mixed with 4th and 3rd frame prediction)
    mixed_frame_3 = (1-mixing_rate)*third_and_fourth_frames[:, :3, :, :] + (mixing_rate)*first_and_second_frames_output
    mixed_frame_4 = (1-mixing_rate)*third_and_fourth_frames[:, 3:, :, :] + (mixing_rate)*second_and_third_frames_output
    input_frames = torch.cat([mixed_frame_3, mixed_frame_4], dim=1)
    target_prediction = model(input_frames)
    loss += criterion(target_prediction, targets)

    return loss, target_prediction


youtube_video_urls = [
    "https://www.youtube.com/watch?v=V-mwW526RVU",
    "https://www.youtube.com/watch?v=Tyi5S76cZ-s",
    "https://www.youtube.com/watch?v=d_WFTzQYJ7g",
    # "https://www.youtube.com/watch?v=0fts6x_EE_E",
    # "https://www.youtube.com/watch?v=BSaOoL6Dte4",
    # "https://www.youtube.com/watch?v=ICJvn8h4I7g"
]

def main():
    parser = argparse.ArgumentParser(description='Train UNET model on video frames')
    parser.add_argument('--video-urls', nargs='+', 
                       default=youtube_video_urls,
                       help='List of YouTube video URLs to download')
    parser.add_argument('--frame-width', type=int, default=96, 
                       help='Frame width')
    parser.add_argument('--frame-height', type=int, default=64, 
                       help='Frame height')
    parser.add_argument('--model-size', type=int, default=16, 
                       help='Model size')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, 
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Device setup
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = create_unet(frame_width=args.frame_width, frame_height=args.frame_height, model_size=args.model_size)
    
    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load latest checkpoint if exists
    start_epoch = 0
    wandb_run_id = None
    if os.path.exists('checkpoints/latest.pth'):
        print("Loading latest checkpoint to resume training...")
        checkpoint = torch.load('checkpoints/latest.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Move optimizer state to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        start_epoch = checkpoint['epoch']
        wandb_run_id = checkpoint.get('wandb_run_id')
        print(f"Resuming from epoch {start_epoch}")
        if wandb_run_id:
            print(f"Resuming wandb run: {wandb_run_id}")
    
    # Initialize wandb
    if wandb_run_id:
        # Resume existing run
        wandb.init(project="world-model", id=wandb_run_id, resume="must")
    else:
        # Start new run
        wandb.init(project="world-model", config={
            "model_size": model.model_size,
            "frame_width": model.frame_width,
            "frame_height": model.frame_height,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
        })
    
    # Train on each video
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{args.num_epochs}")
        print(f"{'='*50}")
        
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        epoch_sample_predictions = []
        epoch_sample_targets = []
        videos_processed = 0
        
        for video_idx, video_url in enumerate(args.video_urls):
            print(f"\nTraining on video {video_idx + 1}/{len(args.video_urls)}: {video_url}")
            
            # Download video
            video_path = download_youtube_video(video_url)
            
            # Convert to HSV frames
            frames = video_to_hsv_frames(video_path, args.frame_width, args.frame_height, target_fps=6)
            
            if len(frames) == 0:
                print(f"Skipping video {video_idx + 1} due to processing error")
                continue
            
            # Create sequences
            sequences, targets = create_sequences(frames, sequence_length=4)
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                sequences, targets, test_size=0.15, random_state=42
            )
            
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            
            # Create datasets and dataloaders
            train_dataset = VideoFrameDataset(X_train, y_train)
            test_dataset = VideoFrameDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Train model on this video for one epoch
            video_train_loss, video_val_loss, video_samples = train_single_epoch(
                model, train_loader, test_loader, optimizer, device
            )
            
            # Accumulate losses and samples
            epoch_train_loss += video_train_loss
            epoch_val_loss += video_val_loss
            if video_samples:
                epoch_sample_predictions.extend(video_samples[0])
                epoch_sample_targets.extend(video_samples[1])
            
            videos_processed += 1
        
        # Calculate average losses for the epoch
        if videos_processed > 0:
            epoch_train_loss /= videos_processed
            epoch_val_loss /= videos_processed
        
        print(f'\nEpoch {epoch+1}/{args.num_epochs} Summary: Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}')
        
        # Log to wandb
        wandb.log({
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
        })
        
        # Log sample predictions every 5 epochs
        wandb_images = []
        for i, (pred, target) in enumerate(zip(epoch_sample_predictions, epoch_sample_targets)):
            # Convert to RGB for wandb
            pred_rgb = hsv_to_rgb(pred)
            target_rgb = hsv_to_rgb(target)
            
            # Create side-by-side comparison
            comparison = np.concatenate([target_rgb, pred_rgb], axis=1)
            wandb_images.append(wandb.Image(comparison, caption=f"Sample {i+1}: Target | Prediction"))
        
        wandb.log({"sample_predictions": wandb_images})
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'wandb_run_id': wandb.run.id,
        }
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, 'checkpoints/latest.pth')
        print(f"Checkpoint saved for epoch {epoch+1}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
