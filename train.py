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


def video_to_hsv_frames(video_path, frame_width=96, frame_height=64):
    """Convert video to HSV frames"""
    print("Converting video to HSV frames...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Convert BGR to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Normalize to [0, 1]
        frame_hsv = frame_hsv.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W) format
        frame_hsv = frame_hsv.transpose(2, 0, 1)
        
        frames.append(frame_hsv)
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames


def create_sequences(frames, sequence_length=2):
    """Create sequences of frames for training"""
    sequences = []
    targets = []
    
    for i in range(len(frames) - sequence_length):
        # Get 2 previous frames
        sequence = frames[i:i+sequence_length]
        # Target is the next frame
        target = frames[i+sequence_length]
        
        # Concatenate sequence frames along channel dimension
        sequence_concatenated = np.concatenate(sequence, axis=0)
        
        sequences.append(sequence_concatenated)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda', learning_rate=0.001):
    """Train the model"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, 'checkpoints/latest.pth')
        print(f"Checkpoint saved for epoch {epoch+1}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train UNET model on video frames')
    parser.add_argument('--video-url', default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                       help='YouTube video URL to download')
    parser.add_argument('--frame-width', type=int, default=96, 
                       help='Frame width')
    parser.add_argument('--frame-height', type=int, default=64, 
                       help='Frame height')
    parser.add_argument('--model-size', type=int, default=16, 
                       help='Model size')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=10, 
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, 
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Device setup
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Download video
    video_path = download_youtube_video(args.video_url)
    
    # Convert to HSV frames
    frames = video_to_hsv_frames(video_path, args.frame_width, args.frame_height)
    
    # Create sequences
    sequences, targets = create_sequences(frames, sequence_length=2)
    
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
    
    # Create model
    model = create_unet(frame_width=args.frame_width, frame_height=args.frame_height, model_size=args.model_size)
    
    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("Starting training...")
    model = train_model(model, train_loader, test_loader, args.num_epochs, device, args.learning_rate)


if __name__ == "__main__":
    main()
