import torch
import cv2
import numpy as np
import argparse
import os
from model import create_unet


def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.6f}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    return model


def hsv_to_bgr(hsv_frame):
    """Convert HSV frame to BGR for display"""
    # Transpose from (C, H, W) to (H, W, C)
    hsv_frame = hsv_frame.transpose(1, 2, 0)
    
    # Denormalize from [0, 1] to [0, 255]
    hsv_frame = (hsv_frame * 255).astype(np.uint8)
    
    # Convert HSV to BGR
    bgr_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    return bgr_frame


def generate_video(model, initial_frames, num_frames=100, device='cpu'):
    """Generate video by iteratively predicting next frames with attention masks"""
    model.eval()
    
    # Start with initial frames
    current_sequence = initial_frames.clone()
    generated_frames = []
    attention_masks = []
    
    print(f"Generating {num_frames} frames...")
    
    with torch.no_grad():
        for i in range(num_frames):
            # Predict next frame
            next_frame, attention_mask = model(current_sequence.unsqueeze(0).to(device))
            next_frame = next_frame.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            
            # Add to generated frames and attention masks (move to CPU for numpy conversion)
            generated_frames.append(next_frame.cpu().numpy())
            attention_masks.append(attention_mask.cpu().numpy())
            
            # Update sequence: remove oldest frame, add predicted frame (keep on device)
            current_sequence = torch.cat([current_sequence[3:], next_frame], dim=0)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_frames} frames")
    
    return generated_frames, attention_masks


def save_video(frames, attention_masks, output_path='generated_video.mp4', frame_rate=6):
    """Save generated frames with attention masks as video file"""
    if not frames:
        print("No frames to save")
        return
    
    # Get frame dimensions
    frame_bgr = hsv_to_bgr(frames[0])
    height, width = frame_bgr.shape[:2]
    
    # Create video writer with H.264 codec (triple width for frame + attention)
    if output_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # For .avi files
    
    # Triple the width to accommodate frame + attention mask
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width * 2, height))
    
    if not out.isOpened():
        print("Error: Could not open video writer. Trying alternative codec...")
        # Fallback to XVID
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width * 2, height))
    
    print(f"Saving video to {output_path} at {frame_rate} FPS...")
    
    for i, (frame, attention_mask) in enumerate(zip(frames, attention_masks)):
        # Convert frame to BGR
        frame_bgr = hsv_to_bgr(frame)
        
        # Convert attention mask to colored visualization
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Normalize attention mask and apply colormap
        attention_squeezed = attention_mask.squeeze(0)  # Remove channel dimension
        attention_norm = (attention_squeezed - attention_squeezed.min()) / (attention_squeezed.max() - attention_squeezed.min() + 1e-8)
        attention_colored = cm.hot(attention_norm)
        attention_rgb = (attention_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        attention_bgr = cv2.cvtColor(attention_rgb, cv2.COLOR_RGB2BGR)
        
        # Concatenate frame and attention mask horizontally
        combined_frame = np.concatenate([frame_bgr, attention_bgr], axis=1)
        
        # Ensure frame is in correct format
        combined_frame = combined_frame.astype(np.uint8)
        out.write(combined_frame)
        
        if (i + 1) % 10 == 0:
            print(f"Saved {i + 1}/{len(frames)} frames")
    
    out.release()
    print(f"Video saved to {output_path} at {frame_rate} FPS")


def main():
    parser = argparse.ArgumentParser(description='Generate video using trained UNET model')
    parser.add_argument('--checkpoint', default='checkpoints/latest.pth', 
                       help='Path to checkpoint file')
    parser.add_argument('--frame-width', type=int, default=96, 
                       help='Frame width')
    parser.add_argument('--frame-height', type=int, default=64, 
                       help='Frame height')
    parser.add_argument('--model-size', type=int, default=16, 
                       help='Model size')
    parser.add_argument('--num-frames', type=int, default=100, 
                       help='Number of frames to generate')
    parser.add_argument('--frame-rate', type=int, default=6, 
                       help='Frame rate for video')
    parser.add_argument('--output-path', default='output/generated_video.mp4', 
                       help='Output video path')
    parser.add_argument('--source-video', type=int, default=0, 
                       help='Index of source video to sample initial frames from (0-2)')
    
    args = parser.parse_args()
    
    # Device setup
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = create_unet(frame_width=args.frame_width, frame_height=args.frame_height, 
                       model_size=args.model_size)
    
    # Move model to device first
    model = model.to(device)
    
    # Load checkpoint
    try:
        model = load_checkpoint(args.checkpoint, model, device)
    except FileNotFoundError:
        print(f"Checkpoint file not found: {args.checkpoint}")
        return
    
    # Import video URLs from train.py
    from train import youtube_video_urls
    
    # Sample initial frames from the specified video
    print(f"Sampling initial frames from video {args.source_video}...")
    source_url = youtube_video_urls[args.source_video]
    
    # Download video if not already downloaded
    from train import download_youtube_video, video_to_hsv_frames
    video_path = download_youtube_video(source_url)
    
    # Load frames
    frames = video_to_hsv_frames(video_path, args.frame_width, args.frame_height, target_fps=6)
    
    if len(frames) < 2:
        print("Error: Not enough frames in source video")
        return
    
    # Sample 2 consecutive frames from a random point in the video
    import random
    max_start_point = len(frames) - 2  # Ensure we have 2 consecutive frames
    start_point = random.randint(0, max_start_point)
    frame1 = frames[start_point]
    frame2 = frames[start_point + 1]
    
    # Concatenate the 2 frames (each has 3 channels, so total 6 channels)
    initial_frames = np.concatenate([frame1, frame2], axis=0)
    initial_frames = torch.from_numpy(initial_frames).float().to(device)
    
    # Generate video
    generated_frames, attention_masks = generate_video(model, initial_frames, args.num_frames, device)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save video
    print("Saving generated video...")
    save_video(generated_frames, attention_masks, args.output_path, args.frame_rate)


if __name__ == "__main__":
    main()
