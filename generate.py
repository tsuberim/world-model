import torch
import cv2
import numpy as np
import argparse
from model import create_unet


def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
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
    """Generate video by iteratively predicting next frames"""
    model.eval()
    
    # Start with initial frames
    current_sequence = initial_frames.clone()
    generated_frames = []
    
    print(f"Generating {num_frames} frames...")
    
    with torch.no_grad():
        for i in range(num_frames):
            # Predict next frame
            next_frame = model(current_sequence.unsqueeze(0).to(device))
            next_frame = next_frame.squeeze(0)
            
            # Add to generated frames (move to CPU for numpy conversion)
            generated_frames.append(next_frame.cpu().numpy())
            
            # Update sequence: remove oldest frame, add predicted frame (keep on device)
            current_sequence = torch.cat([current_sequence[3:], next_frame], dim=0)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_frames} frames")
    
    return generated_frames


def save_video(frames, output_path='generated_video.mp4', frame_rate=6):
    """Save generated frames as video file"""
    if not frames:
        print("No frames to save")
        return
    
    # Get frame dimensions
    frame_bgr = hsv_to_bgr(frames[0])
    height, width = frame_bgr.shape[:2]
    
    # Create video writer with H.264 codec
    if output_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # For .avi files
    
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open video writer. Trying alternative codec...")
        # Fallback to XVID
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    
    print(f"Saving video to {output_path} at {frame_rate} FPS...")
    
    for i, frame in enumerate(frames):
        frame_bgr = hsv_to_bgr(frame)
        # Ensure frame is in correct format
        frame_bgr = frame_bgr.astype(np.uint8)
        out.write(frame_bgr)
        
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
    
    # Create initial frames (random normalized HSV frames)
    print("Creating initial frames...")
    initial_frames = torch.rand(6, args.frame_height, args.frame_width, device=device)  # 2 HSV frames in [0, 1]
    
    # Generate video
    generated_frames = generate_video(model, initial_frames, args.num_frames, device)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save video
    print("Saving generated video...")
    save_video(generated_frames, args.output_path, args.frame_rate)


if __name__ == "__main__":
    main()
