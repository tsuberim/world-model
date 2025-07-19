import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size mismatch for non-powers of 2
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, frame_width=64, frame_height=64, model_size=64):
        super().__init__()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model_size = model_size
        
        # Input: 2 HSV frames = 6 channels
        self.inc = DoubleConv(6, model_size)
        self.down1 = Down(model_size, model_size * 2)
        self.down2 = Down(model_size * 2, model_size * 4)
        self.down3 = Down(model_size * 4, model_size * 8)
        self.down4 = Down(model_size * 8, model_size * 16)
        self.down5 = Down(model_size * 16, model_size * 32)
        self.down6 = Down(model_size * 32, model_size * 64)
        
        # Bottleneck layer
        self.bottleneck = DoubleConv(model_size * 64, model_size * 64)
        
        self.up1 = Up(model_size * 64, model_size * 32)
        self.up2 = Up(model_size * 32, model_size * 16)
        self.up3 = Up(model_size * 16, model_size * 8)
        self.up4 = Up(model_size * 8, model_size * 4)
        self.up5 = Up(model_size * 4, model_size * 2)
        self.up6 = Up(model_size * 2, model_size)

        
        # Output: 1 HSV frame = 3 channels
        self.outc = nn.Conv2d(model_size, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, 6, height, width) - 2 HSV frames concatenated
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        
        # Apply bottleneck
        x7 = self.bottleneck(x7)
        
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        
        # Output: (batch, 3, height, width) - 1 HSV frame
        return self.sigmoid(self.outc(x))


def create_unet(frame_width=64, frame_height=64, model_size=64):
    """Create a UNET model with specified frame and model sizes."""
    return UNet(frame_width=frame_width, frame_height=frame_height, model_size=model_size)


# Example usage
if __name__ == "__main__":
    batch_size = 2
    # Create model with custom sizes
    custom_model = create_unet(frame_width=96, frame_height=64, model_size=128)
    custom_frames = torch.randn(batch_size, 6, 96, 64)
    custom_output = custom_model(custom_frames)
    print(f"Custom input shape: {custom_frames.shape}")
    print(f"Custom output shape: {custom_output.shape}")
