import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size mismatch for non-powers of 2
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, frame_width=64, frame_height=64, model_size=64, dropout_rate=0.1, weight_decay=1e-4):
        super().__init__()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model_size = model_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        # Input: 2 HSV frames = 6 channels
        self.inc = DoubleConv(6, model_size, dropout_rate)
        self.down1 = Down(model_size, model_size * 2, dropout_rate)
        self.down2 = Down(model_size * 2, model_size * 4, dropout_rate)
        self.down3 = Down(model_size * 4, model_size * 8, dropout_rate)
        self.down4 = Down(model_size * 8, model_size * 16, dropout_rate)
        self.down5 = Down(model_size * 16, model_size * 32, dropout_rate)
        self.down6 = Down(model_size * 32, model_size * 64, dropout_rate)
        
        # Bottleneck layer
        self.bottleneck = DoubleConv(model_size * 64, model_size * 64, dropout_rate)
        
        self.up1 = Up(model_size * 64, model_size * 32, dropout_rate)
        self.up2 = Up(model_size * 32, model_size * 16, dropout_rate)
        self.up3 = Up(model_size * 16, model_size * 8, dropout_rate)
        self.up4 = Up(model_size * 8, model_size * 4, dropout_rate)
        self.up5 = Up(model_size * 4, model_size * 2, dropout_rate)
        self.up6 = Up(model_size * 2, model_size, dropout_rate)

        
        # Output: 1 HSV frame = 3 channels + 1 attention mask
        self.outc = nn.Conv2d(model_size, 3, kernel_size=1)
        self.attention_out = nn.Conv2d(model_size, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights with Xavier/Glorot initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

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
        
        # Output: HSV frame and attention mask
        hsv_output = self.sigmoid(self.outc(x))
        attention_logits = self.attention_out(x)
        
        # Apply softmax across spatial dimensions (H, W) to create attention mask
        # Reshape to (batch, H*W) for softmax, then reshape back
        batch_size, _, height, width = attention_logits.shape
        attention_flat = attention_logits.view(batch_size, -1)
        attention_softmax = F.softmax(attention_flat, dim=1)
        attention_mask = attention_softmax.view(batch_size, 1, height, width)
        
        return hsv_output, attention_mask


def create_unet(frame_width=64, frame_height=64, model_size=64, dropout_rate=0.1, weight_decay=1e-4):
    """Create a UNET model with specified frame and model sizes."""
    return UNet(frame_width=frame_width, frame_height=frame_height, model_size=model_size, 
                dropout_rate=dropout_rate, weight_decay=weight_decay)


# Example usage
if __name__ == "__main__":
    batch_size = 2
    # Create model with custom sizes
    custom_model = create_unet(frame_width=96, frame_height=64, model_size=128)
    custom_frames = torch.randn(batch_size, 6, 96, 64)
    custom_output = custom_model(custom_frames)
    print(f"Custom input shape: {custom_frames.shape}")
    print(f"Custom output shape: {custom_output.shape}")
