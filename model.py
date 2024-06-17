import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Define a single convolutional block as CONV->BN->RELU."""

    def __init__(self, in_channels, out_channels):
        """Initialize the ConvBlock."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Run forward pass through the layers."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    """Define the encoder block with skip connections as CONV->CONV->Pooling."""

    def __init__(self, in_channels, out_channels):
        """Initialize the EncoderBlock."""
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """Run forward pass through the layers."""
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        return self.pool(x), skip


class DecoderBlock(nn.Module):
    """Define the decoder block with skip connections as UPCONV->CONV->CONV."""

    def __init__(self, in_channels, out_channels, skip_channels=None):
        """Initialize the DecoderBlock."""
        super().__init__()
        if skip_channels is None:
            skip_channels = out_channels
        self.upconv = nn.ConvTranspose2d(
            in_channels, skip_channels, kernel_size=2, stride=2
        )
        self.conv1 = ConvBlock(skip_channels + out_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip=None):
        """Run forward pass through the layers."""
        x = self.upconv(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetWithSkipv2(nn.Module):
    """Construct the UNet model."""

    def __init__(
        self, input_ch=1, use_act=None, encoder_channels=(32, 64, 128, 256, 512, 1024)
    ):
        """Initialize the model.

        Args:
            input_ch (int, optional): Number of input channels. Defaults to 10.
            use_act (nn.module, optional): Activation function. Defaults to None.
            encoder_channels (tuple, optional): Encoder channels. Defaults to (32, 64, 128, 256, 512, 1024).

        """  # noqa: E501
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.use_act = use_act

        in_ch = input_ch
        for out_ch in encoder_channels[:-1]:
            self.encoders.append(EncoderBlock(in_ch, out_ch))
            in_ch = out_ch

        center_ch = encoder_channels[-1]
        self.center = ConvBlock(in_ch, center_ch)

        rev_enc_channels = list(encoder_channels[::-1])
        for i in range(len(rev_enc_channels) - 1):
            in_ch = rev_enc_channels[i]
            out_ch = rev_enc_channels[i + 1]
            self.decoders.append(DecoderBlock(in_ch, out_ch))

        self.output_conv = nn.Conv2d(encoder_channels[0], 1, kernel_size=1)
        self.act = self.use_act

    def forward(self, x):
        """Run forward pass through the model."""
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.center(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        x = self.output_conv(x)
        if self.use_act:
            x = self.act(x)

        return x


class ConvBlock2D(nn.Module):
    """Define a single convolutional block for 2D UNet as CONV->BN->RELU."""

    def __init__(self, in_channels, out_channels):
        """Initialize the ConvBlock."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Run forward pass through the layers."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock2D(nn.Module):
    """Define the encoder block with skip connections for 2D UNet as CONV->CONV->Pooling."""

    def __init__(self, in_channels, out_channels):
        """Initialize the EncoderBlock."""
        super().__init__()
        self.conv1 = ConvBlock2D(in_channels, out_channels)
        self.conv2 = ConvBlock2D(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """Run forward pass through the layers."""
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        return self.pool(x), skip


class DecoderBlock2D(nn.Module):
    """Define the decoder block with skip connections for 2D UNet as UPCONV->CONV->CONV."""

    def __init__(self, in_channels, out_channels, skip_channels=None):
        """Initialize the DecoderBlock."""
        super().__init__()
        if skip_channels is None:
            skip_channels = out_channels
        self.upconv = nn.ConvTranspose2d(
            in_channels, skip_channels, kernel_size=2, stride=2
        )
        self.conv1 = ConvBlock2D(skip_channels + out_channels, out_channels)
        self.conv2 = ConvBlock2D(out_channels, out_channels)

    def forward(self, x, skip=None):
        """Run forward pass through the layers."""
        x = self.upconv(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet2D(nn.Module):
    """Construct the 2D UNet model."""

    def __init__(
        self, input_ch=1, use_act=None, encoder_channels=(32, 64, 128, 256, 512, 1024)
    ):
        """Initialize the 2D UNet model.

        Args:
            input_ch (int, optional): Number of input channels. Defaults to 1.
            use_act (nn.module, optional): Activation function. Defaults to None.
            encoder_channels (tuple, optional): Encoder channels. Defaults to (32, 64, 128, 256, 512, 1024).

        """  # noqa: E501
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.use_act = use_act

        in_ch = input_ch
        for out_ch in encoder_channels[:-1]:
            self.encoders.append(EncoderBlock2D(in_ch, out_ch))
            in_ch = out_ch

        center_ch = encoder_channels[-1]
        self.center = ConvBlock2D(in_ch, center_ch)

        rev_enc_channels = list(encoder_channels[::-1])
        for i in range(len(rev_enc_channels) - 1):
            in_ch = rev_enc_channels[i]
            out_ch = rev_enc_channels[i + 1]
            self.decoders.append(DecoderBlock2D(in_ch, out_ch))

        self.output_conv = nn.Conv2d(encoder_channels[0], 1, kernel_size=1)
        self.act = self.use_act

    def forward(self, x):
        """Run forward pass through the 2D UNet model."""
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.center(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        x = self.output_conv(x)
        if self.use_act:
            x = self.act(x)

        return x


class ConvBlock3D(nn.Module):
    """Define a single convolutional block as CONV->BN->RELU."""

    def __init__(self, in_channels, out_channels):
        """Initialize the ConvBlock."""
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Run forward pass through the layers."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock3D(nn.Module):
    """Define the encoder block with skip connections as CONV->CONV->Pooling."""

    def __init__(self, in_channels, out_channels):
        """Initialize the EncoderBlock."""
        super().__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels)
        self.conv2 = ConvBlock3D(out_channels, out_channels)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        """Run forward pass through the layers."""
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        return self.pool(x), skip


class DecoderBlock3D(nn.Module):
    """Define the decoder block with skip connections as UPCONV->CONV->CONV."""

    def __init__(self, in_channels, out_channels, skip_channels=None):
        """Initialize the DecoderBlock."""
        super().__init__()
        if skip_channels is None:
            skip_channels = out_channels
        self.upconv = nn.ConvTranspose3d(
            in_channels, skip_channels, kernel_size=2, stride=2
        )
        self.conv1 = ConvBlock3D(skip_channels + out_channels, out_channels)
        self.conv2 = ConvBlock3D(out_channels, out_channels)

    def forward(self, x, skip=None):
        """Run forward pass through the layers."""
        x = self.upconv(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet3D(nn.Module):
    """Construct the 3D UNet model."""

    def __init__(
        self, input_ch=1, use_act=None, encoder_channels=(32, 64, 128, 256, 512, 1024)
    ):
        """Initialize the 3D UNet model.

        Args:
            input_ch (int, optional): Number of input channels. Defaults to 10.
            use_act (nn.module, optional): Activation function. Defaults to None.
            encoder_channels (tuple, optional): Encoder channels. Defaults to (32, 64, 128, 256, 512, 1024).

        """  # noqa: E501
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.use_act = use_act

        in_ch = input_ch
        for out_ch in encoder_channels[:-1]:
            self.encoders.append(EncoderBlock3D(in_ch, out_ch))
            in_ch = out_ch

        center_ch = encoder_channels[-1]
        self.center = ConvBlock3D(in_ch, center_ch)

        rev_enc_channels = list(encoder_channels[::-1])
        for i in range(len(rev_enc_channels) - 1):
            in_ch = rev_enc_channels[i]
            out_ch = rev_enc_channels[i + 1]
            self.decoders.append(DecoderBlock3D(in_ch, out_ch))

        self.output_conv = nn.Conv3d(encoder_channels[0], 1, kernel_size=1)
        self.act = self.use_act

    def forward(self, x):
        """Run forward pass through the 3D UNet model."""
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.center(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        x = self.output_conv(x)
        if self.use_act:
            x = self.act(x)

        return x
