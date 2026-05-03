from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class AttentionBlock(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.att1 = AttentionBlock(256, 512, 128)
        self.up2 = Up(512, 128)
        self.att2 = AttentionBlock(128, 256, 64)
        self.up3 = Up(256, 64)
        self.att3 = AttentionBlock(64, 128, 32)
        self.up4 = Up(128, 64)
        self.att4 = AttentionBlock(64, 64, 32)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4a = self.att1(x5, x4)
        x = self.up1(x5, x4a)
        x3a = self.att2(x, x3)
        x = self.up2(x, x3a)
        x2a = self.att3(x, x2)
        x = self.up3(x, x2a)
        x1a = self.att4(x, x1)
        x = self.up4(x, x1a)

        return self.outc(x)


def build_segmentation_model(name: str, in_channels: int = 1, out_channels: int = 1) -> nn.Module:
    key = name.lower()

    if key == "unet":
        return UNet(in_channels, out_channels)
    if key in {"attention_unet", "attention-u-net"}:
        return AttentionUNet(in_channels, out_channels)
    if key in {"unet++", "unetpp", "unet_plus_plus"}:
        try:
            import segmentation_models_pytorch as smp
        except Exception as exc:
            raise RuntimeError("Install segmentation_models_pytorch for U-Net++ support.") from exc
        return smp.UnetPlusPlus(encoder_name="resnet34", in_channels=in_channels, classes=out_channels)
    if key in {"monai_unet", "unetr", "swinunetr"}:
        try:
            from monai.networks.nets import UNETR, SwinUNETR, UNet
        except Exception as exc:
            raise RuntimeError("Install MONAI for advanced segmentation models.") from exc

        if key == "monai_unet":
            return UNet(spatial_dims=2, in_channels=in_channels, out_channels=out_channels, channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=2)
        if key == "unetr":
            return UNETR(in_channels=in_channels, out_channels=out_channels, img_size=(224, 224), feature_size=16, spatial_dims=2)
        return SwinUNETR(img_size=(224, 224), in_channels=in_channels, out_channels=out_channels, feature_size=24, spatial_dims=2)

    raise ValueError(f"Unknown segmentation model: {name}")
