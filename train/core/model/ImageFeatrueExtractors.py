import torch
import torch.nn as nn
import torch.nn.functional as F


# Depthwise separable conv block: depthwise -> pointwise, with BN+ReLU
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=False):
        super().__init__()
        # depthwise conv (groups=in_ch)
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                   padding=padding, groups=in_ch, bias=bias)
        # pointwise conv
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_sep=False):
        super().__init__()
        if use_sep:
            self.conv1 = DepthwiseSeparableConv(in_ch, out_ch)
            self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_ch)
            self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        # Check if the block is a standard convolution block
        if hasattr(self, 'activation'):
            x = self.activation(x)
            x = self.conv2(x)
            x = self.bn(x)
            x = self.activation(x)
        else:  # use_sep==True case (DSConvBlock)
            x = self.conv2(x)
        return x


class UNetThin(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.in_ch = cfg['in_ch']
        self.out_ch = cfg['out_ch']
        self.num_filters = cfg['num_filters']
        self.out_size = cfg['mask_size']

        self.use_sep = cfg['use_sep']
        self.norm_type = 'BN'
        self.output_name = cfg['output_name']

        base_filters = self.num_filters

        # Encoder: Standard UNet encoder path with DSConvBlock
        # the first encoder doesn't use SeparableConv
        self.enc1 = DSConvBlock(self.in_ch, base_filters, use_sep=False)
        self.enc2 = DSConvBlock(base_filters, base_filters, use_sep=self.use_sep)
        self.enc3 = DSConvBlock(base_filters, base_filters, use_sep=self.use_sep)
        self.enc4 = DSConvBlock(base_filters, base_filters, use_sep=self.use_sep)
        self.enc5 = DSConvBlock(base_filters, base_filters, use_sep=self.use_sep)
        self.enc6 = DSConvBlock(base_filters, base_filters, use_sep=self.use_sep)

        # Max pooling layers
        self.pool = nn.AvgPool2d(2)
        # self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DSConvBlock(base_filters, base_filters)
        self.bottleneck_norm = self._make_norm(base_filters)

        # Decoder blocks
        # No more up_proj layers; we'll handle upsampling and fusion inside forward
        self.dec6 = DSConvBlock(base_filters * 2, base_filters, use_sep=self.use_sep)
        self.dec5 = DSConvBlock(base_filters * 2, base_filters, use_sep=self.use_sep)
        self.dec4 = DSConvBlock(base_filters * 2, base_filters, use_sep=self.use_sep)
        self.dec3 = DSConvBlock(base_filters * 2, base_filters, use_sep=self.use_sep)
        self.dec2 = DSConvBlock(base_filters * 2, base_filters, use_sep=self.use_sep)

        # Segmentation head
        self.seg_norm = self._make_norm(base_filters)
        self.seg_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1, bias=False),
            self.seg_norm,
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, self.out_ch, kernel_size=1)
        )

        # Final block for additional feature learning after upsampling
        self.final_upsample_conv = DSConvBlock(self.out_ch, self.out_ch, use_sep=False)

    def _make_norm(self, channels):
        if self.norm_type == 'group':
            groups = min(32, channels)
            while channels % groups != 0:
                groups -= 1
                if groups == 0:
                    groups = 1
                    break
            return nn.GroupNorm(max(1, groups), channels)
        else:
            return nn.BatchNorm2d(channels)

    def forward(self, x):
        # Encoder stages
        e1 = self.enc1(x)  # H, W
        p1 = self.pool(e1)  # H/2, W/2

        e2 = self.enc2(p1)  # H/2, W/2
        p2 = self.pool(e2)  # H/4, W/4

        e3 = self.enc3(p2)  # H/4, W/4
        p3 = self.pool(e3)  # H/8, W/8

        e4 = self.enc4(p3)  # H/8, W/8
        p4 = self.pool(e4)  # H/16, W/16

        e5 = self.enc5(p4)  # H/16, W/16
        p5 = self.pool(e5)  # H/32, W/32

        e6 = self.enc6(p5)  # H/32, W/32
        p6 = self.pool(e6)  # H/64, W/64

        # Bottleneck
        b = self.bottleneck(p6)
        b = self.bottleneck_norm(b)

        if self.output_name == 'bottle_neck':
            return b

        # Decoder stages with additive fusion

        # Decoder 6: Up-sample from bottleneck and add to e6 (H/32, W/32)
        d6_upsample = F.interpolate(b, size=e6.shape[2:], mode='bilinear', align_corners=False)
        d6 = self.dec6(torch.cat([d6_upsample, e6], dim=1))

        # Decoder 5: Up-sample from d6 and add to e5 (H/16, W/16)
        d5_upsample = F.interpolate(d6, size=e5.shape[2:], mode='bilinear', align_corners=False)
        d5 = self.dec5(torch.cat([d5_upsample, e5], dim=1))

        # Decoder 4: Up-sample from d5 and add to e4 (H/8, W/8)
        d4_upsample = F.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4_upsample, e4], dim=1))

        # Decoder 3: Up-sample from d4 and add to e3 (H/4, W/4)
        d3_upsample = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3_upsample, e3], dim=1))

        # Decoder 2: Up-sample from d3 and add to e2 (H/2, W/2)
        d2_upsample = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2_upsample, e2], dim=1))

        if self.output_name == 'dec_all':
            # Determine the target size based on logits
            target_size = x.shape[2:]

            # Upsample all decoder outputs to the target size
            d2_resized = F.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)
            d3_resized = F.interpolate(d3, size=target_size, mode='bilinear', align_corners=False)
            d4_resized = F.interpolate(d4, size=target_size, mode='bilinear', align_corners=False)
            d5_resized = F.interpolate(d5, size=target_size, mode='bilinear', align_corners=False)
            d6_resized = F.interpolate(d6, size=target_size, mode='bilinear', align_corners=False)

            # Concatenate all resized tensors along the channel dimension (dim=1)
            # Note: d1 is not in the original code, so we concatenate from d2 to d6.
            combined_features = torch.cat([d2_resized, d3_resized, d4_resized, d5_resized, d6_resized], dim=1)
            return combined_features

        if self.output_name == 'head_in':
            return d2

        # Segmentation head
        seg_in = self.seg_norm(d2)
        logits = self.seg_head(seg_in)

        # Final upsampling and convolution if out_size is specified
        if self.out_size is None or self.out_size['type'] == 'Original':
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            out_size = self.out_size['size']
            logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)
        logits = self.final_upsample_conv(logits)

        if self.output_name == 'all':
            # Determine the target size based on logits
            target_size = x.shape[2:]

            # Upsample all decoder outputs to the target size
            d2_resized = F.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)
            d3_resized = F.interpolate(d3, size=target_size, mode='bilinear', align_corners=False)
            d4_resized = F.interpolate(d4, size=target_size, mode='bilinear', align_corners=False)
            d5_resized = F.interpolate(d5, size=target_size, mode='bilinear', align_corners=False)
            d6_resized = F.interpolate(d6, size=target_size, mode='bilinear', align_corners=False)

            logits = F.softmax(logits)
            combined_features = torch.cat([d2_resized, d3_resized, d4_resized, d5_resized, d6_resized, logits], dim=1)
            return combined_features

        return logits

class UNetThin2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.in_ch = cfg['in_ch']
        self.out_ch = cfg['out_ch']
        self.num_filters = cfg['num_filters']
        self.out_size = cfg['mask_size']
        self.use_sep = cfg['use_sep']
        self.norm_type = 'BN'
        self.output_name = cfg['output_name']

        base_filters = self.num_filters

        # Encoder blocks
        self.enc1 = DSConvBlock(self.in_ch, base_filters, use_sep=self.use_sep)   # keep input size
        self.enc2 = DSConvBlock(base_filters, base_filters, use_sep=self.use_sep)
        self.enc3 = DSConvBlock(base_filters, base_filters, use_sep=self.use_sep)
        self.enc4 = DSConvBlock(base_filters, base_filters, use_sep=self.use_sep)
        self.enc5 = DSConvBlock(base_filters, base_filters, use_sep=self.use_sep)
        self.enc6 = DSConvBlock(base_filters, base_filters, use_sep=self.use_sep)

        # GAP layers for fixed output sizes
        self.gap2 = nn.AdaptiveAvgPool2d((50, 50))
        self.gap3 = nn.AdaptiveAvgPool2d((25, 25))
        self.gap4 = nn.AdaptiveAvgPool2d((10, 10))
        self.gap5 = nn.AdaptiveAvgPool2d((5, 5))
        self.gap6 = nn.AdaptiveAvgPool2d((3, 3))

        # Fusion layer after concatenation (reduce channels back to base_filters)
        self.fusion_conv = nn.Conv2d(base_filters, base_filters, kernel_size=1)

        # Segmentation head
        self.seg_norm = self._make_norm(base_filters)
        self.seg_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1, bias=False),
            self.seg_norm,
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, self.out_ch, kernel_size=1)
        )

    def _make_norm(self, channels):
        if self.norm_type == 'group':
            groups = min(32, channels)
            while channels % groups != 0:
                groups -= 1
                if groups == 0:
                    groups = 1
                    break
            return nn.GroupNorm(max(1, groups), channels)
        else:
            return nn.BatchNorm2d(channels)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # Encoder1: keep input size
        e1 = self.enc1(x)   # (B, C, H, W)

        # Encoder2: GAP(50x50)
        e2 = self.gap2(e1)
        e2 = self.enc2(e2)

        # Encoder3: GAP(25x25)
        e3 = self.gap3(e2)
        e3 = self.enc3(e3)

        # Encoder4: GAP(10x10)
        e4 = self.gap4(e3)
        e4 = self.enc4(e4)

        # Encoder5: GAP(5x5)
        e5 = self.gap5(e4)
        e5 = self.enc5(e5)

        # Encoder6: GAP(3x3)
        e6 = self.enc6(e5)
        e6 = self.gap6(e6)

        # Upsample all encoder outputs to input size
        e2_up = F.interpolate(e2, size=(H, W), mode='bilinear', align_corners=False)
        e3_up = F.interpolate(e3, size=(H, W), mode='bilinear', align_corners=False)
        e4_up = F.interpolate(e4, size=(H, W), mode='bilinear', align_corners=False)
        e5_up = F.interpolate(e5, size=(H, W), mode='bilinear', align_corners=False)
        e6_up = F.interpolate(e6, size=(H, W), mode='bilinear', align_corners=False)

        if self.output_name == 'fusion-concat':
            return torch.concat([e1, e2_up, e3_up, e4_up, e5_up, e6_up], dim=1)

        # Concatenate instead of sum
        # fused = torch.cat([e1, e2_up, e3_up, e4_up, e5_up, e6_up], dim=1)  # (B, C*6, H, W)
        fused = e1 + e2_up + e3_up + e4_up + e5_up + e6_up
        if self.output_name == 'fusion':
            return fused

        # Reduce channels back to base_filters
        fused = self.fusion_conv(fused)
        seg_in = self.seg_norm(fused)
        if self.output_name == 'seg_in':
            return seg_in

        # Segmentation head
        logits = self.seg_head(seg_in)
        if self.output_name == 'all':
            return torch.concat([e1, e2_up, e3_up, e4_up, e5_up, e6_up, logits], dim=1)
        return logits


class SimpleDiscriminator(nn.Module):
    """
    Discriminator network for GAN-style training.
    It takes an image and a mask (either ground-truth or generated)
    and predicts if the mask is 'real' or 'fake'.
    """
    def __init__(self, in_channels, num_filters=64):
        super().__init__()
        # Discriminator takes an image (1 channel) and a mask (num_classes channels) as input
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, 3, 1, 1),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 4, 3, 1, 1),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Modified: The final layer should be a convolutional layer to handle
        # different input sizes and output a single value.
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters * 4, 1, 3, 1, 1)
        )

    def forward(self, image, mask):
        # Concatenate image and mask along the channel dimension
        x = torch.cat([image, mask], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_conv(x)
        # Squeeze to get a single value per batch element
        return x.view(x.size(0), -1)


class VAEUNetMultiScale(nn.Module):
    def __init__(self, num_classes, num_filters=32, out_size=(100, 100),
                 use_sep=False, output_name='out', norm_type='batch', input_channels=1):
        super().__init__()
        self.out_size = out_size
        self.num_filters = num_filters
        self.norm_type = norm_type
        self.output_name = output_name
        self.input_channels = input_channels

        base_filters = num_filters

        # Encoder
        self.enc1 = DSConvBlock(self.input_channels, base_filters, use_sep=use_sep)
        self.enc2 = DSConvBlock(base_filters, base_filters, use_sep=use_sep)
        self.enc3 = DSConvBlock(base_filters, base_filters, use_sep=use_sep)
        self.enc4 = DSConvBlock(base_filters, base_filters, use_sep=use_sep)
        self.enc5 = DSConvBlock(base_filters, base_filters, use_sep=use_sep)
        self.enc6 = DSConvBlock(base_filters, base_filters, use_sep=use_sep)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DSConvBlock(base_filters, base_filters)
        self.fc_mu = nn.Conv2d(base_filters, base_filters, kernel_size=1)
        self.fc_logvar = nn.Conv2d(base_filters, base_filters, kernel_size=1)

        # Decoder:
        self.dec6 = DSConvBlock(base_filters * 2, base_filters, use_sep=use_sep)
        self.dec5 = DSConvBlock(base_filters * 2, base_filters, use_sep=use_sep)
        self.dec4 = DSConvBlock(base_filters * 2, base_filters, use_sep=use_sep)
        self.dec3 = DSConvBlock(base_filters * 2, base_filters, use_sep=use_sep)
        self.dec2 = DSConvBlock(base_filters * 2, base_filters, use_sep=use_sep)

        # Segmentation head
        self.seg_norm = self._make_norm(base_filters)
        self.seg_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1, bias=False),
            self.seg_norm,
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, num_classes, kernel_size=1)
        )

        if self.out_size is not None:
            self.final_upsample_conv = DSConvBlock(num_classes, num_classes, use_sep=False)

    def _make_norm(self, channels):
        if self.norm_type == 'group':
            groups = min(32, channels)
            while channels % groups != 0:
                groups -= 1
                if groups == 0:
                    groups = 1
                    break
            return nn.GroupNorm(max(1, groups), channels)
        else:
            return nn.BatchNorm2d(channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder stages
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        # e4 = self.enc4(p3)
        # p4 = self.pool(e4)
        #
        # e5 = self.enc5(p4)
        # p5 = self.pool(e5)
        #
        # e6 = self.enc6(p5)
        # p6 = self.pool(e6)

        # VAE Bottleneck:
        b = self.bottleneck(p3)
        mu = self.fc_mu(b)

        if self.output_name == 'vae-mu':
            return mu

        logvar = self.fc_logvar(b)

        z = self.reparameterize(mu, logvar)

        # Decoder stages
        # d6_upsample = F.interpolate(z, size=e6.shape[2:], mode='bilinear', align_corners=False)
        # d6 = self.dec6(torch.cat([d6_upsample, e6], dim=1))
        #
        # d5_upsample = F.interpolate(d6, size=e5.shape[2:], mode='bilinear', align_corners=False)
        # d5 = self.dec5(torch.cat([d5_upsample, e5], dim=1))
        #
        # d4_upsample = F.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        # d4 = self.dec4(torch.cat([d4_upsample, e4], dim=1))

        d3_upsample = F.interpolate(z, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3_upsample, e3], dim=1))

        d2_upsample = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2_upsample, e2], dim=1))

        # Segmentation head
        seg_in = self.seg_norm(d2)
        logits = self.seg_head(seg_in)

        # Final upsampling
        if self.out_size is not None:
            logits = F.interpolate(logits, size=self.out_size, mode='bilinear', align_corners=False)
            logits = self.final_upsample_conv(logits)

        return logits, mu, logvar