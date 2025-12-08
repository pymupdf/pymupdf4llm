import random

import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, DynamicEdgeConv, GATConv, NNConv
from torch.nn import Sequential, Linear, BatchNorm1d, LeakyReLU, ModuleList
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, ConvTranspose2d
from torchvision.ops import roi_align  # Using RoI Align

from train.core.model.CustomDynamicEdgeConv import CustomDynamicEdgeConv
from train.core.model.ImageFeatrueExtractors import UNetThin


class BasicBlock(nn.Module):
    """
    ResNet BasicBlock without downsampling.
    Keeps H, W resolution unchanged.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class AnisotropicBlock(nn.Module):
    """Enhance horizontal context without losing vertical resolution"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=(1, 5), padding=(0, 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=(1, 5), padding=(0, 2), bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ResNetStyleBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters=32):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        self.block1 = BasicBlock(num_filters)
        self.block2 = DilatedBlock(num_filters, dilation=2)
        self.block3 = DilatedBlock(num_filters, dilation=4)
        self.block4 = AnisotropicBlock(num_filters)  # (1×5) conv for long text
        self.block5 = DilatedBlock(num_filters, dilation=8)

        self.last = nn.Conv2d(num_filters, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.out_channels = out_channels

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        x6 = self.block5(x5)
        x7 = self.last(x6)
        return x7


class Clamp(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


def MLP(channels, batch_norm=True):
    layers = []
    for i in range(1, len(channels)):
        layer = [Linear(channels[i - 1], channels[i]), LeakyReLU()]
        if batch_norm:
            layer.append(BatchNorm1d(channels[i]))
        layers.append(Sequential(*layer))

    # Add the Clamp module at the very end
    # layers.append(Clamp(-10, 10))
    return Sequential(*layers)


def visualize_batch_rois(resized_image_batch, node_rois, img_h, img_w):
    """
    Visualize RoIs on a batch of images. RoIs are assumed to be in pixel coordinates
    (not normalized) after converting from normalized format.

    Args:
        resized_image_batch (torch.Tensor): Image batch in BxCxHxW format, pixel values in [0,1].
        node_rois (torch.Tensor): Tensor of shape (N_total, 5):
                                  [batch_index, x1, y1, x2, y2] in pixel coordinates.
        img_h (int): Image height.
        img_w (int): Image width.

    Returns:
        list: A list of BGR NumPy images with bounding boxes drawn.
    """

    # Extract batch index from RoIs
    roi_batch_indices = node_rois[:, 0]

    B = resized_image_batch.size(0)
    visualized_images = []

    # Iterate through each image in the batch
    for b in range(B):
        # Extract the b-th image (CxHxW)
        single_image = resized_image_batch[b]

        # Filter RoIs that belong to this batch element
        mask = (roi_batch_indices == b)
        single_rois = node_rois[mask]

        # Convert image tensor to NumPy (HWC, uint8)
        image_np = single_image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # Convert RGB → BGR for OpenCV
        if image_np.shape[2] == 3:
            vis_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif image_np.shape[2] == 1:
            vis_image = cv2.cvtColor(image_np.squeeze(2), cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("Unsupported number of channels for visualization.")

        # If the image has no RoIs, just append the converted image
        if single_rois.size(0) == 0:
            print(f"Batch {b}: No RoIs found.")
            visualized_images.append(vis_image)
            continue

        # Extract pixel coordinates: [x1, y1, x2, y2]
        coords_px = single_rois[:, 1:5].cpu().numpy().astype(np.int32)

        # Draw bounding boxes
        color = (0, 255, 0)  # Green
        thickness = 2

        for x1, y1, x2, y2 in coords_px:
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

        visualized_images.append(vis_image)

    return visualized_images


class CustomRoIPostProcessor(nn.Module):
    def __init__(self, concat_mean_max=True, add_uncertainty=False):
        super().__init__()
        self.concat_mean_max = concat_mean_max
        self.add_uncertainty = add_uncertainty

    def forward(self, im_feat_rois: torch.Tensor):
        # im_feat_rois shape: [N_total, C', H_out, W_out]

        N_total, C, H_out, W_out = im_feat_rois.shape

        # 1. Apply Softmax
        # We assume im_feat_rois contains the raw logits/features from RoIAlign.

        # Reshape for channel-wise softmax across spatial locations (H_out * W_out)
        # [N, C, H, W] -> [N, H, W, C] -> [N * H * W, C]
        flat_logits = im_feat_rois.permute(0, 2, 3, 1).reshape(-1, C)

        # Apply Softmax over the channel dimension (dim=1) for numerical stability
        probs_flat = F.softmax(flat_logits, dim=1)

        # Reshape back to [N, C, H, W]
        probs = probs_flat.reshape(N_total, H_out, W_out, C).permute(0, 3, 1, 2)

        # Use probabilities for all subsequent pooling and uncertainty metrics
        patch_used = probs

        # 2. Mean/Max Pooling (Reduction over H_out, W_out)

        # Mean pooling over spatial dimensions: [N, C, H_out, W_out] -> [N, C]
        mean_vec = patch_used.mean(dim=(2, 3))

        if self.concat_mean_max:
            # Max pooling over spatial dimensions
            # Need two max operations: max over W_out, then max over H_out
            max_vec = patch_used.max(dim=-1).values.max(dim=-1).values
            # Concatenate [Mean; Max]
            pooled_features = torch.cat([mean_vec, max_vec], dim=1)  # [N, 2C]
        else:
            pooled_features = mean_vec  # [N, C]

        # 3. Uncertainty Metrics (Entropy and Margin)
        if self.add_uncertainty:
            # Entropy: -sum_c p_c log p_c
            # Compute log(p) safely
            eps = 1e-12
            # Entropy map: [N * H * W]
            ent_map_flat = -(probs_flat * torch.log(probs_flat + eps)).sum(dim=1)

            # Entropy mean: Average entropy over the spatial patch for each RoI
            # Reshape [N * H * W] -> [N, H_out * W_out] and compute mean along dim 1
            entropy_mean = ent_map_flat.reshape(N_total, -1).mean(dim=1, keepdim=True)  # [N, 1]

            # Margin: top1 - top2
            # Get top 2 probabilities along the channel dimension (dim=1)
            top_two_probs, _ = torch.topk(probs_flat, k=2, dim=1)  # [N * H * W, 2]

            # Margins: top1 - top2
            margins_flat = top_two_probs[:, 0] - top_two_probs[:, 1]  # [N * H * W]

            # Margin mean: Average margin over the spatial patch for each RoI
            margin_mean = margins_flat.reshape(N_total, -1).mean(dim=1, keepdim=True)  # [N, 1]

            # Append uncertainty features [entropy_mean, margin_mean]
            uncertainty_features = torch.cat([entropy_mean, margin_mean], dim=1)  # [N, 2]

            # Combine pooled features and uncertainty metrics
            pooled_features = torch.cat([pooled_features, uncertainty_features], dim=1)  # [N, BaseDim + 2]

        return pooled_features


class BoxIMFDGCNN(torch.nn.Module):
    def __init__(self, node_input_dim, node_output_dim, edge_input_dim,
                 rf_input_dim, rf_output_dim, txp_input_dim, txp_output_dim, imf_input_dim, imf_output_dim,
                 num_node_classes, num_edge_classes, hidden_dim=128, dgcn_dim=64, aggr='max', k=10, option={}):

        super(BoxIMFDGCNN, self).__init__()
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.option = option
        self.onnx_flag = 'ExportONNX' in option
        self.nn_type = 'KNN'
        self.nn_value = ''
        self.nn_k = k
        self.use_local_nn = False
        self.feature_types = ['bbox', 'rf', 'text_pattern']
        self.normalization_pos = []
        self.gcn_conv_list = ModuleList()

        # Two-stage training control variable
        self.training_stage = 'full'  # Default: 'full' training (CNN + GNN)
        if 'training_stage' in option:
            self.training_stage = option['training_stage']

        conv_type = 'DGC'
        if 'conv_type' in option:
            conv_type = option['conv_type']

        self.gcv_conv_num = 2
        if 'conv_num' in option:
            self.gcv_conv_num = option['conv_num']

        self.gcn_conv_merge = ''
        if 'conv_merge_type' in option:
            self.gcn_conv_merge = option['conv_merge_type']

        if 'nn_type' in self.option:
            self.nn_type = self.option['nn_type']
        if 'nn_value' in self.option:
            self.nn_value = self.option['nn_value']

        if 'conv_option' in option and 'use_local_nn' in option['conv_option']:
            self.use_local_nn = option['conv_option']['use_local_nn']

        self.training_noise = 0.0
        self.training_noise_prob = 0.0
        if 'training_noise' in option:
            self.training_noise = option['training_noise']
            self.training_noise_prob = option['training_noise_prob']

        if 'feature_types' in option:
            self.feature_types = option['feature_types']

        if 'normalization' in option:
            self.normalization_pos = option['normalization']

        # Image backbone and RoI pooling for end-to-end learning
        self.img_w = 320
        self.img_h = 320

        backbone_output_name = ''
        if 'backbone' in option and 'output_name' in option['backbone']:
            backbone_output_name = option['backbone']['output_name']

        # 1. Custom CNN Backbone
        # self.cnn_backbone = ResNetStyleBackbone(in_channels=1, out_channels=num_node_classes,
        #                                         num_filters=64)
        self.cnn_backbone = UNetThin({
            'in_ch': 1,
            'out_ch': num_node_classes,
            'num_filters': 32,
            'mask_size': {
                'type': 'Fix',
                'size': [self.img_w, self.img_h],
            },
            'use_sep': False,
            'output_name': '',
        })
        self.backbone_output_channels = num_node_classes
        self.roi_post_processor = CustomRoIPostProcessor()

        # RoI Align output size (e.g., 3x3 for finer granularity)
        self.roi_size = 1

        # MLP for edge feature extraction (Edge Box RoI)
        edge_mlp_input_dim = self.backbone_output_channels * 1 * self.roi_size * self.roi_size
        self.edge_roi_mlp = Sequential(
            Linear(edge_mlp_input_dim, imf_input_dim),
            LeakyReLU(inplace=True)
        )

        # Node feature embeddings
        if 'bbox' in self.feature_types:
            if 'BBX' in self.normalization_pos:
                self.node_embed = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(node_input_dim, affine=True),
                    torch.nn.Linear(node_input_dim, node_output_dim),
                    torch.nn.ReLU(inplace=True)
                )
            else:
                self.node_embed = torch.nn.Sequential(
                    torch.nn.Linear(node_input_dim, node_output_dim),
                    torch.nn.ReLU(inplace=True)
                )
            self.page_bbox_embed = None
            if node_input_dim == 16:
                # node_dim = node_input_dim * 2 # Unused local variable
                self.page_bbox_embed = torch.nn.Sequential(
                    # torch.nn.BatchNorm1d(8, affine=True),
                    torch.nn.Linear(8, node_output_dim),
                    torch.nn.ReLU(inplace=True)
                )

        if 'rf' in self.feature_types:
            if 'BRF' in self.normalization_pos:
                if 'DROP-RF' in option:
                    self.rf_feature_embed = torch.nn.Sequential(
                        torch.nn.BatchNorm1d(rf_input_dim, affine=True),
                        torch.nn.Linear(rf_input_dim, rf_output_dim),
                        torch.nn.Dropout1d(p=0.01),
                        torch.nn.ReLU(inplace=True)
                    )
                else:
                    self.rf_feature_embed = torch.nn.Sequential(
                        torch.nn.BatchNorm1d(rf_input_dim, affine=True),
                        torch.nn.Linear(rf_input_dim, rf_output_dim),
                        torch.nn.ReLU(inplace=True)
                    )
            elif 'ARF' in self.normalization_pos:
                self.rf_feature_embed = torch.nn.Sequential(
                    torch.nn.Linear(rf_input_dim, rf_output_dim),
                    torch.nn.BatchNorm1d(rf_output_dim, affine=True),
                    torch.nn.ReLU(inplace=True)
                )
            else:
                self.rf_feature_embed = torch.nn.Sequential(
                    torch.nn.Linear(rf_input_dim, rf_output_dim),
                    torch.nn.ReLU(inplace=True)
                )

        if 'text_pattern' in self.feature_types:
            if 'BTX' in self.normalization_pos:
                self.txp_feature_embed = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(txp_input_dim, affine=True),
                    torch.nn.Linear(txp_input_dim, txp_output_dim),
                    torch.nn.ReLU(inplace=True)
                )
            else:
                self.txp_feature_embed = torch.nn.Sequential(
                    torch.nn.Linear(txp_input_dim, txp_output_dim),
                    torch.nn.ReLU(inplace=True)
                )

        # RoI aligned image feature embedding
        if 'image' in self.feature_types:
            # RoI Align output is C'x3x3 (flattened to C'*9)
            imf_input_dim_roialign = self.backbone_output_channels * 1 * self.roi_size * self.roi_size

            if 'BIM' in self.normalization_pos:
                self.img_feature_embed = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(imf_input_dim_roialign, affine=True),
                    torch.nn.Linear(imf_input_dim_roialign, imf_output_dim),
                    torch.nn.ReLU(inplace=True)
                )
            else:
                self.img_feature_embed = torch.nn.Sequential(
                    torch.nn.Linear(imf_input_dim_roialign, imf_output_dim),
                    torch.nn.ReLU(inplace=True)
                )

        # Calculate final fusion dimension
        fusion_dim = 0
        if 'bbox' in self.feature_types  and self.training_stage == 'full':
            fusion_dim += node_output_dim
        if 'rf' in self.feature_types and self.training_stage == 'full':
            fusion_dim += rf_output_dim
        if 'text_pattern' in self.feature_types and self.training_stage == 'full':
            fusion_dim += txp_output_dim
        if 'image' in self.feature_types:
            fusion_dim += imf_output_dim

        # Fusion layer
        if 'BFS' in self.normalization_pos:
            if 'DROP-FUSE' in option:
                self.fusion_layer = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(fusion_dim, affine=True),
                    torch.nn.Linear(fusion_dim, hidden_dim),
                    torch.nn.Dropout1d(p=0.01),
                    torch.nn.LeakyReLU(inplace=True),
                )
            else:
                self.fusion_layer = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(fusion_dim, affine=True),
                    torch.nn.Linear(fusion_dim, hidden_dim),
                    torch.nn.LeakyReLU(inplace=True),
                )
        elif 'AFS' in self.normalization_pos:
            self.fusion_layer = torch.nn.Sequential(
                torch.nn.Linear(fusion_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim, affine=True),
                torch.nn.LeakyReLU(inplace=True),
            )
        else:
            self.fusion_layer = torch.nn.Sequential(
                torch.nn.Linear(fusion_dim, hidden_dim),
                torch.nn.LeakyReLU(inplace=True),
            )

        # 2. DGNN / GNN Conv Layers
        if conv_type == 'GCN':
            for i in range(self.gcv_conv_num):
                if i == 0:
                    self.gcn_conv_list.append(GCNConv(hidden_dim, dgcn_dim))
                else:
                    self.gcn_conv_list.append(GCNConv(dgcn_dim, dgcn_dim))
            self.dconv_out = torch.nn.Sequential(
                torch.nn.BatchNorm1d(dgcn_dim * self.gcv_conv_num, affine=True),
                torch.nn.Linear(dgcn_dim * self.gcv_conv_num, hidden_dim),
                torch.nn.ReLU(inplace=True),
            )
        elif conv_type == 'GAT':
            num_head = option['conv_option']['num_head']
            for i in range(self.gcv_conv_num):
                if i == 0:
                    self.gcn_conv_list.append(GATConv(hidden_dim, dgcn_dim, heads=num_head))
                else:
                    self.gcn_conv_list.append(GATConv(dgcn_dim * num_head, dgcn_dim, heads=num_head))

            if self.gcn_conv_merge == 'DEEP':
                self.dconv_out = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(dgcn_dim * num_head, affine=True),
                    torch.nn.Linear(dgcn_dim * num_head, hidden_dim),
                    torch.nn.ReLU(inplace=True),
                )
            else:
                self.dconv_out = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(dgcn_dim * num_head * len(self.gcn_conv_list), affine=True),
                    torch.nn.Linear(dgcn_dim * num_head * len(self.gcn_conv_list), hidden_dim),
                    torch.nn.ReLU(inplace=True),
                )
        elif conv_type == 'NNConv':
            # Edge input dimension is updated above if 'image' is in feature_types
            self.edge_nn_list = torch.nn.ModuleList()

            for i in range(self.gcv_conv_num):
                if i == 0:
                    in_dim = hidden_dim
                    out_dim = dgcn_dim
                else:
                    in_dim = dgcn_dim
                    out_dim = dgcn_dim

                edge_nn = nn.Sequential(
                    nn.Linear(edge_input_dim, dgcn_dim),
                    nn.ReLU(),
                    nn.Linear(dgcn_dim, in_dim * out_dim)
                )

                self.edge_nn_list.append(edge_nn)
                self.gcn_conv_list.append(NNConv(in_dim, out_dim, edge_nn, aggr=aggr))

            if self.gcn_conv_merge == 'DEEP':
                self.dconv_out = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(dgcn_dim * 2, affine=True),
                    torch.nn.Linear(dgcn_dim * 2, hidden_dim),
                    torch.nn.ReLU(inplace=True),
                )
            else:
                self.dconv_out = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(dgcn_dim * self.gcv_conv_num, affine=True),
                    torch.nn.Linear(dgcn_dim * self.gcv_conv_num, hidden_dim),
                    torch.nn.ReLU(inplace=True),
                )
        elif conv_type == 'CustomDGC':
            for i in range(self.gcv_conv_num):
                if i == 0:
                    self.gcn_conv_list.append(
                        CustomDynamicEdgeConv(MLP([2 * hidden_dim, dgcn_dim, dgcn_dim, dgcn_dim]), self.nn_k, aggr,
                                              nn_type=self.nn_type, onnx_export_flag=self.onnx_flag),
                    )
                else:
                    self.gcn_conv_list.append(
                        CustomDynamicEdgeConv(MLP([dgcn_dim * 2, dgcn_dim]), self.nn_k, aggr,
                                              nn_type=self.nn_type, onnx_export_flag=self.onnx_flag)
                    )

            if self.gcn_conv_merge == 'DEEP':
                self.dconv_out = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(dgcn_dim * 2, affine=True),
                    torch.nn.Linear(dgcn_dim * 2, hidden_dim),
                    torch.nn.ReLU(inplace=True),
                )
            else:
                self.dconv_out = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(dgcn_dim * self.gcv_conv_num, affine=True),
                    torch.nn.Linear(dgcn_dim * self.gcv_conv_num, hidden_dim),
                    torch.nn.ReLU(inplace=True),
                )
        else:
            conv_out = dgcn_dim * 2
            self.conv1 = DynamicEdgeConv(MLP([2 * hidden_dim, dgcn_dim, dgcn_dim, dgcn_dim]), self.nn_k, aggr)
            self.conv2 = DynamicEdgeConv(MLP([conv_out, conv_out]), self.nn_k, aggr)
            self.dconv_out = torch.nn.Sequential(
                torch.nn.BatchNorm1d(dgcn_dim * 3, affine=True),
                torch.nn.Linear(dgcn_dim * 3, hidden_dim),
                torch.nn.ReLU(inplace=True),
            )

        # 3. Node classification MLP
        if 'drop_ngc' in option:
            self.node_classifier = torch.nn.Sequential(
                torch.nn.BatchNorm1d(hidden_dim, affine=True),
                torch.nn.Dropout1d(p=option['drop_ngc']),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(hidden_dim, num_node_classes),
                torch.nn.ReLU(inplace=True),
            )
        else:
            self.node_classifier = torch.nn.Sequential(
                torch.nn.BatchNorm1d(hidden_dim, affine=True),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(hidden_dim, num_node_classes),
                torch.nn.ReLU(inplace=True),
            )

        # 4. Edge classification MLP (Node embedding * 2 + Edge feature dimension)
        # Note: The edge classifier input dimension needs to be correct for both pretrain and full.
        # edge_input_dim is the initial edge feature dim. imf_input_dim is the image feature dim extracted by edge_roi_mlp.
        edge_classifier_input_dim = (hidden_dim * 2) + (edge_input_dim + imf_input_dim)

        if 'drop_egc' in option:
            self.edge_classifier = torch.nn.Sequential(
                torch.nn.BatchNorm1d(edge_classifier_input_dim, affine=True),
                torch.nn.Dropout1d(p=option['drop_egc']),
                torch.nn.Linear(edge_classifier_input_dim, hidden_dim),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(hidden_dim, num_edge_classes),
            )
        else:
            self.edge_classifier = torch.nn.Sequential(
                torch.nn.BatchNorm1d(edge_classifier_input_dim, affine=True),
                torch.nn.Linear(edge_classifier_input_dim, hidden_dim),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(hidden_dim, num_edge_classes),
            )

    def forward_with_batch(self, data):
        # image_data replaces static image_features
        bboxes, edge_index, edge_attr, rf_feature, text_patterns, image_data = \
            (data.x, data.edge_index, data.edge_attr, data.rf_features, data.text_patterns,
             data.image_data)

        k = min(len(bboxes), self.nn_k)
        # Pass image_data instead of image_features
        return self.forward(bboxes, edge_index, edge_attr, rf_feature, text_patterns, image_data, k, data.batch)

    def forward(self, bboxes, edge_index, edge_attr, rf_features, text_patterns, image_data, k, batch):
        fusion_features = []

        # Absolute minimum normalized size (width or height) to ensure for any RoI.
        # This value must be between 0.0 and 1.0 (normalized coordinates).
        node_min_size = 0.0
        edge_expansion_ratio = 0.0

        # --- 1. Image Feature Extraction (Image Backbone + RoI Align) ---
        if 'image' in self.feature_types:
            img_w = self.img_w
            img_h = self.img_h

            # Resize image: B x C x H x W
            resized_image = nn.functional.interpolate(
                image_data,
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False
            )

            # Pass through backbone → feature_map (same resolution for segmentation backbone)
            feature_map = self.cnn_backbone(resized_image)

            # RoI Align Setup
            num_nodes = bboxes.size(0)
            if batch is None:
                roi_batch_idx = torch.zeros(num_nodes, dtype=torch.float32, device=bboxes.device).unsqueeze(1)
            else:
                roi_batch_idx = batch.to(torch.float32).unsqueeze(1)

            # Extract first 4 normalized coordinates (x1,y1,x2,y2)
            node_coords_norm = bboxes[:, :4]

            # --- Ensure minimum RoI size in normalized coordinate space ---
            w = node_coords_norm[:, 2] - node_coords_norm[:, 0]
            h = node_coords_norm[:, 3] - node_coords_norm[:, 1]

            pad_w = torch.max(torch.zeros_like(w), (node_min_size - w) / 2.0)
            pad_h = torch.max(torch.zeros_like(h), (node_min_size - h) / 2.0)

            # Expand in normalized coordinate range and clamp to [0,1]
            expanded_x1 = torch.max(torch.zeros_like(node_coords_norm[:, 0]), node_coords_norm[:, 0] - pad_w)
            expanded_y1 = torch.max(torch.zeros_like(node_coords_norm[:, 1]), node_coords_norm[:, 1] - pad_h)
            expanded_x2 = torch.min(torch.ones_like(node_coords_norm[:, 2]), node_coords_norm[:, 2] + pad_w)
            expanded_y2 = torch.min(torch.ones_like(node_coords_norm[:, 3]), node_coords_norm[:, 3] + pad_h)

            node_coords_norm_expanded = torch.stack([expanded_x1, expanded_y1, expanded_x2, expanded_y2], dim=1)

            # --- Convert normalized coordinates → pixel coordinates ---
            # RoIAlign expects pixel-space coordinates, not normalized.
            node_coords_px = node_coords_norm_expanded.clone()
            node_coords_px[:, 0] *= img_w
            node_coords_px[:, 1] *= img_h
            node_coords_px[:, 2] *= img_w
            node_coords_px[:, 3] *= img_h

            # Construct RoIAlign input [batch_idx, x1, y1, x2, y2]
            node_rois = torch.cat([roi_batch_idx, node_coords_px], dim=1).to(torch.float32)

            # Feature map resolution = image resolution for segmentation backbone
            fm_h, fm_w = feature_map.shape[-2:]
            spatial_scale = fm_h / img_h

            # Optional visualization
            if '' == 'D':
                vis_imgs = visualize_batch_rois(resized_image, node_rois, img_h, img_w)
                for img_idx, vis_img in enumerate(vis_imgs):
                    cv2.imwrite(f'temp/{img_idx}.jpg', vis_img)
                exit(0)

            # Run RoIAlign (Node)
            im_feat_rois = roi_align(
                input=feature_map,
                boxes=node_rois,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=spatial_scale,
                sampling_ratio=1
            )

            # Post process and flatten
            # im_feat_rois = self.roi_post_processor(im_feat_rois)
            im_feat_rois = im_feat_rois.view(im_feat_rois.size(0), -1)

            # MLP embedding
            im_feat = self.img_feature_embed(im_feat_rois)
            # im_feat = im_feat.detach()
            fusion_features.append(im_feat)

            # -------------------------------------------------------------------------
            # 3. RoI Align for Edges (union bboxes)
            # -------------------------------------------------------------------------
            row, col = edge_index

            # Extract normalized boxes
            bbox_i = bboxes[row][:, :4]
            bbox_j = bboxes[col][:, :4]

            # Union in normalized coordinate space
            edge_x1 = torch.min(bbox_i[:, 0], bbox_j[:, 0])
            edge_y1 = torch.min(bbox_i[:, 1], bbox_j[:, 1])
            edge_x2 = torch.max(bbox_i[:, 2], bbox_j[:, 2])
            edge_y2 = torch.max(bbox_i[:, 3], bbox_j[:, 3])

            edge_coords_norm = torch.stack([edge_x1, edge_y1, edge_x2, edge_y2], dim=1)

            # Expand edges in normalized coordinate space
            w_edge = edge_coords_norm[:, 2] - edge_coords_norm[:, 0]
            h_edge = edge_coords_norm[:, 3] - edge_coords_norm[:, 1]

            pad_w_edge = w_edge * (edge_expansion_ratio / 2)
            pad_h_edge = h_edge * (edge_expansion_ratio / 2)

            expanded_edge_x1 = torch.max(torch.zeros_like(edge_coords_norm[:, 0]),
                                         edge_coords_norm[:, 0] - pad_w_edge)
            expanded_edge_y1 = torch.max(torch.zeros_like(edge_coords_norm[:, 1]),
                                         edge_coords_norm[:, 1] - pad_h_edge)
            expanded_edge_x2 = torch.min(torch.ones_like(edge_coords_norm[:, 2]),
                                         edge_coords_norm[:, 2] + pad_w_edge)
            expanded_edge_y2 = torch.min(torch.ones_like(edge_coords_norm[:, 3]),
                                         edge_coords_norm[:, 3] + pad_h_edge)

            edge_coords_norm_exp = torch.stack(
                [expanded_edge_x1, expanded_edge_y1, expanded_edge_x2, expanded_edge_y2], dim=1
            )

            # --- Convert normalized edge union → pixel coordinates ---
            edge_coords_px = edge_coords_norm_exp.clone()
            edge_coords_px[:, 0] *= img_w
            edge_coords_px[:, 1] *= img_h
            edge_coords_px[:, 2] *= img_w
            edge_coords_px[:, 3] *= img_h

            # Prepare batch indices
            edge_batch_idx = roi_batch_idx[row].to(torch.float32)

            # Construct final RoI input
            edge_rois = torch.cat([edge_batch_idx, edge_coords_px], dim=1).to(torch.float32)

            # Run RoIAlign (Edge)
            edge_im_feat_rois = roi_align(
                input=feature_map,
                boxes=edge_rois,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=spatial_scale,
                sampling_ratio=-1
            )

            # Post process
            # edge_im_feat_rois = self.roi_post_processor(edge_im_feat_rois)
            edge_im_feat_rois = edge_im_feat_rois.view(edge_im_feat_rois.size(0), -1)

            # MLP convert and append to edge_attr
            edge_im_feat = self.edge_roi_mlp(edge_im_feat_rois)
            # edge_im_feat = edge_im_feat.detach()
            edge_attr = torch.cat([edge_attr, edge_im_feat], dim=-1)


        if 'bbox' in self.feature_types and self.training_stage == 'full':
            # bbox features are 8D, node_embed takes all 8 dimensions
            box_feat = self.node_embed(bboxes)
            fusion_features.append(box_feat)

        if 'rf' in self.feature_types and self.training_stage == 'full':
            rf_feat = self.rf_feature_embed(rf_features)
            fusion_features.append(rf_feat)

        if 'text_pattern' in self.feature_types and self.training_stage == 'full':
            txp_feat = self.txp_feature_embed(text_patterns)
            fusion_features.append(txp_feat)

        # All features (including image feature) are concatenated and passed through the fusion layer
        fusion_feat = self.fusion_layer(torch.cat(fusion_features, dim=1))

        if self.training and self.training_noise > 0 and self.training_noise_prob > random.random():
            sigma = np.random.uniform(0, self.training_noise)
            noise = torch.randn_like(fusion_feat)
            scaled_noise = noise * sigma
            fusion_feat = fusion_feat + scaled_noise

        # GNN processing step
        if batch is None:
            batch = torch.zeros(bboxes.size(0), dtype=torch.long, device=bboxes.device)

        # --- Two-Stage Training Logic ---
        if self.training_stage == 'pretrain':
            # Pretrain stage: Skip GNN. Use fusion_feat directly as gcn_feat
            gcn_feat = fusion_feat
        else:  # self.training_stage == 'full'
            conv_type = 'DGC'
            if 'conv_type' in self.option:
                conv_type = self.option['conv_type']

            if conv_type == 'DGC':
                gcn_feat_1 = self.conv1(fusion_feat, batch)
                gcn_feat_2 = self.conv2(gcn_feat_1, batch)
                gcn_feat = self.dconv_out(torch.cat([gcn_feat_1, gcn_feat_2], dim=1))  # box_len*512

            elif conv_type == 'CustomDGC':
                nn_bboxes = None
                if self.nn_value == 'BBOX':
                    nn_bboxes = bboxes
                # Assuming box_feat is calculated above if 'bbox' in feature_types
                elif self.nn_value == 'BBOX-FEAT':
                    if 'bbox' in self.feature_types:
                        nn_bboxes = box_feat
                    else:
                        # Handle case where BBOX-FEAT is requested but bbox feature is not used for node embedding
                        nn_bboxes = None

                feat_list = []
                x = fusion_feat
                for gcn_idx, gcn_conv in enumerate(self.gcn_conv_list):
                    if self.use_local_nn:
                        x = gcn_conv(x, k, batch, bboxes=nn_bboxes, nn_index=edge_index)
                    else:
                        x = gcn_conv(x, k, batch, bboxes=nn_bboxes)
                    feat_list.append(x)

                if self.gcn_conv_merge == 'DEEP':
                    gcn_feat = self.dconv_out(x)  # box_len*512
                else:
                    gcn_feat = self.dconv_out(torch.cat(feat_list, dim=1))  # box_len*512

            elif conv_type == 'NNConv':
                feat_list = []
                x = fusion_feat
                for gcn_idx, gcn_conv in enumerate(self.gcn_conv_list):
                    # NNConv uses edge_attr, which now includes image features
                    if self.use_local_nn:
                        x = gcn_conv(x, edge_index, edge_attr)
                    else:
                        x = gcn_conv(x, edge_index, edge_attr)
                    feat_list.append(x)

                if self.gcn_conv_merge == 'DEEP':
                    gcn_feat = self.dconv_out(x)  # box_len*512
                else:
                    gcn_feat = self.dconv_out(torch.cat(feat_list, dim=1))  # box_len*512
            else:  # GCN/GAT
                feat_list = []
                x = fusion_feat

                for gcn_conv in self.gcn_conv_list:
                    if self.use_local_nn:
                        local_edge_index = edge_index
                    else:
                        local_edge_index = edge_index

                    # GCNConv or GATConv call
                    x = gcn_conv(x, local_edge_index)

                    feat_list.append(x)

                if self.gcn_conv_merge == 'DEEP':
                    gcn_feat = self.dconv_out(x)  # box_len*512
                else:
                    gcn_feat = self.dconv_out(torch.cat(feat_list, dim=1))

        # Combine initial feature and GNN feature (residual-like connection)
        # If pretrain: node_feat = max(fusion_feat, fusion_feat) = fusion_feat
        node_feat = torch.maximum(fusion_feat, gcn_feat)
        node_logits = self.node_classifier(node_feat)

        # Edge classification
        row, col = edge_index

        # Concatenate node features and edge attributes (which now includes RoI features)
        edge_embedding_and_attr = torch.cat([node_feat[row], node_feat[col], edge_attr], dim=-1)
        edge_logits = self.edge_classifier(edge_embedding_and_attr)

        return node_logits, edge_logits
