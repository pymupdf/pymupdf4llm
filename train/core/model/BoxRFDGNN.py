# encoding: utf-8
'''
@software: Pycharm
@time: 2021/11/24 6:27 下午
@desc:
'''
import random

import torch
import torch.nn as nn

import numpy as np

from torch_geometric.nn import GCNConv, DynamicEdgeConv, GATConv, NNConv
from torch.nn import Sequential, Linear, BatchNorm1d, LeakyReLU, ModuleList

from train.core.model.CustomDynamicEdgeConv import CustomDynamicEdgeConv

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
    layers.append(Clamp(-10, 10))
    return Sequential(*layers)


class BoxRFDGCNN(torch.nn.Module):
    def __init__(self, node_input_dim, node_output_dim, edge_input_dim,
                 rf_input_dim, rf_output_dim, txp_input_dim, txp_output_dim, imf_input_dim, imf_output_dim,
                 num_node_classes, num_edge_classes, hidden_dim=128, dgcn_dim=64, aggr='max', k=10, option={}):

        super(BoxRFDGCNN, self).__init__()
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

        conv_type = 'DGC'
        if 'conv_type' in option:
            conv_type = option['conv_type']

        self.gcv_conv_num = 2
        if 'conv_num' in option:
            self.gcv_conv_num = option['conv_num']

        self.gcn_conv_merge = ''
        if 'conv_merge_type' in option:
            self.gcn_conv_merge= option['conv_merge_type']

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
                node_dim = node_input_dim * 2
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

        if 'image' in self.feature_types:
            if 'BIM' in self.normalization_pos:
                self.img_feature_embed = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(imf_input_dim, affine=True),
                    torch.nn.Linear(imf_input_dim, imf_output_dim),
                    torch.nn.ReLU(inplace=True)
                )
            else:
                self.img_feature_embed = torch.nn.Sequential(
                    torch.nn.Linear(imf_input_dim, imf_output_dim),
                    torch.nn.ReLU(inplace=True)
                )

        fusion_dim = 0
        if 'bbox' in self.feature_types:
            fusion_dim += node_output_dim
        if 'rf' in self.feature_types:
            fusion_dim += rf_output_dim
        if 'text_pattern' in self.feature_types:
            fusion_dim += txp_output_dim
        if 'image' in self.feature_types:
            fusion_dim += imf_output_dim

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

        # 2. DGNN 레이어
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
            edge_input_dim = 18
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

        # 3. 노드 분류를 위한 MLP
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

        # 4. 엣지 분류를 위한 MLP (두 노드의 임베딩 + 엣지 특징을 결합하여 예측)
        # 엣지 임베딩 (hidden_dim * 2) + 엣지 특징 (edge_feature_dim)
        if 'drop_egc' in option:
            self.edge_classifier = torch.nn.Sequential(
                torch.nn.BatchNorm1d((hidden_dim * 2) + edge_input_dim, affine=True),
                torch.nn.Dropout1d(p=option['drop_egc']),
                torch.nn.Linear((hidden_dim * 2) + edge_input_dim, hidden_dim),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(hidden_dim, num_edge_classes),
                torch.nn.ReLU(inplace=True),
            )
        else:
            torch.nn.BatchNorm1d(hidden_dim, affine=True),
            self.edge_classifier = torch.nn.Sequential(
                torch.nn.BatchNorm1d((hidden_dim * 2) + edge_input_dim, affine=True),
                torch.nn.Linear((hidden_dim * 2) + edge_input_dim, hidden_dim),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(hidden_dim, num_edge_classes),
                torch.nn.ReLU(inplace=True),
            )

    def forward_with_batch(self, data):
        bboxes, edge_index, edge_attr, rf_feature, text_patterns, image_feature =\
            (data.x, data.edge_index, data.edge_attr, data.rf_features, data.text_patterns,
             data.img_features)

        k = min(len(bboxes), self.nn_k)
        return self.forward(bboxes, edge_index, edge_attr, rf_feature, text_patterns, image_feature, k, data.batch)


    def forward(self, bboxes, edge_index, edge_attr, rf_features, text_patterns, image_features, k, batch):
        fusion_features = []

        if 'bbox' in self.feature_types:
            box_feat = self.node_embed(bboxes)
            fusion_features.append(box_feat)

        if 'rf' in self.feature_types:
            rf_feat = self.rf_feature_embed(rf_features)
            fusion_features.append(rf_feat)

        if 'text_pattern' in self.feature_types:
            txp_feat = self.txp_feature_embed(text_patterns)
            fusion_features.append(txp_feat)

        if 'image' in self.feature_types:
            im_feat = self.img_feature_embed(image_features)
            fusion_features.append(im_feat)

        fusion_feat = self.fusion_layer(torch.cat(fusion_features, dim=1))

        if self.training and self.training_noise > 0 and self.training_noise_prob > random.random():
            scale = torch.rand(1, device=fusion_feat.device) * self.training_noise
            sigma = fusion_feat.std() * scale
            noise = torch.randn_like(fusion_feat) * sigma
            fusion_feat = fusion_feat + noise

        # GNN 레이어 통과
        if batch is None:
            torch.zeros(bboxes.size(0), dtype=torch.long, device=bboxes.device)

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
            elif self.nn_value == 'BBOX-FEAT':
                nn_bboxes = box_feat

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
                if self.use_local_nn:
                    x = gcn_conv(x, edge_index, edge_attr)
                else:
                    x = gcn_conv(x, edge_index, edge_attr)
                feat_list.append(x)

            if self.gcn_conv_merge == 'DEEP':
                gcn_feat = self.dconv_out(x)  # box_len*512
            else:
                gcn_feat = self.dconv_out(torch.cat(feat_list, dim=1))  # box_len*512
        else:
            feat_list = []
            x = fusion_feat

            for gcn_conv in self.gcn_conv_list:
                if self.use_local_nn:
                    local_edge_index = edge_index
                else:
                    local_edge_index = edge_index
                x = gcn_conv(x, local_edge_index)
                feat_list.append(x)

            if self.gcn_conv_merge == 'DEEP':
                gcn_feat = self.dconv_out(x)  # box_len*512
            else:
                gcn_feat = self.dconv_out(torch.cat(feat_list, dim=1))

        # 노드 종류 예측
        node_feat = torch.maximum(fusion_feat, gcn_feat)
        node_logits = self.node_classifier(node_feat)

        # 엣지 연결 예측
        row, col = edge_index

        # 두 노드 임베딩과 엣지 특징을 concatenate
        edge_embedding_and_attr = torch.cat([node_feat[row], node_feat[col], edge_attr], dim=-1)
        edge_logits = self.edge_classifier(edge_embedding_and_attr)

        return node_logits, edge_logits
