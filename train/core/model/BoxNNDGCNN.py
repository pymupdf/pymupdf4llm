# encoding: utf-8
'''
@software: Pycharm
@time: 2021/11/24 6:27 下午
@desc:
'''
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch.nn import Sequential, Linear, BatchNorm1d, LeakyReLU

from torch_geometric.data import Batch as GraphBatch
from torch_geometric.data import Data as GraphData


def MLP(channels, batch_norm=True):
    layers = []
    for i in range(1, len(channels)):
        layer = [Linear(channels[i - 1], channels[i]), LeakyReLU()]
        if batch_norm:
            layer.append(BatchNorm1d(channels[i]))
        layers.append(Sequential(*layer))
    return Sequential(*layers)


class BoxNNDGCNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, nn_feature_dim,
                 num_node_classes, num_edge_classes, hidden_dim=128, dgcn_dim=64, aggr='max', k=10):
        super(BoxNNDGCNN, self).__init__()
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes

        # 1. Node 특징 임베딩
        self.node_embed = torch.nn.Sequential(
            torch.nn.BatchNorm1d(node_feature_dim, affine=False),
            torch.nn.Linear(node_feature_dim, hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
        )

        self.nn_feature_embed = torch.nn.Sequential(
            torch.nn.BatchNorm1d(nn_feature_dim, affine=False),
            torch.nn.Linear(nn_feature_dim, hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
        )

        self.fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
        )

        # 2. DGNN 레이어
        conv_out = dgcn_dim * 2
        self.conv1 = DynamicEdgeConv(MLP([2 * hidden_dim, dgcn_dim, dgcn_dim, dgcn_dim]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([conv_out, conv_out]), k, aggr)
        self.gcn_out = MLP([dgcn_dim + conv_out, hidden_dim])

        # 3. 노드 분류를 위한 MLP
        self.node_classifier = torch.nn.Sequential(
            # torch.nn.Linear((hidden_dim * 2) + edge_feature_dim, hidden_dim),
            # torch.nn.Linear(hidden_dim, hidden_dim // 2),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.BatchNorm1d(hidden_dim // 2, affine=False),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(hidden_dim, num_node_classes),
            torch.nn.ReLU(inplace=True),
        )

        # 5. 엣지 분류를 위한 MLP (두 노드의 임베딩 + 엣지 특징을 결합하여 예측)
        # 엣지 임베딩 (hidden_dim * 2) + 엣지 특징 (edge_feature_dim)
        self.edge_classifier = torch.nn.Sequential(
            # torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.Linear((hidden_dim * 2) + edge_feature_dim, hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
            # torch.nn.BatchNorm1d(hidden_dim, affine=False),
            torch.nn.Linear(hidden_dim, num_edge_classes),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, data):
        x, edge_index, edge_attr, nn_feature = data.x, data.edge_index, data.edge_attr, data.nn_feature
        self.device = x.device

        # 노드 특징 임베딩
        box_feat = self.node_embed(x)
        nn_feat = self.nn_feature_embed(nn_feature)
        fusion_feat = self.fusion_layer(torch.cat([box_feat, nn_feat], dim=1))

        # GNN 레이어 통과
        batch = data.batch
        if batch is None:
            torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        gcn_feat_1 = self.conv1(fusion_feat, batch)
        gcn_feat_2 = self.conv2(gcn_feat_1, batch)
        gcn_feat = self.gcn_out(torch.cat([gcn_feat_1, gcn_feat_2], dim=1))  # box_len*512

        # 노드 종류 예측
        node_feat = torch.maximum(box_feat, gcn_feat)
        node_logits = self.node_classifier(node_feat)

        # 엣지 연결 예측
        row, col = edge_index

        # 두 노드 임베딩과 엣지 특징을 concatenate
        edge_embedding_and_attr = torch.cat([node_feat[row], node_feat[col], edge_attr], dim=-1)
        edge_logits = self.edge_classifier(edge_embedding_and_attr)

        return node_logits, edge_logits
