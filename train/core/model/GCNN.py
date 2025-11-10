import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, num_node_classes, num_edge_classes, hidden_dim=128):
        super(GCNN, self).__init__()
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes

        # 1. 노드 특징 임베딩
        self.node_embed = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(node_feature_dim, affine=False),
            torch.nn.Linear(node_feature_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
        )

        # 2. GNN 레이어
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 3. 노드 분류를 위한 MLP
        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(hidden_dim // 2, affine=False),
            torch.nn.Linear(hidden_dim // 2, num_node_classes)
        )

        # 4. 엣지 분류를 위한 MLP (두 노드의 임베딩 + 엣지 특징을 결합하여 예측)
        # 엣지 임베딩 (hidden_dim * 2) + 엣지 특징 (edge_feature_dim)
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(hidden_dim, affine=False),
            torch.nn.Linear(hidden_dim, num_edge_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 노드 특징 임베딩
        x = self.node_embed(x)
        x = F.relu(x)

        # GNN 레이어 통과
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 노드 종류 예측
        node_logits = self.node_classifier(x)

        # 엣지 연결 예측
        row, col = edge_index
        # 두 노드 임베딩과 엣지 특징을 concatenate
        edge_embedding_and_attr = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_logits = self.edge_classifier(edge_embedding_and_attr)

        return node_logits, edge_logits
