import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional
from torch_scatter import scatter_max

from train.core.common.model_util import (custom_knn_batched_onnx2 as custom_kNN,
                                          custom_around_box_batched, safe_max_aggregation, softmax_aggregation)


class CustomDynamicEdgeConv(nn.Module):
    """
    DynamicEdgeConv implementation that supports batching during training
    and single-graph processing for ONNX export (inference).
    """

    def __init__(self, nn_model: Callable, k: int, aggr: str = 'mean', nn_type='KNN',
                 onnx_export_flag=False):
        super().__init__()
        self.nn_model = nn_model
        self.k = k
        self.aggr = aggr
        self.onnx_export_flag = onnx_export_flag
        self.nn_type = nn_type

        if aggr not in ['max', 'mean', 'add', 'softmax']:
            raise ValueError(f"Aggregation method '{aggr}' not supported.")
        # Max aggregation is still NotImplemented in the aggregation step

    def forward(self, x: Tensor, k, batch: Optional[Tensor] = None, bboxes: Optional[Tensor] = None,
                nn_index: Optional[Tensor] = None) -> Tensor:

        if nn_index is None:
            if self.nn_type == 'KNN':
                if bboxes is None:
                    edge_index = custom_kNN(x, k, batch)
                else:
                    edge_index = custom_kNN(bboxes, k, batch)
            elif self.nn_type == 'RANGE':
                edge_index = custom_around_box_batched(bboxes, gap=0.05, batch=batch)
            else:
                raise Exception(f'Invalid nn_type = {self.nn_type}!')
        else:
            edge_index = nn_index

        # Handle cases where no edges are found (e.g., empty graphs)
        if not self.onnx_export_flag:
            if edge_index.numel() == 0:
                # If no edges, return a zero tensor of appropriate shape for the output
                # This handles cases where all graphs in a batch are too small/empty
                # Get the output feature size from a dummy pass through nn_model
                dummy_input_for_nn = torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype)
                output_feature_dim = self.nn_model(torch.cat([dummy_input_for_nn, dummy_input_for_nn], dim=-1)).shape[1]
                return torch.zeros(x.shape[0], output_feature_dim, device=x.device, dtype=x.dtype)

        # 2. Prepare messages
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        x_i = x[target_nodes]  # Features of the target nodes
        x_j = x[source_nodes]  # Features of the source nodes

        # Message function logic
        message = self.nn_model(torch.cat([x_i, x_j - x_i], dim=-1))

        # 3. Aggregation
        num_nodes = x.shape[0]
        num_features = message.shape[1]

        # Initialize output tensor
        out = torch.zeros(num_nodes, num_features, device=x.device, dtype=message.dtype)

        if self.aggr == 'add':
            expanded_index = target_nodes.unsqueeze(1).expand(-1, num_features)  # [E] -> [E, F]
            out.scatter_add_(0, expanded_index, message)  # ONNX-safe

        elif self.aggr == 'mean':
            expanded_index = target_nodes.unsqueeze(1).expand(-1, num_features)  # [E] -> [E, F]
            out.scatter_add_(0, expanded_index, message)  # ONNX-safe

            degree = torch.zeros(num_nodes, 1, device=x.device, dtype=message.dtype)
            ones = torch.ones_like(message[:, :1])  # shape [E,1]
            degree.scatter_add_(0, expanded_index[:, :1], ones)  # [E,1] vs [E,1]
            out = out / (degree + 1e-8)
        elif self.aggr == 'softmax':
            out = softmax_aggregation(x, out, target_nodes, num_features, num_nodes, message)
        elif self.aggr == 'max':
            out = safe_max_aggregation(message, target_nodes, num_nodes)
        return out
