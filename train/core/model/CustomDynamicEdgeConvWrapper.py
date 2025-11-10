import torch


class MaxAggregatePlaceholder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, messages, indices, num_nodes):
        # Return zeros, actual op will be replaced by ONNX custom op
        N, F = num_nodes.item(), messages.shape[1]
        return torch.zeros(N, F, device=messages.device, dtype=messages.dtype)

    @staticmethod
    def symbolic(g, messages, indices, num_nodes):
        return g.op(
            "pymulayout::MaxAggregate",  # Custom op type name
            messages,
            indices,
            num_nodes,
            outputs=1
        )
