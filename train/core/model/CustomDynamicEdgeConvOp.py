from onnxruntime_extensions import onnx_op, PyOp
import numpy as np


@onnx_op(op_type="MaxAggregate", domain="pymulayout", inputs=[PyOp.dt_float, PyOp.dt_int64, PyOp.dt_int64], outputs=[PyOp.dt_float])
def max_aggregate_numpy(messages: np.ndarray, indices: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    Optimized NumPy-based Max Aggregation.
    messages: [E, F]
    indices: [E]
    num_nodes: int
    returns: [N, F] where N == num_nodes
    """
    E, F = messages.shape

    # Handle the case of no edges
    if E == 0:
        return np.zeros((num_nodes, F), dtype=messages.dtype)

    # Use argsort to sort messages based on their target indices
    sorted_indices = np.argsort(indices)
    sorted_messages = messages[sorted_indices]
    sorted_indices = indices[sorted_indices]

    # Find the unique target nodes and their first appearance
    unique_indices, first_occurrence_indices = np.unique(sorted_indices, return_index=True)

    # Use split to partition the sorted messages and compute max
    split_points = first_occurrence_indices[1:]
    split_messages = np.split(sorted_messages, split_points)

    aggregated_messages = np.array([np.max(group, axis=0) for group in split_messages], dtype=messages.dtype)

    # Initialize the final output tensor and fill in the aggregated values
    output = np.zeros((num_nodes, F), dtype=messages.dtype)
    output[unique_indices] = aggregated_messages

    return output
