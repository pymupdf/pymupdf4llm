import torch
import torch.nn as nn
import onnxruntime as ort

from train.core.common.model_util import custom_knn_batched_onnx2 as custom_knn


class KNNModule(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x, batch):
        return custom_knn(x, self.k, batch)


# ---------------------------
# Test deterministic behavior
# ---------------------------
def test_knn_repeat(num_tests=100, N=16, D=3, k=5):
    """
    Run repeated tests to check PyTorch vs ONNX deterministic kNN.
    """
    match_count = 0

    for i in range(num_tests):
        x = torch.randn(N, D)
        batch = torch.randint(0, 2, (N,), dtype=torch.int64)

        model = KNNModule(k).eval()

        # PyTorch output
        out_pt = model(x, batch)

        # Export to ONNX
        onnx_path = "det_knn.onnx"
        torch.onnx.export(
            model, (x, batch), onnx_path,
            input_names=["x", "batch"],
            output_names=["edge_index"],
            opset_version=17
        )

        # Run ONNX Runtime
        session = ort.InferenceSession(onnx_path)
        out_onnx = session.run(
            ["edge_index"],
            {"x": x.numpy().astype("float32"), "batch": batch.numpy().astype("int64")}
        )[0]

        out_onnx = torch.from_numpy(out_onnx)

        if torch.equal(out_pt, out_onnx):
            match_count += 1

    print(f"PyTorch and ONNX match count: {match_count} out of {num_tests}")


if __name__ == "__main__":
    test_knn_repeat(num_tests=100, N=16, D=3, k=5)
