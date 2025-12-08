import numpy as np

def test_edge_by_alignment():
    from train.infer.common_util import get_edge_by_alignment

    # Normalized bounding boxes (0~1 range)
    bbox_array = np.array([
        [0.1, 0.1, 0.2, 0.2],  # Box 0
        [0.105, 0.105, 0.205, 0.205],  # Box 1 (aligned with Box 0)
        [0.8, 0.1, 0.9, 0.2],  # Box 2 (same top, far away)
        [0.1, 0.5, 0.2, 0.6],  # Box 3 (same left, vertical alignment)
        [0.9, 0.9, 1.0, 1.0]  # Box 4 (far from all others)
    ])
    edges = get_edge_by_alignment(bbox_array, dist_threshold=0.01)

    print("Detected edges:")
    for edge in edges:
        print(f"Box {edge[0]} <--> Box {edge[1]}")


if __name__ == '__main__':
    test_edge_by_alignment()
