import math
import torch
import numpy as np
import networkx as nx

from torch import Tensor

from collections import Counter
from typing import Optional, List, Dict, Tuple


def get_neighbor_features(bboxes, directional_nearest_indices):
    """
    각 bbox에 대한 동서남북 방향의 가장 가까운 이웃 특징을 계산합니다.
    get_bboxes_knn_with_directional_nearest_indices 함수에서 반환된
    directional_nearest_indices 딕셔너리를 사용합니다.

    Args:
        bboxes (np.array): 바운딩 박스들의 배열.
        directional_nearest_indices (dict): 각 bbox에 대한 방향별 가장 가까운 이웃 인덱스 딕셔너리.

    Returns:
        list: 각 bbox에 대한 20차원 특징 벡터 리스트.
    """
    n = len(bboxes)
    features = []

    for i in range(n):
        current = bboxes[i]
        x1, y1, x2, y2 = current
        current_width = x2 - x1
        current_height = y2 - y1
        feature = [0.0] * 20  # Initialize all to 0.0

        # 이웃이 없는 경우를 대비하여 .get() 사용
        neighbors_info = directional_nearest_indices.get(i, {})

        # Check Up direction (North)
        up_neighbor_idx = neighbors_info.get('up')
        if up_neighbor_idx is not None:
            neighbor = bboxes[up_neighbor_idx]
            nx1, ny1, nx2, ny2 = neighbor
            # 원본 get_neighbor_features의 조건 (ny2 < y1)은 이미 이웃 검색에서 고려되었을 것으로 가정
            # 만약 directional_nearest_indices에 잘못된 방향의 이웃이 포함될 수 있다면 추가 검증 필요
            distance = y1 - ny2
            neighbor_w = nx2 - nx1
            neighbor_h = ny2 - ny1
            left_diff = abs(x1 - nx1)
            right_diff = abs(x2 - nx2)
            width_diff = abs(current_width - neighbor_w)
            height_diff = abs(current_height - neighbor_h)
            feature[0] = distance
            feature[1] = left_diff
            feature[2] = right_diff
            feature[3] = width_diff
            feature[4] = height_diff

        # Check Right direction (East)
        right_neighbor_idx = neighbors_info.get('right')
        if right_neighbor_idx is not None:
            neighbor = bboxes[right_neighbor_idx]
            nx1, ny1, nx2, ny2 = neighbor
            distance = nx1 - x2
            neighbor_w = nx2 - nx1
            neighbor_h = ny2 - ny1
            left_diff = abs(x1 - nx1)
            right_diff = abs(x2 - nx2)
            width_diff = abs(current_width - neighbor_w)
            height_diff = abs(current_height - neighbor_h)
            feature[5] = distance
            feature[6] = left_diff
            feature[7] = right_diff
            feature[8] = width_diff
            feature[9] = height_diff

        # Check Down direction (South)
        down_neighbor_idx = neighbors_info.get('down')
        if down_neighbor_idx is not None:
            neighbor = bboxes[down_neighbor_idx]
            nx1, ny1, nx2, ny2 = neighbor
            distance = ny1 - y2
            neighbor_w = nx2 - nx1
            neighbor_h = ny2 - ny1
            left_diff = abs(x1 - nx1)
            right_diff = abs(x2 - nx2)
            width_diff = abs(current_width - neighbor_w)
            height_diff = abs(current_height - neighbor_h)
            feature[10] = distance
            feature[11] = left_diff
            feature[12] = right_diff
            feature[13] = width_diff
            feature[14] = height_diff

        # Check Left direction (West)
        left_neighbor_idx = neighbors_info.get('left')
        if left_neighbor_idx is not None:
            neighbor = bboxes[left_neighbor_idx]
            nx1, ny1, nx2, ny2 = neighbor
            distance = x1 - nx2
            neighbor_w = nx2 - nx1
            neighbor_h = ny2 - ny1
            left_diff = abs(x1 - nx1)
            right_diff = abs(x2 - nx2)
            width_diff = abs(current_width - neighbor_w)
            height_diff = abs(current_height - neighbor_h)
            feature[15] = distance
            feature[16] = left_diff
            feature[17] = right_diff
            feature[18] = width_diff
            feature[19] = height_diff

        features.append(feature)
    return np.array(features)  # NumPy 배열로 반환


def get_edge_transform_bbox(bboxes: np.ndarray, edge_index: list):
    """
    Vectorized edge feature computation in PyTorch,
    synced to produce the same output as get_relation_feature.
    bboxes: (N, 4) numpy array in [x_min, y_min, x_max, y_max] format
    edge_index: (E, 2) list of tuples
    Returns: (E, 18) torch.Tensor
    """
    if len(edge_index) == 0:
        return torch.empty(0, 18, dtype=torch.float32)

    # Convert edge_index list of tuples to numpy array, then transpose
    edge_index_np = np.array(edge_index).T # (2, E)

    # Extract source (S) and object (O) bboxes using numpy indexing
    # These will be (E, 4) numpy arrays: [x_min, y_min, x_max, y_max]
    S_np = bboxes[edge_index_np[0]]
    O_np = bboxes[edge_index_np[1]]

    # Convert numpy arrays to PyTorch tensors for calculations
    S = torch.from_numpy(S_np).float() # (E, 4)
    O = torch.from_numpy(O_np).float() # (E, 4)

    delta = 1e-10 # Prevent division by zero

    # Calculate widths and heights for S and O
    sw = S[:, 2] - S[:, 0] # S_width
    sh = S[:, 3] - S[:, 1] # S_height
    ow = O[:, 2] - O[:, 0] # O_width
    oh = O[:, 3] - O[:, 1] # O_height

    # Calculate enclosing bounding box (R)
    # R: [out_x_min, out_y_min, out_x_max, out_y_max]
    R_x1 = torch.min(S[:, 0], O[:, 0])
    R_y1 = torch.min(S[:, 1], O[:, 1])
    R_x2 = torch.max(S[:, 2], O[:, 2])
    R_y2 = torch.max(S[:, 3], O[:, 3])
    R = torch.stack([R_x1, R_y1, R_x2, R_y2], dim=1) # (E, 4)

    rw = R[:, 2] - R[:, 0] # R_width
    rh = R[:, 3] - R[:, 1] # R_height

    # Construct the 18 features exactly as in get_relation_feature
    features = []

    # Features 1-6: Relative to S and O
    features.append((S[:, 0] - O[:, 0]) / (sw + delta)) # 1. (x1_min - x2_min) / width1
    features.append((S[:, 1] - O[:, 1]) / (sh + delta)) # 2. (y1_min - y2_min) / height1
    features.append((O[:, 0] - S[:, 0]) / (ow + delta)) # 3. (x2_min - x1_min) / width2
    features.append((O[:, 1] - S[:, 1]) / (oh + delta)) # 4. (y2_min - y1_min) / height2
    features.append(torch.log(sw / (ow + delta)))      # 5. log(width1 / width2)
    features.append(torch.log(sh / (oh + delta)))      # 6. log(height1 / height2)

    # Features 7-12: Relative to S and R (out_box)
    features.append((S[:, 0] - R[:, 0]) / (sw + delta)) # 7. (x1_min - out_x_min) / width1
    features.append((S[:, 1] - R[:, 1]) / (sh + delta)) # 8. (y1_min - out_y_min) / height1
    features.append((R[:, 0] - S[:, 0]) / (rw + delta)) # 9. (out_x_min - x1_min) / out_width
    features.append((R[:, 1] - S[:, 1]) / (rh + delta)) # 10. (out_y_min - y1_min) / out_height
    features.append(torch.log(sw / (rw + delta)))       # 11. log(width1 / out_width)
    features.append(torch.log(sh / (rh + delta)))       # 12. log(height1 / out_height)

    # Features 13-18: Relative to O and R (out_box)
    features.append((O[:, 0] - R[:, 0]) / (ow + delta)) # 13. (x2_min - out_x_min) / width2
    features.append((O[:, 1] - R[:, 1]) / (oh + delta)) # 14. (y2_min - out_y_min) / height2
    features.append((R[:, 0] - O[:, 0]) / (rw + delta)) # 15. (out_x_min - x2_min) / out_width
    features.append((R[:, 1] - O[:, 1]) / (rh + delta)) # 16. (out_y_min - y2_min) / out_height
    features.append(torch.log(ow / (rw + delta)))       # 17. log(width2 / out_width)
    features.append(torch.log(oh / (rh + delta)))       # 18. log(height2 / out_height)

    # Stack all features column-wise
    edge_attr = torch.stack(features, dim=1) # (E, 18)

    return edge_attr


def get_edge_transform_bbox_tensor(bboxes: torch.Tensor, edge_index: torch.Tensor):
    """
    Computes 18-dimensional edge features between connected bounding boxes.
    This version operates directly on PyTorch tensors, with edge_index of shape (2, E).

    Args:
        bboxes (torch.Tensor): A tensor of shape (N, 8) where each row is a bounding box
                               in the format [norm_x1, norm_y1, norm_x2, norm_y2, ctr_x, ctr_y, box_w, box_h].
        edge_index (torch.Tensor): A tensor of shape (2, E) containing the indices of
                                   connected bounding boxes.

    Returns:
        torch.Tensor: A tensor of shape (E, 18) containing the computed edge features.
    """
    if edge_index.numel() == 0:
        return torch.empty(0, 18, dtype=bboxes.dtype)

    # Use the first 4 columns of bboxes which are [norm_x1, norm_y1, norm_x2, norm_y2]
    # S: source bounding boxes, O: object bounding boxes
    # Note: edge_index is now (2, E), so we index with [0] and [1]
    S = bboxes[edge_index[0], :4] # (E, 4)
    O = bboxes[edge_index[1], :4] # (E, 4)

    delta = 1e-10 # Prevent division by zero

    # Calculate widths and heights for S and O
    sw = S[:, 2] - S[:, 0] # S_width
    sh = S[:, 3] - S[:, 1] # S_height
    ow = O[:, 2] - O[:, 0] # O_width
    oh = O[:, 3] - O[:, 1] # O_height

    # Calculate enclosing bounding box (R)
    R_x1 = torch.min(S[:, 0], O[:, 0])
    R_y1 = torch.min(S[:, 1], O[:, 1])
    R_x2 = torch.max(S[:, 2], O[:, 2])
    R_y2 = torch.max(S[:, 3], O[:, 3])
    R = torch.stack([R_x1, R_y1, R_x2, R_y2], dim=1) # (E, 4)

    rw = R[:, 2] - R[:, 0] # R_width
    rh = R[:, 3] - R[:, 1] # R_height

    # Construct the 18 features exactly as in the original function
    features = []

    # Features 1-6: Relative to S and O
    features.append((S[:, 0] - O[:, 0]) / (sw + delta))
    features.append((S[:, 1] - O[:, 1]) / (sh + delta))
    features.append((O[:, 0] - S[:, 0]) / (ow + delta))
    features.append((O[:, 1] - S[:, 1]) / (oh + delta))
    features.append(torch.log(sw / (ow + delta)))
    features.append(torch.log(sh / (oh + delta)))

    # Features 7-12: Relative to S and R (out_box)
    features.append((S[:, 0] - R[:, 0]) / (sw + delta))
    features.append((S[:, 1] - R[:, 1]) / (sh + delta))
    features.append((R[:, 0] - S[:, 0]) / (rw + delta))
    features.append((R[:, 1] - S[:, 1]) / (rh + delta))
    features.append(torch.log(sw / (rw + delta)))
    features.append(torch.log(sh / (rh + delta)))

    # Features 13-18: Relative to O and R (out_box)
    features.append((O[:, 0] - R[:, 0]) / (ow + delta))
    features.append((O[:, 1] - R[:, 1]) / (oh + delta))
    features.append((R[:, 0] - O[:, 0]) / (rw + delta))
    features.append((R[:, 1] - O[:, 1]) / (rh + delta))
    features.append(torch.log(ow / (rw + delta)))
    features.append(torch.log(oh / (rh + delta)))

    # Stack all features column-wise
    edge_attr = torch.stack(features, dim=1) # (E, 18)

    return edge_attr


def get_edge_by_knn(bboxes: np.ndarray, k: int) -> np.ndarray:
    """
    Computes a k-nearest neighbor graph based on the center points of bounding boxes,
    returning the result as a numpy array.

    Args:
        bboxes (np.ndarray): A numpy array of shape (N, 4) where each row is a bounding box
                             in the format [x1, y1, x2, y2].
        k (int): The number of nearest neighbors for each node.

    Returns:
        np.ndarray: A numpy array of shape (E, 2) representing the directed edge_index.
    """
    from torch_geometric.nn import knn_graph

    if bboxes.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)

    # Convert bboxes to a torch tensor for knn_graph computation
    bboxes_tensor = torch.from_numpy(bboxes).float()

    # Calculate center points of each bounding box
    center_points = torch.zeros(bboxes_tensor.shape[0], 2)
    center_points[:, 0] = (bboxes_tensor[:, 0] + bboxes_tensor[:, 2]) / 2
    center_points[:, 1] = (bboxes_tensor[:, 1] + bboxes_tensor[:, 3]) / 2

    # Create a k-nearest neighbor graph based on the center points
    # knn_graph returns a tensor of shape (2, E)
    edge_index = knn_graph(center_points, k=k, loop=False)

    # Transpose the tensor to (E, 2) and convert to a numpy array
    edge_index_np = edge_index.T.numpy()

    return edge_index_np

def get_edge_by_corner_knn(bboxes, k=1):
    """
    bboxes: numpy array of shape (N, 4), each row = (x1, y1, x2, y2)
    k: number of neighbors per direction (int)
    returns: numpy array of shape (2, E) with dtype=int (src_indices, dst_indices)
    """
    bboxes = np.asarray(bboxes)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError("bboxes must be shape (N, 4)")
    N = bboxes.shape[0]
    if N == 0:
        return np.zeros((2, 0), dtype=np.int64)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # Safely bound k (self excluded)
    max_k = max(0, N - 1)
    k = int(min(k, max_k))
    if k == 0:
        return np.zeros((2, 0), dtype=np.int64)

    # Compute distance matrices for each directional relation.
    # Row i, col j: distance between feature_of_i and feature_of_j
    # top(i) vs bottom(j)
    dist_top_vs_bottom = np.abs(y1[:, None] - y2[None, :])
    # left(i) vs right(j)
    dist_left_vs_right = np.abs(x1[:, None] - x2[None, :])
    # right(i) vs left(j)
    dist_right_vs_left = np.abs(x2[:, None] - x1[None, :])
    # bottom(i) vs top(j)
    dist_bottom_vs_top = np.abs(y2[:, None] - y1[None, :])

    # Exclude self by setting diagonal to infinity
    for D in (dist_top_vs_bottom, dist_left_vs_right, dist_right_vs_left, dist_bottom_vs_top):
        np.fill_diagonal(D, np.inf)

    src_list = []
    dst_list = []

    # Helper: for a distance matrix D (N,N), for each row i pick up to k smallest columns
    def pick_k_neighbors(D, k):
        # Use argpartition to get k candidate column indices per row (unsorted)
        idx_sorted = np.argpartition(D, kth=k-1, axis=1)[:, :k]
        neighbors = []
        for i in range(N):
            # Sort the candidates by actual distance and filter out invalid (inf) entries
            candidates = idx_sorted[i]
            valid_mask = np.isfinite(D[i, candidates])
            valid_cands = candidates[valid_mask]
            if valid_cands.size == 0:
                neighbors.append(np.array([], dtype=np.int64))
                continue
            # Order valid candidates by distance
            order = np.argsort(D[i, valid_cands])
            neighbors.append(valid_cands[order])
        return neighbors  # list length N, each element is array (<=k)

    # Neighbor lists for each direction
    neigh_tb = pick_k_neighbors(dist_top_vs_bottom, k)    # top -> bottom
    neigh_lr = pick_k_neighbors(dist_left_vs_right, k)    # left -> right
    neigh_rl = pick_k_neighbors(dist_right_vs_left, k)    # right -> left
    neigh_bt = pick_k_neighbors(dist_bottom_vs_top, k)    # bottom -> top

    # Collect edges as src -> dst
    for i in range(N):
        if neigh_tb[i].size > 0:
            src_list.extend([i] * neigh_tb[i].size)
            dst_list.extend(neigh_tb[i].tolist())
        if neigh_lr[i].size > 0:
            src_list.extend([i] * neigh_lr[i].size)
            dst_list.extend(neigh_lr[i].tolist())
        if neigh_rl[i].size > 0:
            src_list.extend([i] * neigh_rl[i].size)
            dst_list.extend(neigh_rl[i].tolist())
        if neigh_bt[i].size > 0:
            src_list.extend([i] * neigh_bt[i].size)
            dst_list.extend(neigh_bt[i].tolist())

    if len(src_list) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    edge_index = np.vstack([np.array(src_list, dtype=np.int64),
                            np.array(dst_list, dtype=np.int64)])
    return edge_index.T.tolist()



def get_edge_by_directional_nn(bbox: np.ndarray, dist_threshold: int, vertical_gap: float = 0.3) -> tuple[
    list[tuple[int, int]], dict[str, list[tuple[int, int]]]]:
    bbox_num = len(bbox)
    eye_matrix = np.eye(bbox_num, dtype=bool)  # eye_matrix를 명시적으로 bool 타입으로 생성

    # 중심 좌표 및 높이 계산
    center_y_box = (bbox[:, 1] + bbox[:, 3]) / 2
    center_x_box = (bbox[:, 0] + bbox[:, 2]) / 2
    bbox_height = bbox[:, 3] - bbox[:, 1]

    # 각 bbox의 중심이 다른 bbox의 수직 범위(vertical_gap 확장) 밖에 있는지 플래그
    y_center_flag = np.logical_or(
        center_y_box.repeat(bbox_num).reshape((bbox_num, -1)).T < (bbox[:, 1] - bbox_height * vertical_gap).repeat(
            bbox_num).reshape((bbox_num, -1)),
        center_y_box.repeat(bbox_num).reshape((bbox_num, -1)).T > (bbox[:, 3] + bbox_height * vertical_gap).repeat(
            bbox_num).reshape((bbox_num, -1))
    )

    # 각 bbox의 중심이 다른 bbox의 수평 범위 밖에 있는지 플래그
    x_center_flag = np.logical_or(
        center_x_box.repeat(bbox_num).reshape((bbox_num, -1)).T < bbox[:, 0].repeat(bbox_num).reshape((bbox_num, -1)),
        center_x_box.repeat(bbox_num).reshape((bbox_num, -1)).T > bbox[:, 2].repeat(bbox_num).reshape((bbox_num, -1))
    )

    # 유클리드 거리 행렬 계산 (이 부분도 원본과 동일하게 repeat().reshape().T 패턴 적용)
    hor_x_dis_matrix = (bbox[:, 0].repeat(bbox_num).reshape((bbox_num, -1)).T - bbox[:, 2].repeat(bbox_num).reshape(
        (bbox_num, -1))) ** 2
    hor_y_dis_matrix = (bbox[:, 1].repeat(bbox_num).reshape((bbox_num, -1)).T - bbox[:, 1].repeat(bbox_num).reshape(
        (bbox_num, -1))) ** 2

    ver_x_dis_matrix = (bbox[:, 0].repeat(bbox_num).reshape((bbox_num, -1)).T - bbox[:, 0].repeat(bbox_num).reshape(
        (bbox_num, -1))) ** 2
    ver_y_dis_matrix = (bbox[:, 1].repeat(bbox_num).reshape((bbox_num, -1)).T - bbox[:, 3].repeat(bbox_num).reshape(
        (bbox_num, -1))) ** 2

    # --- get_filtered_distance_matrix 함수 ---
    def get_filtered_distance_matrix(base_distance_matrix, *conditions):
        filtered_matrix = np.copy(base_distance_matrix)

        all_conditions = []
        for i, cond in enumerate(conditions):
            if cond is not None:
                all_conditions.append(cond)
            else:
                all_conditions.append(np.full(base_distance_matrix.shape, False, dtype=bool))

        if not all_conditions:
            filter_mask = np.full(base_distance_matrix.shape, False, dtype=bool)
        else:
            filter_mask = np.logical_or.reduce(all_conditions)

        filtered_matrix[filter_mask] = math.inf
        return filtered_matrix

    # 각 방향별 거리 행렬 및 필터링
    # 조건식도 원본 함수의 패턴에 맞춰서 생성하여 전달
    left_dis_matrix = get_filtered_distance_matrix(
        hor_x_dis_matrix + hor_y_dis_matrix,
        y_center_flag,
        eye_matrix,
        bbox[:, 2].repeat(bbox_num).reshape((bbox_num, -1)).T <= bbox[:, 2].repeat(bbox_num).reshape((bbox_num, -1))
    )

    right_dis_matrix = get_filtered_distance_matrix(
        hor_x_dis_matrix + hor_y_dis_matrix,
        y_center_flag,
        eye_matrix,
        bbox[:, 0].repeat(bbox_num).reshape((bbox_num, -1)).T >= bbox[:, 0].repeat(bbox_num).reshape((bbox_num, -1))
    )

    up_dis_matrix = get_filtered_distance_matrix(
        ver_x_dis_matrix + ver_y_dis_matrix,
        eye_matrix,
        bbox[:, 3].repeat(bbox_num).reshape((bbox_num, -1)).T <= bbox[:, 3].repeat(bbox_num).reshape((bbox_num, -1))
    )

    down_dis_matrix = get_filtered_distance_matrix(
        ver_x_dis_matrix + ver_y_dis_matrix,
        eye_matrix,
        bbox[:, 1].repeat(bbox_num).reshape((bbox_num, -1)).T >= bbox[:, 1].repeat(bbox_num).reshape((bbox_num, -1))
    )
    down_y_dis_matrix = np.copy(ver_y_dis_matrix)
    up_dis_flag_for_y_original_logic = np.logical_or(
        eye_matrix,
        bbox[:, 3].repeat(bbox_num).reshape((bbox_num, -1)).T <= bbox[:, 3].repeat(bbox_num).reshape((bbox_num, -1))
    )

    down_y_filter_mask = np.logical_or(up_dis_flag_for_y_original_logic, x_center_flag)
    down_y_dis_matrix[down_y_filter_mask] = math.inf

    # 각 방향에서 가장 가까운 이웃 찾기
    def get_nearest_indices(distance_matrix, matrix_name="Unnamed"):
        min_distances = np.min(distance_matrix, axis=1)
        min_indices = np.argmin(distance_matrix, axis=1)
        valid_mask = min_distances != math.inf

        indices = [(int(i), int(min_indices[i])) for i in range(bbox_num) if valid_mask[i]]
        return indices

    # 각 방향별 원본 엣지들을 저장할 딕셔너리
    direction_edges = {}

    left_edges = get_nearest_indices(left_dis_matrix, "left_dis_matrix")
    direction_edges['left'] = left_edges

    right_edges = get_nearest_indices(right_dis_matrix, "right_dis_matrix")
    direction_edges['right'] = right_edges

    up_edges = get_nearest_indices(up_dis_matrix, "up_dis_matrix")
    direction_edges['up'] = up_edges

    down_edges = get_nearest_indices(down_dis_matrix, "down_dis_matrix")
    direction_edges['down'] = down_edges

    down_y_edges = get_nearest_indices(down_y_dis_matrix, "down_y_dis_matrix")
    direction_edges['down_y'] = down_y_edges

    # 수직 방향으로 두 번째 가까운 이웃 찾기
    up_dis_matrix_copy = np.copy(up_dis_matrix)
    down_dis_matrix_copy = np.copy(down_dis_matrix)

    # 첫 번째 이웃을 무한대로 설정하여 두 번째 이웃 찾기
    for i in range(bbox_num):
        if np.min(up_dis_matrix_copy[i]) != math.inf:
            up_dis_matrix_copy[i, np.argmin(up_dis_matrix_copy[i])] = math.inf
        if np.min(down_dis_matrix_copy[i]) != math.inf:
            down_dis_matrix_copy[i, np.argmin(down_dis_matrix_copy[i])] = math.inf

    second_up_edges = get_nearest_indices(up_dis_matrix_copy, "second_up_dis_matrix")
    direction_edges['second_up'] = second_up_edges

    second_down_edges = get_nearest_indices(down_dis_matrix_copy, "second_down_dis_matrix")
    direction_edges['second_down'] = second_down_edges

    # 모든 원본 엣지들을 하나의 리스트로 합치기 (중복 제거 및 정규화 전)
    all_raw_edges = []
    for edges in direction_edges.values():
        all_raw_edges.extend(edges)

    # 중복 제거 및 정규화 (항상 (작은 인덱스, 큰 인덱스) 형태로 유지)
    ori_edge_index_list = list(
        set([(min(item), max(item)) for item in all_raw_edges])
    )
    # dist_threshold 필터링
    final_edge_list = [
        item for item in ori_edge_index_list if abs(item[1] - item[0]) < dist_threshold
    ]
    final_edge_list_sorted = sorted(final_edge_list, key=lambda x: (x[0], x[1]))

    return final_edge_list, direction_edges


def get_boxes_transform(bboxes_list):
    """
    주어진 바운딩 박스 리스트(페이지 내 여러 박스)를 정규화하고 추가 특징을 계산합니다.

    Args:
        bboxes_list (list of list): [[x1, y1, x2, y2], ...] 형태의 바운딩 박스 리스트.
                                    이 리스트는 일반적으로 단일 페이지 내의 모든 바운딩 박스를 나타냅니다.

    Returns:
        torch.Tensor: 각 바운딩 박스에 대해 정규화된 (x1, y1, x2, y2)와
                      중심 좌표 (ctr_x, ctr_y), 너비 (box_w), 높이 (box_h)를 포함하는 텐서.
                      형태: [num_boxes, 8]
    """
    if not bboxes_list:
        return torch.empty(0, 8) # 입력이 비어있으면 빈 텐서 반환

    # 리스트를 PyTorch 텐서로 변환
    # cell_box_data는 모든 바운딩 박스를 포함하는 단일 텐서가 됩니다.
    bbox_data = torch.tensor(bboxes_list, dtype=torch.float32)

    # 1. 전체 바운딩 박스 셋의 최소/최대 좌표 계산
    # x 좌표 (x1, x2) 중에서 최소값을 찾습니다.
    min_x = bbox_data[:, [0, 2]].min()
    # y 좌표 (y1, y2) 중에서 최소값을 찾습니다.
    min_y = bbox_data[:, [1, 3]].min()

    # 2. 전체 바운딩 박스 셋의 너비와 높이 계산
    # x 좌표 (x1, x2) 중에서 최대값에서 min_x를 빼서 전체 너비를 얻습니다.
    cell_box_w = bbox_data[:, [0, 2]].max() - min_x
    # y 좌표 (y1, y2) 중에서 최대값에서 min_y를 빼서 전체 높이를 얻습니다.
    cell_box_h = bbox_data[:, [1, 3]].max() - min_y

    # ZeroDivisionError 방지를 위해 매우 작은 값으로 대체 (너비/높이가 0인 경우)
    # 실제 환경에서는 너비/높이가 0인 박스가 있을 수 있으므로 방어 코드 추가
    cell_box_w = torch.clamp(cell_box_w, min=1e-6)
    cell_box_h = torch.clamp(cell_box_h, min=1e-6)

    # 3. 바운딩 박스 좌표 정규화
    # x 좌표 (x1, x2)를 min_x를 빼고 cell_box_w로 나눕니다.
    bbox_data[:, [0, 2]] = (bbox_data[:, [0, 2]] - min_x) / cell_box_w
    # y 좌표 (y1, y2)를 min_y를 빼고 cell_box_h로 나눕니다.
    bbox_data[:, [1, 3]] = (bbox_data[:, [1, 3]] - min_y) / cell_box_h

    # 정규화된 바운딩 박스 저장
    boxes = bbox_data

    # 4. 추가 특징 (너비, 높이, 중심 좌표) 계산
    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    ctr_x = (boxes[:, 2] + boxes[:, 0]) / 2
    ctr_y = (boxes[:, 3] + boxes[:, 1]) / 2

    # 5. 모든 특징을 하나의 텐서로 스택
    # [정규화된 x1, 정규화된 y1, 정규화된 x2, 정규화된 y2, ctr_x, ctr_y, box_w, box_h]
    boxes_feat = torch.stack((boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3],
                              ctr_x, ctr_y, box_w, box_h), dim=1)

    return boxes_feat


def get_boxes_transform_page(bboxes_list, page_width, page_height):
    """
    주어진 바운딩 박스 리스트(페이지 내 여러 박스)를 정규화하고 추가 특징을 계산합니다.
    이 함수는 page_width와 page_height를 사용하여 좌표를 정규화합니다.

    Args:
        bboxes_list (list of list): [[x1, y1, x2, y2], ...] 형태의 바운딩 박스 리스트.
                                     이 리스트는 일반적으로 단일 페이지 내의 모든 바운딩 박스를 나타냅니다.
        page_width (int or float): 페이지의 전체 너비.
        page_height (int or float): 페이지의 전체 높이.

    Returns:
        torch.Tensor: 각 바운딩 박스에 대해 정규화된 (x1, y1, x2, y2)와
                      중심 좌표 (ctr_x, ctr_y), 너비 (box_w), 높이 (box_h)를 포함하는 텐서.
                      형태: [num_boxes, 8]
    """
    if not bboxes_list:
        return torch.empty(0, 8) # 입력이 비어있으면 빈 텐서 반환

    # 리스트를 PyTorch 텐서로 변환
    bbox_data = torch.tensor(bboxes_list, dtype=torch.float32)

    # ZeroDivisionError 방지를 위해 page_width, page_height가 0인 경우를 처리합니다.
    # 실제 환경에서는 너비/높이가 0인 페이지가 있을 수 있으므로 방어 코드 추가
    page_width = torch.clamp(torch.tensor(page_width, dtype=torch.float32), min=1e-6)
    page_height = torch.clamp(torch.tensor(page_height, dtype=torch.float32), min=1e-6)

    # 1. 바운딩 박스 좌표 정규화 (페이지 크기 기준)
    # x 좌표 (x1, x2)를 page_width로 나눕니다.
    bbox_data[:, [0, 2]] = bbox_data[:, [0, 2]] / page_width
    # y 좌표 (y1, y2)를 page_height로 나눕니다.
    bbox_data[:, [1, 3]] = bbox_data[:, [1, 3]] / page_height

    # 정규화된 바운딩 박스 저장
    boxes = bbox_data

    # 2. 추가 특징 (너비, 높이, 중심 좌표) 계산
    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    ctr_x = (boxes[:, 2] + boxes[:, 0]) / 2
    ctr_y = (boxes[:, 3] + boxes[:, 1]) / 2

    # 3. 모든 특징을 하나의 텐서로 스택
    # [정규화된 x1, 정규화된 y1, 정규화된 x2, 정규화된 y2, ctr_x, ctr_y, box_w, box_h]
    boxes_feat = torch.stack((boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3],
                              ctr_x, ctr_y, box_w, box_h), dim=1)

    return boxes_feat


def get_text_pattern(text, text_pattern=None, return_vector=True):
    if text_pattern is None:
        text_pattern = ''
        for c in text:
            # Number
            if c.isdigit():
                text_pattern += 'N'
            # Symbol
            elif c in [':', '-', '=', '+', '-', ',', '.', '!', '?', '@', '*', '(', ')', '&']:
                text_pattern += 'S'
            # Space
            elif c in [' ', '\t']:
                text_pattern += 'W'
            # Character
            else:
                text_pattern += 'C'
        compressed_pattern = ''
        prev_c = ''
        prev_repeat = 1
        for c_idx, c in enumerate(text_pattern):
            if c_idx == 0:
                compressed_pattern += c
            else:
                if prev_c == c:
                    prev_repeat += 1
                    continue
                else:
                    compressed_pattern += str(prev_repeat)
                    compressed_pattern += c
                    prev_repeat = 1
            prev_c = c
        if prev_c != '':
            compressed_pattern += str(prev_repeat)
        text_pattern = compressed_pattern

    if return_vector:
        f_text = [0.0] * 10
        for c_idx, c in enumerate(text_pattern):
            if c_idx >= len(f_text):
                break
            if c == 'C':
                f_text[c_idx] = 1 / 13
            elif c == 'S':
                f_text[c_idx] = 2 / 13
            elif c == 'N':
                f_text[c_idx] = 3 / 13
            elif c == 'W':
                f_text[c_idx] = 4 / 13
            elif c.isdigit():
                f_text[c_idx] = (int(c) + 4) / 13
            else:
                raise Exception('Invalid text_pattern = %s' % text_pattern)
        return f_text
    else:
        return text_pattern


def calculate_f1_score(gt_label, predicted_label):
    # 정밀도 계산
    true_positives = np.sum((gt_label == predicted_label) & (gt_label != 0))
    false_positives = np.sum((predicted_label == gt_label) & (gt_label == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # 재현율 계산
    true_positives = np.sum((gt_label == predicted_label) & (gt_label != 0))
    false_negatives = np.sum((predicted_label != gt_label) & (gt_label != 0))

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # F1 스코어 계산
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


def get_edge_matrix(num_node, edge_index, edge_label):
    """
    edge_index와 edge_label을 사용하여 n x n 근접이웃 행렬을 반환합니다.

    Args:
      edge_index: 연결의 인덱스를 나타내는 2D 리스트.
      edge_label: 연결 레이블을 나타내는 1D 리스트. 1은 연결이 있음을, 0은 연결이 없음을 의미합니다.

    Returns:
      n x n 근접이웃 행렬 (numpy array).
    """
    adj_matrix = np.zeros((num_node, num_node), dtype=np.int64)  # 초기 0으로 채워진 n x n 행렬

    row, col = edge_index
    for i in range(len(row)):
        adj_matrix[row[i]][col[i]] = int(edge_label[i])

    return adj_matrix


def group_node_by_edge(node_cls: np.ndarray, edge_matrix: np.ndarray, bboxes: list,
                       class_priority: Optional[list] = None) -> list:
    """
    노드 클래스, 엣지 정보 및 바운딩 박스를 기반으로 연결된 노드들을 그룹화하고
    각 그룹의 대표 클래스 및 통합 바운딩 박스를 결정합니다.

    Args:
        node_cls (np.ndarray): 각 노드의 클래스를 담고 있는 1D NumPy 배열.
                                node_cls[i]는 i번째 노드의 클래스입니다.
        edge_matrix (np.ndarray): 노드 간의 연결 정보를 담고 있는 NxN NumPy 배열.
                                    edge_matrix[i][j] = 1은 i번째 노드와 j번째 노드가 연결되었음을 의미합니다.
        bboxes (list): 각 노드에 해당하는 바운딩 박스 리스트. [[x1, y1, x2, y2], ...] 형태입니다.
                        node_cls 및 edge_matrix의 노드 수와 길이가 동일해야 합니다.
        class_priority (Optional[list]): 각 클래스 ID(인덱스)에 해당하는 우선순위 점수를 담고 있는 리스트.
                                         점수가 높을수록 우선순위가 높습니다. None일 경우, 기존의 동작을 유지합니다.
                                         예: class_priority[class_id]는 해당 클래스의 우선순위 점수입니다.

    Returns:
        list: 각 그룹의 정보(인덱스, 그룹 클래스 및 통합 바운딩 박스)를 담고 있는 딕셔너리 리스트.
              예시: [{'indicies': [그룹 내의 인덱스들], 'group_class': 그룹의 클래스, 'group_bbox': [x1, y1, x2, y2]}]
    """
    num_nodes = len(node_cls)
    visited = [False] * num_nodes
    groups = []

    if len(bboxes) != num_nodes:
        raise ValueError("bboxes의 길이가 node_cls 또는 edge_matrix의 노드 수와 일치해야 합니다.")

    for i in range(num_nodes):
        if not visited[i]:
            current_group_indices = []
            stack = [i]
            visited[i] = True

            while stack:
                node = stack.pop()
                current_group_indices.append(node)

                for neighbor in range(num_nodes):
                    if edge_matrix[node][neighbor] == 1 and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

            group_node_classes = [node_cls[idx] for idx in current_group_indices]
            class_counts = Counter(group_node_classes)

            group_class = None
            if class_counts:
                # class_priority가 None이면 기존 동작을 유지
                if class_priority is None:
                    group_class = class_counts.most_common(1)[0][0]
                else:
                    max_count = 0
                    for cls, count in class_counts.items():
                        if count > max_count:
                            max_count = count

                    tied_classes = [cls for cls, count in class_counts.items() if count == max_count]

                    if len(tied_classes) == 1:
                        group_class = tied_classes[0]
                    else:
                        # 동률 클래스가 여러 개인 경우, class_priority에 따라 선택합니다.
                        best_class = None
                        max_priority_score = -1

                        for cls_id in tied_classes:
                            if 0 <= cls_id < len(class_priority):
                                current_priority_score = class_priority[cls_id]
                            else:
                                current_priority_score = -1

                            if current_priority_score > max_priority_score:
                                max_priority_score = current_priority_score
                                best_class = cls_id
                            elif current_priority_score == max_priority_score:
                                if best_class is None or cls_id < best_class:
                                    best_class = cls_id
                        group_class = best_class

            min_x1, min_y1 = float('inf'), float('inf')
            max_x2, max_y2 = float('-inf'), float('-inf')

            for idx in current_group_indices:
                x1, y1, x2, y2 = bboxes[idx]
                min_x1 = min(min_x1, x1)
                min_y1 = min(min_y1, y1)
                max_x2 = max(max_x2, x2)
                max_y2 = max(max_y2, y2)
            group_bbox = [float(min_x1), float(min_y1), float(max_x2), float(max_y2)]

            groups.append({
                'indicies': sorted(current_group_indices),
                'group_class': int(group_class) if group_class is not None else None,
                'group_bbox': group_bbox
            })
    return groups


def group_node_by_edge_with_networkx(node_cls: np.ndarray, node_score: np.ndarray,
                                      edge_matrix: np.ndarray, bboxes: List[List[float]]) -> List[Dict]:
    """
    노드 클래스, 예측 스코어, 엣지 정보 및 바운딩 박스를 기반으로 연결된 노드들을 그룹화하고
    각 그룹의 대표 클래스 및 통합 바운딩 박스를 결정합니다.
    networkx.connected_components()를 사용하여 연결된 컴포넌트를 찾습니다.
    majority vote에서 동률이 발생하면 해당 클래스에 속하는 노드들의 예측 스코어 합이 가장 큰 것을 선택합니다.

    Args:
        node_cls (np.ndarray): 각 노드의 클래스를 담고 있는 1D NumPy 배열.
                                node_cls[i]는 i번째 노드의 클래스입니다.
        node_score (np.ndarray): 각 노드의 해당 클래스에 대한 예측 스코어를 담고 있는 1D NumPy 배열.
                                 node_score[i]는 i번째 노드의 예측 스코어입니다.
        edge_matrix (np.ndarray): 노드 간의 연결 정보를 담고 있는 NxN NumPy 배열.
                                 edge_matrix[i][j] = 1은 i번째 노드와 j번째 노드가 연결되었음을 의미합니다.
        bboxes (list): 각 노드에 해당하는 바운딩 박스 리스트. [[x1, y1, x2, y2], ...] 형태입니다.
                        node_cls 및 edge_matrix의 노드 수와 길이가 동일해야 합니다.

    Returns:
        list: 각 그룹의 정보(인덱스, 그룹 클래스 및 통합 바운딩 박스)를 담고 있는 딕셔너리 리스트.
              예시: [{'indicies': [그룹 내의 인덱스들], 'group_class': 그룹의 클래스, 'group_bbox': [x1, y1, x2, y2]}]
    """
    num_nodes = len(node_cls)

    if len(bboxes) != num_nodes or len(node_score) != num_nodes:
        raise ValueError("bboxes, node_score의 길이가 node_cls 또는 edge_matrix의 노드 수와 일치해야 합니다.")

    # 1. NetworkX 그래프 생성
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes)) # 0부터 num_nodes-1 까지의 노드 추가

    # edge_matrix를 기반으로 엣지 추가
    # edge_matrix는 대칭 행렬이라고 가정 (무방향 그래프)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # 중복 엣지 추가 방지 및 자기 자신으로의 엣지 제외
            if edge_matrix[i][j] == 1:
                G.add_edge(i, j)

    # 2. connected_components를 사용하여 연결된 컴포넌트(그룹) 찾기
    connected_components = list(nx.connected_components(G))

    groups = []

    for component_indices_set in connected_components:
        current_group_indices = sorted(list(component_indices_set)) # set을 list로 변환하고 정렬

        # 그룹 클래스 결정
        group_node_classes = [node_cls[idx] for idx in current_group_indices]
        group_node_scores = [node_score[idx] for idx in current_group_indices]

        class_counts = Counter(group_node_classes)

        group_class = None
        if class_counts:
            max_count = 0
            for cls, count in class_counts.items():
                if count > max_count:
                    max_count = count

            tied_classes = [cls for cls, count in class_counts.items() if count == max_count]

            if len(tied_classes) == 1:
                group_class = tied_classes[0]
            else:
                # 동률일 경우, 해당 클래스의 예측 스코어 합이 가장 큰 것을 선택
                best_class = None
                max_score_sum = -1.0 # 스코어는 0 이상이므로 -1.0으로 초기화

                for cls_id in tied_classes:
                    # 현재 클래스에 해당하는 노드들의 스코어 합 계산
                    current_class_score_sum = sum(
                        score for i, score in enumerate(group_node_scores)
                        if group_node_classes[i] == cls_id
                    )

                    if current_class_score_sum > max_score_sum:
                        max_score_sum = current_class_score_sum
                        best_class = cls_id
                    elif current_class_score_sum == max_score_sum:
                        # 스코어 합이 같으면 클래스 ID가 더 작은 것을 선택
                        if best_class is None or cls_id < best_class:
                            best_class = cls_id
                group_class = best_class

        # 통합 바운딩 박스 계산
        min_x1, min_y1 = float('inf'), float('inf')
        max_x2, max_y2 = float('-inf'), float('-inf')

        for idx in current_group_indices:
            x1, y1, x2, y2 = bboxes[idx]
            min_x1 = min(min_x1, x1)
            min_y1 = min(min_y1, y1)
            max_x2 = max(max_x2, x2)
            max_y2 = max(max_y2, y2)
        group_bbox = [float(min_x1), float(min_y1), float(max_x2), float(max_y2)]

        groups.append({
            'indicies': current_group_indices, # 이미 정렬됨
            'group_class': int(group_class) if group_class is not None else None,
            'group_bbox': group_bbox
        })
    return groups


def group_node_by_edge_with_networkx_and_class_prior(
    node_cls: np.ndarray,
    node_score: np.ndarray,
    edge_matrix: np.ndarray,
    bboxes: List[List[float]],
    label_priority_list: List[int] # 추가된 인자
) -> List[Dict]:
    """
    노드 클래스, 예측 스코어, 엣지 정보 및 바운딩 박스를 기반으로 연결된 노드들을 그룹화하고
    각 그룹의 대표 클래스 및 통합 바운딩 박스를 결정합니다.
    networkx.connected_components()를 사용하여 연결된 컴포넌트를 찾습니다.
    majority vote에서 동률이 발생하면, 주어진 label_priority_list에 따라 최종 클래스를 결정합니다.

    Args:
        node_cls (np.ndarray): 각 노드의 클래스를 담고 있는 1D NumPy 배열.
                                node_cls[i]는 i번째 노드의 클래스입니다.
        node_score (np.ndarray): 각 노드의 해당 클래스에 대한 예측 스코어를 담고 있는 1D NumPy 배열.
                                 node_score[i]는 i번째 노드의 예측 스코어입니다.
        edge_matrix (np.ndarray): 노드 간의 연결 정보를 담고 있는 NxN NumPy 배열.
                                  edge_matrix[i][j] = 1은 i번째 노드와 j번째 노드가 연결되었음을 의미합니다.
        bboxes (list): 각 노드에 해당하는 바운딩 박스 리스트. [[x1, y1, x2, y2], ...] 형태입니다.
                       node_cls 및 edge_matrix의 노드 수와 길이가 동일해야 합니다.
        label_priority_list (list): 레이블 우선순위를 정의하는 리스트.
                                    낮은 인덱스가 높은 우선순위를 의미합니다.

    Returns:
        list: 각 그룹의 정보(인덱스, 그룹 클래스 및 통합 바운딩 박스)를 담고 있는 딕셔너리 리스트.
              예시: [{'indicies': [그룹 내의 인덱스들], 'group_class': 그룹의 클래스, 'group_bbox': [x1, y1, x2, y2]}]
    """
    num_nodes = len(node_cls)
    tie_class = -1

    if len(bboxes) != num_nodes or len(node_score) != num_nodes:
        raise ValueError("bboxes, node_score의 길이가 node_cls 또는 edge_matrix의 노드 수와 일치해야 합니다.")

    # 1. NetworkX 그래프 생성
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes)) # 0부터 num_nodes-1 까지의 노드 추가

    # edge_matrix를 기반으로 엣지 추가
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # 중복 엣지 추가 방지 및 자기 자신으로의 엣지 제외
            if edge_matrix[i][j] == 1:
                G.add_edge(i, j)

    # 2. connected_components를 사용하여 연결된 컴포넌트(그룹) 찾기
    connected_components = list(nx.connected_components(G))

    groups = []

    for component_indices_set in connected_components:
        current_group_indices = sorted(list(component_indices_set)) # set을 list로 변환하고 정렬

        # 그룹 클래스 결정
        group_node_classes = [node_cls[idx] for idx in current_group_indices]
        # group_node_scores는 더 이상 동률 처리 로직에 직접 사용되지 않지만,
        # 나중에 다른 목적으로 사용될 수 있으므로 일단 추출 유지
        # group_node_scores = [node_score[idx] for idx in current_group_indices]

        class_counts = Counter(group_node_classes)

        group_class = None
        if class_counts:
            max_count = 0
            for cls, count in class_counts.items():
                if count > max_count:
                    max_count = count

            # 최빈값에 해당하는 모든 클래스들을 찾음
            tied_classes = [cls for cls, count in class_counts.items() if count == max_count]

            if len(tied_classes) == 1:
                # 동률이 없으면 단일 최빈 클래스를 선택
                group_class = tied_classes[0]
            else:
                # 동률일 경우, label_priority_list를 사용하여 결정 (get_od_label과 동일한 정책)
                best_class_priority = float('inf') # 가장 낮은 인덱스(가장 높은 우선순위)를 찾기 위함
                selected_class = None

                for cls_id in tied_classes:
                    try:
                        # label_priority_list에서 클래스의 인덱스를 찾음
                        current_priority = label_priority_list.index(cls_id)
                        if current_priority < best_class_priority:
                            best_class_priority = current_priority
                            selected_class = cls_id
                    except ValueError:
                        # label_priority_list에 없는 클래스 ID인 경우, 이를 처리하는 방식은 도메인에 따라 달라질 수 있음.
                        # 여기서는 일단 무시하거나, 기본값으로 처리하거나, 오류를 발생시킬 수 있습니다.
                        # 여기서는 우선순위 목록에 없는 클래스는 선택되지 않도록 합니다.
                        # 만약 모든 tied_classes가 우선순위 목록에 없다면 selected_class는 None으로 남을 수 있습니다.
                        pass
                group_class = selected_class if selected_class is not None else tied_classes[0] if tied_classes else None
                # 만약 label_priority_list에 동률 클래스 중 하나도 없다면, 임의로 첫 번째 동률 클래스를 선택
                # 또는 예외를 발생시키거나, None을 반환하는 등의 정책을 선택할 수 있습니다.
                # 여기서는 tied_classes가 비어있지 않으면 첫 번째를 기본값으로 설정.
                # 일반적으로 label_priority_list는 모든 가능한 클래스를 포함해야 합니다.
                tie_class = group_class

        # 통합 바운딩 박스 계산
        min_x1, min_y1 = float('inf'), float('inf')
        max_x2, max_y2 = float('-inf'), float('-inf')

        # 그룹에 노드가 없는 경우 (예: G.nodes()가 비어있을 때 connected_components가 빈 list를 반환하거나)
        # 또는 connected_components가 단일 노드일 때 등
        if not current_group_indices:
            group_bbox = [0.0, 0.0, 0.0, 0.0] # 기본값 또는 오류 처리
        else:
            for idx in current_group_indices:
                x1, y1, x2, y2 = bboxes[idx]
                min_x1 = min(min_x1, x1)
                min_y1 = min(min_y1, y1)
                max_x2 = max(max_x2, x2)
                max_y2 = max(max_y2, y2)
            group_bbox = [float(min_x1), float(min_y1), float(max_x2), float(max_y2)]

        groups.append({
            'indicies': current_group_indices,
            'group_class': int(group_class) if group_class is not None else None,
            'group_bbox': group_bbox,
            'tie_class': tie_class,
        })
    return groups


def save_model_and_optimizer(model, optimizer, final_model_path):
    if optimizer is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, final_model_path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, final_model_path)


def load_model_and_optimizer(model, model_path, optimizer=None):
    """
    Loads model and optimizer from a file, automatically ignoring layers with mismatched shapes.
    """
    data = torch.load(model_path, map_location=torch.device('cpu'))

    # Get the state dict to load
    ckpt_state_dict = data['model_state_dict'] if 'model_state_dict' in data else data

    # Get the current model's state dict
    model_state_dict = model.state_dict()

    # Identify keys to ignore
    keys_to_ignore = []
    for k, v in ckpt_state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape != v.shape:
            print(f"Ignoring key '{k}' due to shape mismatch: "
                  f"current shape {model_state_dict[k].shape} vs checkpoint shape {v.shape}")
            keys_to_ignore.append(k)

    # Create a new state dict with only the matching keys
    filtered_state_dict = {
        k: v for k, v in ckpt_state_dict.items() if k not in keys_to_ignore
    }

    # Load the filtered state dict
    try:
        model.load_state_dict(filtered_state_dict, strict=False)
        print("Model loaded successfully, ignoring mismatched layers.")
    except RuntimeError as ex:
        print(f"Error loading model state dict: {ex}")

    try:
        if optimizer is not None:
            optimizer.load_state_dict(data['optimizer_state_dict'])
    except Exception as ex:
        print(f'Error loading optimizer state dict: {ex}')


def safe_max_aggregation(message: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    E, F = message.size()
    out = torch.full((num_nodes, F), float('-inf'), device=message.device, dtype=message.dtype)
    for i in range(E):
        node_idx = index[i]
        out[node_idx] = torch.maximum(out[node_idx], message[i])
    out = torch.where(out == float('-inf'), torch.zeros_like(out), out)
    return out

def softmax_aggregation(x, out, target_nodes, num_features, num_nodes, message):
    expanded_index = target_nodes.unsqueeze(1).expand(-1, num_features)  # [E] -> [E, F]
    exp_msg = torch.exp(message)
    denom = torch.zeros(num_nodes, num_features, device=x.device, dtype=message.dtype)
    denom.scatter_add_(0, expanded_index, exp_msg)  # [N, F]

    alpha = exp_msg / (denom[target_nodes] + 1e-8)  # [E, F]
    weighted_msg = alpha * message  # [E, F]
    out.scatter_add_(0, expanded_index, weighted_msg)  # [N, F]
    return out


def custom_knn_single(x: Tensor, k: int) -> Tensor:
    num_nodes = x.shape[0]

    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    y_square = x_square.T
    dot_product = torch.matmul(x, x.T)
    distances = x_square - 2 * dot_product + y_square

    idx = torch.arange(num_nodes, device=x.device)
    eye_mask = idx.unsqueeze(1) == idx.unsqueeze(0)  # bool matrix
    distances = torch.where(eye_mask, torch.full_like(distances, float('inf')), distances)

    actual_k = min(k, num_nodes)
    topk_distances, topk_indices = torch.topk(distances, actual_k, largest=False, dim=1)
    valid_mask = topk_distances != float('inf')

    source_nodes = torch.arange(num_nodes, device=x.device).unsqueeze(1).repeat(1, k).flatten()
    target_nodes = topk_indices.flatten()
    valid_edges_mask = valid_mask.flatten()

    source_nodes = source_nodes[valid_edges_mask]
    target_nodes = target_nodes[valid_edges_mask]

    edge_index = torch.stack([source_nodes, target_nodes], dim=0)

    return edge_index



def custom_knn_batched(x: Tensor, k: int, batch: Optional[Tensor] = None) -> Tensor:
    """
    KNN implementation that can handle batches during training.
    Not directly ONNX-exportable for batch_size > 1 due to dynamic loop.
    """
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

    unique_batches = torch.unique(batch)

    all_edge_indices = []

    for b_idx in unique_batches:
        mask = (batch == b_idx)
        x_graph = x[mask]

        num_nodes_graph = x_graph.shape[0]
        k_graph = min(k, num_nodes_graph)

        if num_nodes_graph == 0 or k_graph == 0:
            continue

        x_square = torch.sum(x_graph ** 2, dim=1, keepdim=True)
        y_square = torch.sum(x_graph ** 2, dim=1, keepdim=True).T
        dot_product = torch.matmul(x_graph, x_graph.T)
        distances = x_square - 2 * dot_product + y_square

        # indices_graph = torch.arange(num_nodes_graph, device=x.device)
        # distances[indices_graph, indices_graph] = float('inf')

        _, topk_indices_graph = torch.topk(distances, k_graph, largest=False, dim=1)

        source_nodes_graph = torch.arange(num_nodes_graph, device=x.device).repeat_interleave(k_graph)
        target_nodes_graph = topk_indices_graph.flatten()

        offset = torch.where(mask)[0][0]
        global_source_nodes = source_nodes_graph + offset
        global_target_nodes = target_nodes_graph + offset

        all_edge_indices.append(torch.stack([global_source_nodes, global_target_nodes]))

    if len(all_edge_indices) == 0:
        # Return empty tensor with correct shape and device
        # Ensure it has 2 rows (source, target)
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    return torch.cat(all_edge_indices, dim=1)


def custom_knn_batched_onnx(x: Tensor, k: int, batch: Optional[Tensor] = None, max_batch_size: int = 1024) -> Tensor:
    """
    배치 데이터를 처리하는 ONNX 호환 k-NN 구현.
    정적 그래프를 위해 최대 배치 크기를 가정.

    Args:
        x: (N, D) 형태의 텐서, N은 총 포인트 수, D는 피처 차원.
        k: 최근접 이웃 수.
        batch: 각 포인트의 배치 인덱스를 나타내는 (N,) 형태의 텐서 (선택).
        max_batch_size: 정적 할당을 위한 최대 배치 크기.

    Returns:
        (2, E) 형태의 텐서, E는 엣지 수로, 소스와 타겟 노드 인덱스를 포함.
    """
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

    N, D = x.shape
    device = x.device

    # 모든 포인트 쌍의 유클리드 거리 제곱 계산
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)  # (N, 1)
    y_square = x_square.T  # (1, N)
    dot_product = torch.matmul(x, x.T)  # (N, N)
    distances = x_square - 2 * dot_product + y_square  # (N, N)

    # 자기 자신과의 엣지(자기 루프) 방지
    indices = torch.arange(N, device=device)
    distances[indices, indices] = float('inf')

    # 배치 간 엣지를 방지하기 위한 마스크 생성
    batch_mask = (batch.unsqueeze(0) == batch.unsqueeze(1))  # (N, N)
    distances = torch.where(batch_mask, distances, torch.full_like(distances, float('inf')))

    # 모든 포인트에 대해 k-최근접 이웃 계산
    k = min(k, N)  # k가 포인트 수를 초과하지 않도록
    _, topk_indices = torch.topk(distances, k, largest=False, dim=1)  # (N, k)

    # 엣지 인덱스 생성
    source_nodes = torch.arange(N, device=device).repeat_interleave(k)  # (N * k,)
    target_nodes = topk_indices.flatten()  # (N * k,)

    # 유효하지 않은 엣지(거리가 무한대인 경우) 필터링
    valid_mask = distances[source_nodes, target_nodes] != float('inf')
    source_nodes = source_nodes[valid_mask]
    target_nodes = target_nodes[valid_mask]

    # 엣지 인덱스 반환
    edge_indices = torch.stack([source_nodes, target_nodes], dim=0)  # (2, E)
    if edge_indices.shape[1] == 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    return edge_indices


def custom_around_box_batched(x: Tensor, gap: float, batch: Optional[Tensor] = None) -> Tensor:
    """
    Given a batch of bounding box features, creates edges between boxes
    that are vertically close, based on a given gap.

    Args:
        x (Tensor): A tensor of bounding box features, typically of shape [num_boxes, 8],
                    where the columns are [x1, y1, x2, y2, ctr_x, ctr_y, box_w, box_h].
        gap (float): An absolute threshold for the vertical distance between boxes,
                     which should be between 0 and 1.
        batch (Optional[Tensor]): A 1D tensor of shape [num_boxes] that specifies
                                  the graph index for each bounding box.

    Returns:
        Tensor: A tensor of shape [2, num_edges] representing the source and target
                nodes of the created edges (edge_index).
    """
    # 1. 입력 텐서의 차원이 8인지 확인
    assert x.shape[1] == 8, f"Input tensor x must have 8 features, but got {x.shape[1]}."
    # 2. gap이 0과 1 사이의 절대값인지 확인
    assert 0 < gap < 1, f"Gap must be an absolute value between 0 and 1, but got {gap}."

    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

    unique_batches = torch.unique(batch)
    all_edge_indices = []

    for b_idx in unique_batches:
        mask = (batch == b_idx)
        x_graph = x[mask]
        num_nodes_graph = x_graph.shape[0]

        if num_nodes_graph < 2:
            continue

        # Get the normalized box coordinates
        y1 = x_graph[:, 1].unsqueeze(1)  # Top y-coordinate, shape [N, 1]
        y2 = x_graph[:, 3].unsqueeze(1)  # Bottom y-coordinate, shape [N, 1]

        # Calculate the vertical distance from box i's bottom to box j's top
        # y2 - y1.T -> [N, 1] - [1, N] = [N, N]
        # This gives the vertical distance between the bottom of box i and top of box j.
        # A positive value means box i is below box j.
        y_dist = y2 - y1.T

        # Define adjacency matrix based on proximity
        # Condition 1: Box i's bottom is above box j's top, with a gap < threshold.
        # This finds connections from a box to a box directly below it.
        # The condition is: -gap < y_dist < 0, which means box i is just above box j.
        adj_downward = (y_dist < 0) & (y_dist > -gap)

        # Condition 2: Box i's top is below box j's bottom, with a gap < threshold.
        # This finds connections from a box to a box directly above it.
        # y1 - y2.T -> [N, 1] - [1, N] = [N, N]
        # A positive value means box i is below box j.
        y_dist_upward = y1 - y2.T
        adj_upward = (y_dist_upward > 0) & (y_dist_upward < gap)

        # Combine the conditions to create the final adjacency matrix.
        # We use a logical OR to connect both downward and upward neighbors.
        adj_matrix = adj_downward | adj_upward

        # Get the indices of the connected nodes
        source_nodes_graph, target_nodes_graph = torch.where(adj_matrix)

        # Apply the batch offset
        offset = torch.where(mask)[0][0]
        global_source_nodes = source_nodes_graph + offset
        global_target_nodes = target_nodes_graph + offset

        all_edge_indices.append(torch.stack([global_source_nodes, global_target_nodes]))

    if len(all_edge_indices) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    return torch.cat(all_edge_indices, dim=1)
