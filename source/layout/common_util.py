import os
import math
import numpy as np
import pickle

import networkx as nx

from collections import Counter
from typing import Optional, List, Dict, Tuple

def get_boxes_transform(bboxes_list):
    """
    Normalizes bounding box coordinates (x1, y1, x2, y2) and
    calculates center coordinates and dimensions.

    Args:
        bboxes_list (list of list): A list of bounding boxes,
                                     e.g., [[x1, y1, x2, y2], ...].
                                     It is assumed that the coordinates are
                                     in the format (top-left, bottom-right).

    Returns:
        numpy.ndarray: A NumPy array where each row contains the normalized
                       (x1, y1, x2, y2), center coordinates (ctr_x, ctr_y),
                       width (box_w), and height (box_h).
                       Shape: [num_boxes, 8]
    """
    if not bboxes_list:
        return np.empty((0, 8), dtype=np.float32) # Return empty array for no boxes

    # Convert the input list of lists to a NumPy array
    # Assume cell_box_data holds the entire bounding box feature map.
    bbox_data = np.array(bboxes_list, dtype=np.float32)

    # 1. Calculate the minimum coordinates of the entire set of bounding boxes
    # Find the minimum x from (x1, x2) across all boxes.
    min_x = bbox_data[:, [0, 2]].min()
    # Find the minimum y from (y1, y2) across all boxes.
    min_y = bbox_data[:, [1, 3]].min()

    # 2. Calculate the width and height of the entire bounding box region
    # Calculate the width based on the maximum x from (x1, x2) and min_x.
    cell_box_w = bbox_data[:, [0, 2]].max() - min_x
    # Calculate the height based on the maximum y from (y1, y2) and min_y.
    cell_box_h = bbox_data[:, [1, 3]].max() - min_y

    # Handle ZeroDivisionError by clamping width/height to a minimum value (like epsilon)
    # If width/height is 0, it would cause division by zero. This prevents that.
    cell_box_w = np.maximum(cell_box_w, 1e-6) # Equivalent to torch.clamp(value, min=1e-6)
    cell_box_h = np.maximum(cell_box_h, 1e-6) # Equivalent to torch.clamp(value, min=1e-6)

    # 3. Normalize bounding box coordinates
    # Normalize x coordinates (x1, x2) by subtracting min_x and dividing by cell_box_w.
    bbox_data[:, [0, 2]] = (bbox_data[:, [0, 2]] - min_x) / cell_box_w
    # Normalize y coordinates (y1, y2) by subtracting min_y and dividing by cell_box_h.
    bbox_data[:, [1, 3]] = (bbox_data[:, [1, 3]] - min_y) / cell_box_h

    # Renaming for clarity (consistent with original logic)
    boxes = bbox_data

    # 4. Calculate center coordinates and dimensions (width, height)
    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    ctr_x = (boxes[:, 2] + boxes[:, 0]) / 2
    ctr_y = (boxes[:, 3] + boxes[:, 1]) / 2

    # 5. Stack all features into a single array
    # Output format: [normalized x1, normalized y1, normalized x2, normalized y2, ctr_x, ctr_y, box_w, box_h]
    boxes_feat = np.stack((boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3],
                           ctr_x, ctr_y, box_w, box_h), axis=1)

    return boxes_feat


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
    if bboxes.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)

    # Calculate center points of each bounding box
    center_points = np.zeros((bboxes.shape[0], 2))
    center_points[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
    center_points[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2

    # Initialize edge list
    edges = []

    # Compute distances and find k-nearest neighbors for each point
    for i in range(len(center_points)):
        # Calculate Euclidean distances from point i to all points
        distances = np.sqrt(np.sum((center_points - center_points[i]) ** 2, axis=1))
        # Set distance to self as infinity to exclude it
        distances[i] = np.inf
        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        # Add directed edges (i -> j)
        for j in nearest_indices:
            edges.append([i, j])

    # Convert edges to numpy array
    edge_index = np.array(edges, dtype=np.int64)

    return edge_index


def get_edge_transform_bbox(bboxes: np.ndarray, edge_index: list):
    """
    Vectorized edge feature computation using NumPy,
    Args:
        bboxes (np.ndarray): (N, 4) NumPy array in [x_min, y_min, x_max, y_max] format.
        edge_index (list): (E, 2) list of tuples, where each tuple (src_idx, dst_idx)
                           represents an edge between bounding boxes.
    Returns:
        np.ndarray: (E, 18) NumPy array representing the edge features.
    """
    if len(edge_index) == 0:
        return np.empty((0, 18), dtype=np.float32)

    # Convert edge_index list of tuples to numpy array
    # edge_index_np will be (E, 2)
    edge_index_np = np.array(edge_index)

    # Extract source (S) and object (O) bboxes using numpy indexing
    # These will be (E, 4) numpy arrays: [x_min, y_min, x_max, y_max]
    S = bboxes[edge_index_np[:, 0]]
    O = bboxes[edge_index_np[:, 1]]

    delta = 1e-10 # Prevent division by zero

    # Calculate widths and heights for S and O
    sw = S[:, 2] - S[:, 0] # S_width
    sh = S[:, 3] - S[:, 1] # S_height
    ow = O[:, 2] - O[:, 0] # O_width
    oh = O[:, 3] - O[:, 1] # O_height

    # Calculate enclosing bounding box (R)
    # R: [out_x_min, out_y_min, out_x_max, out_y_max]
    R_x1 = np.minimum(S[:, 0], O[:, 0])
    R_y1 = np.minimum(S[:, 1], O[:, 1])
    R_x2 = np.maximum(S[:, 2], O[:, 2])
    R_y2 = np.maximum(S[:, 3], O[:, 3])
    R = np.stack([R_x1, R_y1, R_x2, R_y2], axis=1) # (E, 4)

    rw = R[:, 2] - R[:, 0] # R_width
    rh = R[:, 3] - R[:, 1] # R_height

    # Construct the 18 features exactly as in get_relation_feature
    features = []

    # Features 1-6: Relative to S and O
    features.append((S[:, 0] - O[:, 0]) / (sw + delta)) # 1. (x1_min - x2_min) / width1
    features.append((S[:, 1] - O[:, 1]) / (sh + delta)) # 2. (y1_min - y2_min) / height1
    features.append((O[:, 0] - S[:, 0]) / (ow + delta)) # 3. (x2_min - x1_min) / width2
    features.append((O[:, 1] - S[:, 1]) / (oh + delta)) # 4. (y2_min - y1_min) / height2
    features.append(np.log(sw / (ow + delta)))          # 5. log(width1 / width2)
    features.append(np.log(sh / (oh + delta)))          # 6. log(height1 / height2)

    # Features 7-12: Relative to S and R (out_box)
    features.append((S[:, 0] - R[:, 0]) / (sw + delta)) # 7. (x1_min - out_x_min) / width1
    features.append((S[:, 1] - R[:, 1]) / (sh + delta)) # 8. (y1_min - out_y_min) / height1
    features.append((R[:, 0] - S[:, 0]) / (rw + delta)) # 9. (out_x_min - x1_min) / out_width
    features.append((R[:, 1] - S[:, 1]) / (rh + delta)) # 10. (out_y_min - y1_min) / out_height
    features.append(np.log(sw / (rw + delta)))          # 11. log(width1 / out_width)
    features.append(np.log(sh / (rh + delta)))          # 12. log(height1 / out_height)

    # Features 13-18: Relative to O and R (out_box)
    features.append((O[:, 0] - R[:, 0]) / (ow + delta)) # 13. (x2_min - out_x_min) / width2
    features.append((O[:, 1] - R[:, 1]) / (oh + delta)) # 14. (y2_min - out_y_min) / height2
    features.append((R[:, 0] - O[:, 0]) / (rw + delta)) # 15. (out_x_min - x2_min) / out_width
    features.append((R[:, 1] - O[:, 1]) / (rh + delta)) # 16. (out_y_min - y2_min) / out_height
    features.append(np.log(ow / (rw + delta)))          # 17. log(width2 / out_width)
    features.append(np.log(oh / (rh + delta)))          # 18. log(height2 / out_height)

    # Stack all features column-wise
    edge_attr = np.stack(features, axis=1) # (E, 18)
    edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=1e5, neginf=-1e5)
    return edge_attr


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


def extract_bbox_features(
    features: np.ndarray,
    bboxes: List[Tuple[float, float, float, float]],
    page_width: int,
    page_height: int,
    apply_softmax: bool = True,
    concat_mean_max: bool = True,
    add_uncertainty: bool = True
) -> np.ndarray:
    """
    Extract per-bbox features with mean/max pooling and uncertainty metrics.

    Args:
        features (np.ndarray): Feature map tensor (N, C, H, W). Typically N=1.
        bboxes (List[Tuple[float, float, float, float]]): Boxes (x1, y1, x2, y2)
            defined on the same coordinate system as page_width/page_height.
        page_width (int): Width of the image used to produce 'features'.
        page_height (int): Height of the image used to produce 'features'.
        apply_softmax (bool): If True, apply channel-wise softmax per spatial location.
        concat_mean_max (bool): If True, output [mean; max] -> 2C dims; else only mean -> C dims.
        add_uncertainty (bool): If True, append entropy_mean and margin_mean -> +2 dims.

    Returns:
        np.ndarray: Shape (len(bboxes), C or 2C plus optional +2 for uncertainty).
    """
    # Validate shapes
    assert features.ndim == 4, f"features must be (N,C,H,W), got {features.shape}"
    N, C, H, W = features.shape
    assert N == 1, "Only batch size 1 is supported."
    fmap = features[0]  # (C, H, W)

    # Prepare channel-wise normalized map if requested
    # We keep both logits (for pooling) and probs (for uncertainty)
    if apply_softmax:
        f = fmap.reshape(C, -1)
        f = f - f.max(axis=0, keepdims=True)      # numerical stability
        np.exp(f, out=f)
        denom = f.sum(axis=0, keepdims=True) + 1e-12
        probs = (f / denom).reshape(C, H, W)      # (C, H, W)
        fmap_used = probs                          # use probs for pooling if desired
    else:
        fmap_used = fmap                           # use raw features/logits for pooling
        # For uncertainty metrics we still need probabilities
        g = fmap.reshape(C, -1)
        g = g - g.max(axis=0, keepdims=True)
        np.exp(g, out=g)
        denom_g = g.sum(axis=0, keepdims=True) + 1e-12
        probs = (g / denom_g).reshape(C, H, W)     # (C, H, W)

    # Output dimension
    base_dim = C * 2 if concat_mean_max else C
    out_dim = base_dim + (2 if add_uncertainty else 0)
    out = np.zeros((len(bboxes), out_dim), dtype=np.float32)

    # Map image-space bbox -> feature grid indices (inclusive)
    def to_grid(x1, y1, x2, y2):
        # Use floor for start and ceil-1 for end to ensure at least one cell
        gx1 = int(np.floor(x1 * W / float(page_width)))
        gy1 = int(np.floor(y1 * H / float(page_height)))
        gx2 = int(np.ceil (x2 * W / float(page_width))) - 1
        gy2 = int(np.ceil (y2 * H / float(page_height))) - 1

        # Clip to valid range
        gx1 = np.clip(gx1, 0, W - 1)
        gy1 = np.clip(gy1, 0, H - 1)
        gx2 = np.clip(gx2, 0, W - 1)
        gy2 = np.clip(gy2, 0, H - 1)

        # Guarantee at least one cell
        if gx2 < gx1: gx2 = gx1
        if gy2 < gy1: gy2 = gy1
        return gx1, gy1, gx2, gy2

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        gx1, gy1, gx2, gy2 = to_grid(x1, y1, x2, y2)

        # Spatial patch for pooling
        patch = fmap_used[:, gy1:gy2+1, gx1:gx2+1]  # (C, ph, pw)
        if patch.size == 0:
            continue

        # Mean pooling over spatial dims
        mean_vec = patch.mean(axis=(1, 2))  # (C,)

        if concat_mean_max:
            max_vec = patch.max(axis=(1, 2))    # (C,)
            pooled = np.concatenate([mean_vec, max_vec], axis=0)  # (2C,)
        else:
            pooled = mean_vec  # (C,)

        if add_uncertainty:
            # Uncertainty metrics computed from per-location probabilities
            prob_patch = probs[:, gy1:gy2+1, gx1:gx2+1]  # (C, ph, pw)

            # Entropy: -sum_c p_c log p_c, averaged over the patch
            # Add epsilon to avoid log(0)
            eps = 1e-12
            ent_map = -(prob_patch * np.log(prob_patch + eps)).sum(axis=0)  # (ph, pw)
            entropy_mean = ent_map.mean()

            # Margin: top1 - top2, averaged over the patch (lower margin -> more uncertain)
            # Compute per-location top2 using partial sort for efficiency
            C_, ph, pw = prob_patch.shape
            flat = prob_patch.reshape(C_, -1)               # (C, ph*pw)
            # argsort last two is fine for C ~ small; for big C, use argpartition
            # Here we compute top1 and top2 values directly
            # top1
            top1 = flat.max(axis=0)                         # (ph*pw,)
            # zero-out top1 positions to get top2 (safe because probs in [0,1])
            mask_top1 = (flat == top1[None, :])
            # For ties, multiple True; set one of them to 0 by keeping first True
            # Simpler approach: use partition to get top2 without modifying
            # Using argpartition for robustness:
            idx_part = np.argpartition(flat, -2, axis=0)[-2:]  # (2, ph*pw)
            top_two = np.take_along_axis(flat, idx_part, axis=0)  # (2, ph*pw)
            # Sort the top two so [top2, top1]
            top_two.sort(axis=0)
            margins = top_two[1] - top_two[0]               # (ph*pw,)
            margin_mean = margins.mean()

            # Append [entropy_mean, margin_mean]
            pooled = np.concatenate([pooled, np.array([entropy_mean, margin_mean], dtype=np.float32)], axis=0)

        out[i, :pooled.shape[0]] = pooled.astype(np.float32)

    return out


def resize_image(image: np.ndarray, new_size: tuple) -> np.ndarray:
    """
    Resize an image using bilinear interpolation with numpy (vectorized).
    Equivalent to cv2.resize(..., INTER_LINEAR).

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W) or (H, W, C).
    new_size : tuple
        (new_width, new_height)

    Returns
    -------
    np.ndarray
        Resized image.
    """
    h, w = image.shape[:2]
    new_w, new_h = new_size

    # Generate grid of coordinates in output image
    x = (np.arange(new_w) + 0.5) * (w / new_w) - 0.5
    y = (np.arange(new_h) + 0.5) * (h / new_h) - 0.5

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = (x - x0)[None, :]  # shape (1, new_w)
    dy = (y - y0)[:, None]  # shape (new_h, 1)

    # Clip to valid range
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    if image.ndim == 3:
        # Expand dimensions for broadcasting
        Ia = image[y0[:, None], x0[None, :]]  # top-left
        Ib = image[y0[:, None], x1[None, :]]  # top-right
        Ic = image[y1[:, None], x0[None, :]]  # bottom-left
        Id = image[y1[:, None], x1[None, :]]  # bottom-right
    else:
        Ia = image[np.ix_(y0, x0)]
        Ib = image[np.ix_(y0, x1)]
        Ic = image[np.ix_(y1, x0)]
        Id = image[np.ix_(y1, x1)]

    # Bilinear interpolation
    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy)
    wc = (1 - dx) * dy
    wd = dx * dy

    out = (wa[..., None] * Ia + wb[..., None] * Ib +
           wc[..., None] * Ic + wd[..., None] * Id) if image.ndim == 3 else \
        (wa * Ia + wb * Ib + wc * Ic + wd * Id)

    return np.clip(out, 0, 255).astype(image.dtype)


def to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale using numpy only.
    Equivalent to cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Grayscale image, shape (H, W).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a BGR image with shape (H, W, 3).")

    # Extract B, G, R channels
    B = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    R = image[:, :, 2].astype(np.float32)

    # Apply standard BGR ¡æ Gray conversion formula
    gray = 0.114 * B + 0.587 * G + 0.299 * R

    # Clip to valid range and convert back to uint8
    return np.clip(gray, 0, 255).astype(np.uint8)
