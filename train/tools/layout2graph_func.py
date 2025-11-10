
import math
import torch
import networkx

import numpy as np

from collections import Counter


def _get_nearest_pair_custom(cell_boxes_array, rope_max_length):
    cell_boxes_num = len(cell_boxes_array)
    eye_matrix = np.eye(cell_boxes_num)
    cell_boxes_height = cell_boxes_array[:, 3] - cell_boxes_array[:, 1]
    center_y_box_array = (cell_boxes_array[:, 1] + cell_boxes_array[:, 3]) / 2
    # find all i where mean(y0_i, y1_i) < y0_j, or mean(y0_i, y1_i) > y1_j for all j
    y_center_flag = np.logical_or(
        center_y_box_array.repeat(cell_boxes_num).reshape((cell_boxes_num, -1)).T <
        (cell_boxes_array[:, 1] - cell_boxes_height * 0.3).repeat(cell_boxes_num).reshape((cell_boxes_num, -1)),
        center_y_box_array.repeat(cell_boxes_num).reshape((cell_boxes_num, -1)).T >
        (cell_boxes_array[:, 3] + cell_boxes_height * 0.3).repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))
    center_x_box_array = (cell_boxes_array[:, 0] + cell_boxes_array[:, 2]) / 2
    x_center_flag = np.logical_or(
        center_x_box_array.repeat(cell_boxes_num).reshape(
            (cell_boxes_num, -1)).T < cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)),
        center_x_box_array.repeat(cell_boxes_num).reshape(
            (cell_boxes_num, -1)).T > cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))
    # find all i where x1_i <= x1_j for all j
    left_dis_flag = np.logical_or(
        y_center_flag,
        np.logical_or(
            eye_matrix, cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape(
                (cell_boxes_num, -1)).T <= cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape((cell_boxes_num, -1))))
    # find all i where x0_i >= x0_j for all j
    right_dis_flag = np.logical_or(
        y_center_flag,
        np.logical_or(
            eye_matrix, cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
                (cell_boxes_num, -1)).T >= cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape((cell_boxes_num, -1))))
    # find all i where y1_i <= y1_j for all j
    up_dis_flag = np.logical_or(
        eye_matrix, cell_boxes_array[:, 3].repeat(cell_boxes_num).reshape(
            (cell_boxes_num, -1)).T <= cell_boxes_array[:, 3].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))
    # find all i where y0_i >= y0_j for all j
    down_dis_flag = np.logical_or(
        eye_matrix, cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
            (cell_boxes_num, -1)).T >= cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))
    # euclidean(x0_i - y1_j)
    hor_x_dis_matrix = (cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
        (cell_boxes_num, -1)).T - cell_boxes_array[:, 2].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))**2
    # euclidean(y0_i - y0_j)
    hor_y_dis_matrix = (cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
        (cell_boxes_num, -1)).T - cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))**2
    # euclidean(x0_i - x0_j)
    ver_x_dis_matrix = (cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape(
        (cell_boxes_num, -1)).T - cell_boxes_array[:, 0].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))**2
    # euclidean(y0_i - y1_j)
    ver_y_dis_matrix = (cell_boxes_array[:, 1].repeat(cell_boxes_num).reshape(
        (cell_boxes_num, -1)).T - cell_boxes_array[:, 3].repeat(cell_boxes_num).reshape((cell_boxes_num, -1)))**2

    left_dis_matrix = hor_x_dis_matrix + hor_y_dis_matrix
    left_dis_matrix[left_dis_flag] = math.inf
    right_dis_matrix = hor_x_dis_matrix + hor_y_dis_matrix
    right_dis_matrix[right_dis_flag] = math.inf
    up_dis_matrix = ver_x_dis_matrix + ver_y_dis_matrix
    up_dis_matrix[up_dis_flag] = math.inf
    down_dis_matrix = ver_x_dis_matrix + ver_y_dis_matrix
    down_dis_matrix[down_dis_flag] = math.inf
    down_y_dis_matrix = ver_y_dis_matrix
    down_y_dis_matrix[up_dis_flag] = math.inf
    down_y_dis_matrix[x_center_flag] = math.inf
    left_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                  np.argmin(left_dis_matrix, axis=1)]).T)[np.min(left_dis_matrix, axis=1) != math.inf]
    right_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                   np.argmin(right_dis_matrix,
                                             axis=1)]).T)[np.min(right_dis_matrix, axis=1) != math.inf]
    up_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                np.argmin(up_dis_matrix, axis=1)]).T)[np.min(up_dis_matrix, axis=1) != math.inf]
    down_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                  np.argmin(down_dis_matrix, axis=1)]).T)[np.min(down_dis_matrix, axis=1) != math.inf]
    down_y_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                    np.argmin(down_y_dis_matrix,
                                              axis=1)]).T)[np.min(down_y_dis_matrix, axis=1) != math.inf]

    # for vertical direction, we pick up at most two nodes w.r.t each node
    up_dis_matrix[np.array(range(cell_boxes_num)), np.argmin(up_dis_matrix, axis=1)] = math.inf
    down_dis_matrix[np.array(range(cell_boxes_num)), np.argmin(down_dis_matrix, axis=1)] = math.inf
    second_up_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                       np.argmin(up_dis_matrix, axis=1)]).T)[np.min(up_dis_matrix, axis=1) != math.inf]
    second_down_index_list = (np.vstack([np.array(range(cell_boxes_num)),
                                         np.argmin(down_dis_matrix,
                                                   axis=1)]).T)[np.min(down_dis_matrix, axis=1) != math.inf]

    # bring all selected neighbor node together
    ori_edge_index_list = left_index_list.tolist() + right_index_list.tolist() + up_index_list.tolist(
    ) + down_index_list.tolist() + second_up_index_list.tolist() + second_down_index_list.tolist(
    ) + down_y_index_list.tolist()
    ori_edge_index_list = list(
        set([(item[0], item[1]) if item[0] < item[1] else (item[1], item[0]) for item in ori_edge_index_list]))
    edge_index_list = [item for item in ori_edge_index_list if item[1] - item[0] < rope_max_length]
    # if len(edge_index_list) < len(ori_edge_index_list):
    #     logger.warning("delete edge:{}".format(set(ori_edge_index_list).difference(set(edge_index_list))))
    return edge_index_list


def get_relation_feature(pair_cell, cell_box):
    if len(pair_cell) == 0:
        return []
    pair_cell = np.array(pair_cell)
    two_cell_box = np.concatenate([cell_box[pair_cell[:, 0]], cell_box[pair_cell[:, 1]]], axis=1)
    out_box = np.array([
        np.min(two_cell_box[:, [0, 2, 4, 6]], axis=1),
        np.min(two_cell_box[:, [1, 3, 5, 7]], axis=1),
        np.max(two_cell_box[:, [0, 2, 4, 6]], axis=1),
        np.max(two_cell_box[:, [1, 3, 5, 7]], axis=1)
    ]).T

    delta = 1e-10
    a = (two_cell_box[:, 6] - two_cell_box[:, 4])
    # if np.count_nonzero(a) != len(a):
    #     print(self.images_name)
    #     exit(0)

    relative_feature = np.array([(two_cell_box[:, 0]-two_cell_box[:, 4])/ ((two_cell_box[:, 2]-two_cell_box[:, 0]) + delta) ,\
                                 (two_cell_box[:, 1] - two_cell_box[:, 5]) / ((two_cell_box[:, 3] - two_cell_box[:, 1]) + delta), \
                                 (two_cell_box[:, 4] - two_cell_box[:, 0]) / ((two_cell_box[:, 6] - two_cell_box[:, 4]) + delta), \
                                 (two_cell_box[:, 5] - two_cell_box[:, 1]) / ((two_cell_box[:, 7] - two_cell_box[:, 5]) + delta), \
                                 np.log((two_cell_box[:, 2] - two_cell_box[:, 0]) / ((two_cell_box[:, 6] - two_cell_box[:, 4]) + delta)), \
                                 np.log((two_cell_box[:, 3] - two_cell_box[:, 1]) / ((two_cell_box[:, 7] - two_cell_box[:, 5]) + delta)), \

                                 (two_cell_box[:, 0] - out_box[:, 0]) / ((two_cell_box[:, 2] - two_cell_box[:, 0]) + delta), \
                                 (two_cell_box[:, 1] - out_box[:, 1]) / ((two_cell_box[:, 3] - two_cell_box[:, 1]) + delta), \
                                 (out_box[:, 0] - two_cell_box[:, 0]) / ((out_box[:, 2] - out_box[:, 0]) + delta), \
                                 (out_box[:, 1] - two_cell_box[:, 1]) / ((out_box[:, 3] - out_box[:, 1]) + delta), \
                                 np.log((two_cell_box[:, 2] - two_cell_box[:, 0]) / ((out_box[:, 2] - out_box[:, 0]) + delta)), \
                                 np.log((two_cell_box[:, 3] - two_cell_box[:, 1]) / ((out_box[:, 3] - out_box[:, 1]) + delta)), \

                                 (two_cell_box[:, 4] - out_box[:, 0]) / ((two_cell_box[:, 6] - two_cell_box[:, 4]) + delta), \
                                 (two_cell_box[:, 5] - out_box[:, 1]) / ((two_cell_box[:, 7] - two_cell_box[:, 5]) + delta), \
                                 (out_box[:, 0] - two_cell_box[:, 4]) / ((out_box[:, 2] - out_box[:, 0]) + delta), \
                                 (out_box[:, 1] - two_cell_box[:, 5]) / ((out_box[:, 3] - out_box[:, 1]) + delta), \
                                 np.log((two_cell_box[:, 6] - two_cell_box[:, 4]) / ((out_box[:, 2] - out_box[:, 0]) + delta)), \
                                 np.log((two_cell_box[:, 7] - two_cell_box[:, 5]) / ((out_box[:, 3] - out_box[:, 1]) + delta)), \
                                 ]).T
    return relative_feature


def post_process(cell_box, node_pred, pair_cell, pair_cell_pred, pair_cell_score_list, label_priority_list, pair_score_threshold=0.55,
                 delete_dif_cls=False):

    adj = np.zeros([len(cell_box), len(cell_box)], dtype='int')
    if len(pair_cell) > 0:
        pair_cell_pred = list(map(int, pair_cell_pred.cpu().numpy().tolist()))
        pair_cell_score_list = torch.nn.functional.softmax(pair_cell_score_list, dim=1)
        all_cell_pairs_score = [
            round(float(pair_cell_score_list[i][1].cpu()), 2)
            for i, pair in enumerate(pair_cell)
            if pair_cell_pred[i] == 1
        ]
        all_cell_pairs = [
            pair for i, pair in enumerate(pair_cell)
            if pair_cell_pred[i] == 1 and pair_cell_score_list[i][1] > pair_score_threshold
        ]

        for pair in all_cell_pairs:
            if delete_dif_cls:
                if node_pred[pair[0]] == node_pred[pair[1]]:
                    adj[pair[0], pair[1]], adj[pair[1], pair[0]] = 1, 1
            else:
                adj[pair[0], pair[1]], adj[pair[1], pair[0]] = 1, 1
    nodenum = adj.shape[0]
    edge_temp = np.where(adj != 0)
    edge = list(zip(edge_temp[0], edge_temp[1]))
    layout_graph = networkx.Graph()
    layout_graph.add_nodes_from(list(range(nodenum)))
    layout_graph.add_edges_from(edge)

    od_label_list = []
    for c in networkx.connected_components(layout_graph):
        # 得到不连通的子集
        subgraph = layout_graph.subgraph(c)
        od_label = get_od_label(subgraph, cell_box, node_pred, label_priority_list)
        od_label_list.append(od_label)

    return od_label_list


def get_od_label(subgraph, cell_box, node_pred, label_priority_list):
    nodeSet = subgraph.nodes()
    cell_box_list = []
    label_list = []
    for node in nodeSet:
        box = cell_box[node]
        cell_box_list.append(box)
        label_list.append(int(node_pred[node].cpu()))

    cell_box_array = np.array(cell_box_list)
    label_counters = Counter(label_list)

    if len(label_counters) > 1 and label_counters.most_common(1)[0][1] == label_counters.most_common(2)[1][1]:
        two_label_list = [label_counters.most_common(1)[0][0], label_counters.most_common(2)[1][0]]
        two_label_priority = [label_priority_list.index(label) for label in two_label_list]
        label = two_label_list[two_label_priority.index(min(two_label_priority))]
    else:
        label = label_counters.most_common(1)[0][0]
    points = [
        float(min(cell_box_array[:, 0])),
        float(min(cell_box_array[:, 1])),
        float(max(cell_box_array[:, 2])),
        float(max(cell_box_array[:, 3]))
    ]
    od_label = {
        'label': label,
        'points': points,
        'nodeSet': nodeSet,
        'subgraph': subgraph,
    }
    return od_label
