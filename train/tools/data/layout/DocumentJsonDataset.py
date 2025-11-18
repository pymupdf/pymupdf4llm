import os
import json
import random
import traceback
import hashlib

import anyconfig
import blosc
import pickle

import cv2
import lmdb

import torch
import numpy as np
import pymupdf
import yaml

from torch_geometric.data import Dataset, Data
import matplotlib.cm as cm

from train.infer.common_util import (get_text_pattern, get_boxes_transform, get_edge_by_directional_nn, get_edge_by_knn,
                                     get_edge_transform_bbox, group_node_by_edge_with_networkx_and_class_prior,
                                     get_edge_matrix)
from train.core.common.model_util import get_boxes_transform_page, get_edge_by_corner_knn

from train.infer.common_util import extract_bbox_features, resize_image, to_gray
from train.tools.layout2graph_func import _get_nearest_pair_custom, get_relation_feature, post_process

def visualize_bbox_similarity_heatmap(img_w: int, img_h: int, bboxes: list, image_features: np.ndarray,
                                      colormap_name: str = 'viridis'):
    """
    Visualizes bounding boxes by coloring them based on the similarity of their feature vectors
    to the centroid of all feature vectors, using a heatmap-like continuous color scale.

    Args:
        img_w (int): Width of the entire image.
        img_h (int): Height of the entire image.
        bboxes (list): A list of bounding boxes in [x1, y1, x2, y2] format.
        image_features (np.ndarray): A NumPy array of feature vectors corresponding to each bounding box.
        colormap_name (str): Name of the Matplotlib colormap to use (e.g., 'viridis', 'plasma', 'hot', 'RdYlGn').
                              'viridis' by default provides a good perception of data.
    """
    # Input validation
    if not bboxes or image_features.size == 0:
        print("No bounding boxes or feature vectors provided. Skipping visualization.")
        return

    # Using image_features.shape[0] to get the number of rows (samples) in a NumPy array
    if len(bboxes) != image_features.shape[0]:
        print("Mismatch in the number of bounding boxes and feature vectors. Skipping visualization.")
        return

    num_samples = image_features.shape[0]

    # Create a blank white image (height, width, channels)
    image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # 1. Calculate the centroid of all feature vectors
    feature_centroid = np.mean(image_features, axis=0)

    # 2. Calculate the Euclidean distance of each feature vector from the centroid
    # Using np.linalg.norm for L2 (Euclidean) distance
    distances = np.linalg.norm(image_features - feature_centroid, axis=1)

    # Handle cases where all distances might be the same (e.g., only one bbox, or all features identical)
    if np.max(distances) == np.min(distances):
        # If all distances are the same, assign a neutral color (e.g., green from RdYlGn)
        norm_distances = np.zeros_like(distances) + 0.5
    else:
        # 3. Normalize distances to a 0-1 range for colormap mapping
        norm_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    # Get the colormap
    try:
        cmap = cm.get_cmap(colormap_name)
    except ValueError:
        print(f"Colormap '{colormap_name}' not found. Using 'viridis' as default.")
        cmap = cm.get_cmap('viridis')

    # Convert normalized distances to colors using the colormap
    # Matplotlib colormaps return RGBA (0-1 float), convert to BGR (0-255 int) for OpenCV
    colors_rgba = cmap(norm_distances)  # Returns Nx4 array of RGBA floats
    colors_bgr_255 = (colors_rgba[:, :3] * 255).astype(np.uint8)  # Take RGB, scale to 0-255
    # Convert RGB to BGR for OpenCV
    colors_bgr_255 = colors_bgr_255[:, [2, 1, 0]]

    # Draw each bounding box on the image using its calculated similarity color
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = list(map(int, bbox))
        color = colors_bgr_255[i]

        cv2.rectangle(image, (x1, y1), (x2, y2), tuple(color.tolist()), -1)  # Draw box with thickness 2. Convert np array to tuple.

    return image


class DocumentJsonDataset(Dataset):
    def __init__(self, data_path, class_map, nn_type='directional_nn',
                 nn_k=3,
                 cfg_path=None, rf_names = None, rf_executable = None, pkl_dirs=None,
                 is_train=False, lmdb_save_path=None,
                 feature_extractor_path =None, feature_input_size = None, feature_option=None,
                 pdf_dir=None, filter_class=None, create_from_pdf=False, pdf_input_type=None, opt_save_type=None,
                 remove_unlabelled_data=False,
                 cache_size=0, cache_rate=0.0, show_data=False):

        super().__init__()
        if feature_option is None:
            self.feature_option = {}
        else:
            self.feature_option = feature_option
        self.nn_type = nn_type
        self.nn_k = nn_k
        self.class_map = class_map
        self.filter_class = filter_class
        self.is_train = is_train

        self.create_from_pdf = create_from_pdf
        self.pdf_input_type = pdf_input_type
        self.opt_save_type = opt_save_type
        self.remove_unlabelled_data = remove_unlabelled_data
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.rf_names = rf_names
        self.rf_executable = rf_executable
        self.pkl_dirs = pkl_dirs

        self.feature_extractor = None
        self.feature_extractor_path = feature_extractor_path
        self.do_img_feature_extraction = True
        self.feature_input_size = feature_input_size
        self.pdf_dir = pdf_dir

        self.cache_size = cache_size
        self.cache_rate = cache_rate

        self.filepath_list = []
        self.filename_list = []
        self.data_cache = []
        self.show_data = show_data
        self.read_txn = None
        self.txn_keys = None
        self.on_error = 'None'
        self.lmdb_save_path = lmdb_save_path

        if type(data_path) is str:
            data_path = [data_path]

        for path in data_path:
            for file_name in os.listdir(path):
                if file_name.endswith('.json') or file_name.endswith('.pkl'):
                    json_path = os.path.join(path, file_name)
                    self.filepath_list.append(json_path)
                    self.filename_list.append(file_name)

    def len(self):
        return len(self.filepath_list)

    def get_node_class_and_label(self, bboxes, pdf_path, pkl_data,
                                 remove_unlabelled_data=False):
        from train.core.common.util import get_iou

        # Open PDF and get page image
        doc = pymupdf.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap()
        bytes_img = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = cv2.cvtColor(bytes_img.reshape(pix.height, pix.width, pix.n), cv2.COLOR_BGR2RGB)
        doc.close()

        gt_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)
        gt_label = pkl_data['label']

        if gt_img.shape != page_img.shape:
            gt_resize_x = page_img.shape[1] / gt_img.shape[1]
            gt_resize_y = page_img.shape[0] / gt_img.shape[0]
            for gt_idx in range(len(gt_label)):
                gt_label[gt_idx][0] *= gt_resize_x
                gt_label[gt_idx][1] *= gt_resize_y
                gt_label[gt_idx][2] *= gt_resize_x
                gt_label[gt_idx][3] *= gt_resize_y

        valid_indicies = []
        node_labels = []
        node_cls = []

        margin = 5
        for bbox_idx, bbox in enumerate(bboxes):
            iou_list = []

            for i, gt_data in enumerate(gt_label):
                if gt_data[2] - gt_data[0] <= 0 or gt_data[3] - gt_data[1] <= 0:
                    iou_list.append(-1.0)
                else:
                    label_bbox = gt_data[:4]
                    if ((label_bbox[0] - margin <= bbox[0] and bbox[2] <= label_bbox[2] + margin) and
                            (label_bbox[1] - margin <= bbox[1] and bbox[3] <= label_bbox[
                                3] + margin)):
                        iou_list.append(1.0)
                    else:
                        iou = max(get_iou(bbox, label_bbox), get_iou(label_bbox, bbox))
                        iou_list.append(iou)

            if len(iou_list) > 0:
                max_iou = max(iou_list)
            else:
                max_iou = 0.0

            if max_iou > 0.5:
                index = iou_list.index(max_iou)
                label = gt_label[index][5]
                node_labels.append(f'{label}_{index}')
                valid_indicies.append(bbox_idx)
            else:
                label = 'unlabelled'
                if not remove_unlabelled_data:
                    node_labels.append(f'{label}_{bbox_idx}')
                    valid_indicies.append(bbox_idx)

            if label in self.class_map:
                cls_val = self.class_map[label]
                if label == 'unlabelled' and remove_unlabelled_data:
                    continue
                if cls_val >= 0:
                    node_cls.append(cls_val)
                else:
                    continue
            else:
                # if there is an unlabelled data, assign 'text' class
                if not remove_unlabelled_data:
                    node_cls.append(self.class_map['text'])

        return valid_indicies, node_cls, node_labels

    def get(self, idx):
        if 0 < self.cache_size == len(self.data_cache):
            if np.random.rand() < self.cache_rate:
                idx = np.random.randint(0, self.cache_size)
                return self.data_cache[idx]

        worker_info = torch.utils.data.get_worker_info()
        is_main_process = worker_info is None or worker_info.id == 0

        while True:
            json_path = self.filepath_list[idx]

            # Normal json file
            if json_path.endswith('.json'):
                file_name = self.filename_list[idx][:-5]
                try:
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                except Exception as ex:
                    # self.file_list[idx] = None
                    if self.on_error == 'None':
                        return '%s - %s' % (json_path, str(ex))
                    else:
                        idx = np.random.randint(0, len(self.filepath_list))
                    continue

                if len(json_data['img_data_list']) <= 0 or len(json_data['img_data_list']) > 5000:
                    if self.on_error == 'None':
                        # print(f"Empty or too many bboxes ({len(json_data['img_data_list'])})")
                        return None
                    else:
                        idx = np.random.randint(0, len(self.filepath_list))
                    continue

                page_width, page_height = json_data['meta']['page_width'], json_data['meta']['page_height']

            # No json file, only PDF file
            else:
                file_name = self.filename_list[idx][:-4]

            pdf_path = None
            for pdf_dir in self.pdf_dir:
                pdf_path_temp = f'{pdf_dir}/{file_name}.pdf'
                if os.path.exists(pdf_path_temp):
                    pdf_path = pdf_path_temp
                    break
            if pdf_path is None:
                if self.on_error == 'None':
                    return f'The PDF file for {file_name} was not found...'
                else:
                    raise FileNotFoundError(f'The PDF file for {file_name} was not found...')

            # Check duplicated data if transaction is given
            if self.txn_keys is not None:
                key = hashlib.sha256(file_name.encode('utf-8')).digest()
                if key in self.txn_keys:
                    return f'Skip - already exist - {file_name}'
            else:
                if self.lmdb_save_path is not None and self.read_txn is None:
                    self.lmdb_env = lmdb.open(self.lmdb_save_path, map_size=1024 ** 4, readonly=True, lock=False)
                    self.read_txn = self.lmdb_env.begin(write=False)

                if self.read_txn is not None:
                    key = hashlib.sha256(file_name.encode('utf-8')).digest()
                    if self.read_txn.get(key) is not None:
                        return f'Skip - already exist - {file_name}'

            filter_hit_count = 0

            if self.create_from_pdf:
                try:
                    from train.infer.pymupdf_util import create_input_data_by_pymupdf
                    from train.infer.onnx.BoxRFDGNN import get_nn_input_from_datadict

                    pkl_path = None
                    for pkl_dir in self.pkl_dirs:
                        pkl_path_temp = f'{pkl_dir}/{file_name}.pkl'
                        if os.path.exists(pkl_path_temp):
                            pkl_path = pkl_path_temp
                            break
                    if pkl_path is None:
                        raise FileNotFoundError(f'The PKL file for {file_name} was not found...')

                    # Load and pkl_path ground truth data
                    with open(pkl_path, 'rb') as f:
                        compressed_pickle = f.read()
                    depressed_pickle = blosc.decompress(compressed_pickle)
                    pkl_data = pickle.loads(depressed_pickle)

                    new_labels = []
                    for ant in pkl_data['label']:
                        cls_name = ant[5]
                        # class change
                        if f'_{cls_name}' in self.class_map:
                            cls_name = self.class_map[f'_{cls_name}']
                            ant[5] = cls_name
                            if cls_name is not None:
                                new_labels.append(ant)
                        else:
                            new_labels.append(ant)
                    pkl_data['label'] = new_labels

                    if self.filter_class is not None:
                        for ant in pkl_data['label']:
                            if ant[-1] in self.filter_class:
                                filter_hit_count += 1
                        if filter_hit_count == 0:
                            return 'No target class in pkl_data'

                    try:
                        data_dict = create_input_data_by_pymupdf(pdf_path, input_type=self.pdf_input_type, features_path=self.rf_executable)
                    except Exception as ex:
                        return f'{pdf_path} - {str(ex)}'

                    bboxes = data_dict['bboxes']
                    if len(bboxes) > 5000:
                        return f'Too many bboxes ({len(bboxes)})'

                    valid_indicies, node_cls, node_labels = self.get_node_class_and_label(bboxes, pdf_path, pkl_data,
                                                                                          remove_unlabelled_data=self.remove_unlabelled_data)

                    if len(valid_indicies) != len(data_dict['bboxes']):
                        valid_bboxes = []
                        valid_rf = []
                        valid_text = []
                        for valid_idx in valid_indicies:
                            valid_bboxes.append(data_dict['bboxes'][valid_idx])
                            valid_rf.append(data_dict['custom_features'][valid_idx])
                            valid_text.append(data_dict['text'][valid_idx])
                        data_dict['bboxes'] = valid_bboxes
                        data_dict['custom_features'] = valid_rf
                        data_dict['text'] = valid_text

                    if len(node_cls) != len(data_dict['bboxes']):
                        print(len(data_dict['bboxes']), len(node_cls))

                    assert len(data_dict['bboxes']) == len(data_dict['custom_features']) == len(data_dict['text']) == len(node_cls)
                    node_cls = torch.tensor(node_cls, dtype=torch.long)

                    if len(data_dict['bboxes']) == 0:
                        return 'Empty bboxes'

                    x, edge_index, edge_attr, nn_index, nn_attr, rf_feature, text_patterns, image_feature, image_data = \
                        get_nn_input_from_datadict(data_dict, self.cfg, feature_extractor=self.feature_extractor)

                    assert not np.any(np.isnan(x))
                    assert not np.any(np.isnan(edge_index))
                    assert not np.any(np.isnan(rf_feature))
                    assert not np.any(np.isnan(text_patterns))

                    if image_feature is not None:
                        assert not np.any(np.isnan(image_feature))
                        image_feature = np.array(image_feature)

                    x = torch.tensor(x, dtype=torch.float)
                    edge_index = torch.tensor(edge_index, dtype=torch.long)
                    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                    edge_labels = []
                    row, col = edge_index
                    num_edges = len(row)
                    for i in range(num_edges):
                        node_i = row[i]
                        node_j = col[i]
                        label_i = node_labels[node_i]
                        label_j = node_labels[node_j]
                        if label_i == label_j:
                            edge_labels.append(1)
                        else:
                            edge_labels.append(0)
                    edge_labels = torch.tensor(edge_labels, dtype=torch.long)
                    assert len(edge_labels) == edge_index.shape[1]

                    text_patterns = []
                    for text in data_dict['text']:
                        txt_pattern = get_text_pattern(text=text, return_vector=True)
                        text_patterns.append(txt_pattern)

                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=node_cls, edge_label=edge_labels)
                    data.raw_data = {
                        'bboxes': data_dict['bboxes'],
                        'file_name': self.filename_list[idx],
                        'text_patterns': text_patterns,
                        'image_features': image_feature,
                        'custom_features': data_dict['custom_features']
                    }

                    if self.opt_save_type is not None and 'image' in self.opt_save_type:
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        enc_ret, encimg = cv2.imencode('.jpg', image_data, encode_param)
                        data.raw_data['image_data'] = encimg

                except Exception as ex:
                    print(file_name)
                    traceback.print_exc()
                    raise ex
            else:
                raise Exception('From jsons - Unimplemented yet!')


            # The visualization code will only run in the main process
            if self.show_data and is_main_process and idx % 100 == 0:
                disp = data_dict['image'].copy()
                img_h, img_w, _ = disp.shape
                bboxes = list(data.raw_data['bboxes'])
                cls_colors = {}
                for bbox_idx, bbox in enumerate(bboxes):
                    if len(node_labels) > 0:
                        label = node_labels[bbox_idx]
                        if label in cls_colors:
                            cls_color = cls_colors[label]
                        else:
                            cls_color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
                            cls_colors[label] = cls_color
                        cv2.rectangle(disp, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), cls_color, 1)
                    else:
                        cv2.rectangle(disp, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)

                    if len(node_labels) > 0:
                        label = node_labels[bbox_idx]
                        cv2.putText(disp, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_PLAIN, 0.8,
                                    (0, 0, 0), 1)
                resize_rate = 900 / max(img_h, img_w)
                if resize_rate < 1.0:
                    disp = cv2.resize(disp, (int(img_w * resize_rate), int(img_h * resize_rate)))
                cv2.imshow('Image', disp)

                image_features = data.raw_data['image_features']
                if image_features is not None:
                    disp = visualize_bbox_similarity_heatmap(img_w, img_h, bboxes, image_features)
                    if resize_rate < 1.0:
                        disp = cv2.resize(disp, (int(img_w * resize_rate), int(img_h * resize_rate)))
                    cv2.imshow('Box features', disp)
                cv2.waitKey(10)

            if self.cache_size > 0:
                # Add to cache
                if len(self.data_cache) < self.cache_size:
                    self.data_cache.append(data)
                # Update cache
                else:
                    idx = np.random.randint(0, self.cache_size)
                    self.data_cache[idx] = data

            assert data.x.shape[0] == data.y.shape[0]
            return data


def test_edge_lists_equality():
    # 새로운 bboxes_data를 임의로 생성
    # [x_min, y_min, x_max, y_max] 형식
    np.random.seed(42)  # 재현성을 위해 시드 설정
    num_boxes = np.random.randint(5, 10)  # 5개에서 9개 사이의 박스 생성
    bboxes_data = []
    for _ in range(num_boxes):
        x_min = np.random.randint(0, 500)
        y_min = np.random.randint(0, 500)
        width = np.random.randint(20, 200)
        height = np.random.randint(20, 200)
        x_max = x_min + width
        y_max = y_min + height
        bboxes_data.append([x_min, y_min, x_max, y_max])

    # numpy 배열로 변환
    bboxes_np = np.array(bboxes_data, dtype=np.float32)

    # 테스트에 사용할 rope_max_length / dist_threshold 값
    test_threshold = 50000  # 원본에서 사용된 값

    print("--- 테스트 시작 ---")

    # 1. 원본 함수 _get_nearest_pair_custom 호출
    original_edge_list = _get_nearest_pair_custom(bboxes_np, test_threshold)
    # set으로 변환하여 순서에 상관없이 비교 가능하도록 준비
    original_edge_set = set(original_edge_list)
    print(f"원본 함수 결과 엣지 수: {len(original_edge_set)}")
    # print(f"원본 함수 엣지: {sorted(list(original_edge_set))}") # 디버깅용

    # 2. 개선된 함수 get_edge_by_nn 호출
    # get_edge_by_nn은 튜플을 반환하므로 첫 번째 요소만 가져옴
    improved_edge_list, _ = get_edge_by_directional_nn(bboxes_np, test_threshold, vertical_gap=0.3)
    # set으로 변환하여 순서에 상관없이 비교 가능하도록 준비
    improved_edge_set = set(improved_edge_list)
    print(f"개선된 함수 결과 엣지 수: {len(improved_edge_set)}")
    # print(f"개선된 함수 엣지: {sorted(list(improved_edge_set))}") # 디버깅용

    # 3. 두 결과 비교
    if original_edge_set == improved_edge_set:
        print("\n? 성공: 두 함수의 최종 엣지 리스트가 동일합니다.")
    else:
        print("\n? 실패: 두 함수의 최종 엣지 리스트가 다릅니다.")
        print("\n원본 함수에만 있는 엣지:")
        print(original_edge_set - improved_edge_set)
        print("\n개선된 함수에만 있는 엣지:")
        print(improved_edge_set - original_edge_set)

    print("\n--- 테스트 종료 ---")


def test_relation_features():
    # 테스트 데이터
    bboxes = np.array([
        [12., 10.429993, 583.8452, 829.0302],
        [13.29, 12., 582.37, 827.75],
        [66.5076, 55.77238, 452.2701, 96.67738],
        [66.5076, 104.77287, 540.60504, 145.67787],
        [66.5076, 153.77336, 537.62616, 194.67836],
        [66.5076, 218.73438, 107.234, 238.60039],
        [66.4457, 802.9685, 319.01096, 811.09546],
        [560.7874, 804.0604, 564.7874, 811.2844]
    ])

    edge_index = [
        (3, 4), (4, 6), (5, 7), (0, 2), (0, 5), (1, 6), (1, 3), (4, 5), (5, 6), (0, 1), (0, 7),
        (2, 4), (1, 2), (1, 5), (6, 7), (3, 5), (1, 4), (2, 3), (1, 7)
    ]

    print("bboxes shape:", bboxes.shape)
    print("edge_index length:", len(edge_index))

    # get_relation_feature 호출
    result_relation_feature = get_relation_feature(edge_index, bboxes)
    print("\nget_relation_feature output shape:", np.array(result_relation_feature).shape)

    # get_edge_transform_modified 호출 (bboxes 직접 전달)
    result_edge_transform_modified_torch = get_edge_transform_bbox(bboxes, edge_index)
    print("get_edge_transform_modified output shape:", result_edge_transform_modified_torch.shape)

    # 결과 비교
    result_edge_transform_modified_np = result_edge_transform_modified_torch.numpy()

    try:
        np.testing.assert_allclose(result_relation_feature, result_edge_transform_modified_np, rtol=1e-5, atol=1e-8)
        print("\n테스트 성공: 두 함수의 결과가 허용 오차 범위 내에서 동일합니다.")
    except AssertionError as e:
        print("\n테스트 실패: 두 함수의 결과가 다릅니다.")
        print(e)

        diff = np.abs(result_relation_feature - result_edge_transform_modified_np)
        max_diff = np.max(diff)
        print(f"\n최대 차이: {max_diff}")


def test_graph_post():
    print("Running test_graph_post...")

    # Input values from the prompt
    cell_box = [
        [12.0, 10.42999267578125, 583.84521484375, 829.0302124023438],
        [13.289999961853027, 12.0, 582.3699951171875, 827.75],
        [66.5076, 55.77238, 452.27010000000007, 96.67738],
        [66.5076, 104.77287, 540.60507, 145.67787],
        [66.5076, 153.77336, 537.62619, 194.67836],
        [66.5076, 218.73438, 107.234, 238.60039],
        [66.4457, 802.9685, 319.01097, 811.09548],
        [560.7874, 804.0604, 564.7874, 811.2844]
    ]
    node_pred_tensor = torch.tensor([7, 7, 7, 7, 7, 7, 6, 6], device='cpu')
    pair_cell = np.array([
        [3, 4], [4, 6], [5, 7], [0, 2], [0, 5], [1, 6], [1, 3], [4, 5],
        [5, 6], [0, 1], [0, 7], [2, 4], [1, 2], [1, 5], [6, 7], [3, 5],
        [1, 4], [2, 3], [1, 7]
    ])
    pair_cell_pred_tensor = torch.tensor([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0], device='cpu')
    pair_cell_score_list_tensor = torch.tensor([
        [0.3987, 5.1128], [6.3733, -0.0315], [5.6963, -0.0295], [1.5991, 1.8040],
        [1.8253, 1.4542], [2.8117, -0.0110], [1.4833, 1.9888], [3.2875, 1.7729],
        [5.3414, -0.0219], [0.6873, 2.4419], [3.8597, -0.0256], [1.2155, 4.0719],
        [1.9597, 1.3629], [1.8675, 1.4008], [4.8807, -0.0098], [3.5506, 1.4282],
        [1.2475, 2.2906], [0.7880, 4.5541], [3.9331, -0.0261]
    ], device='cpu')
    label_priority_list = [1, 5, 6, 4, 0, 3, 2, 7, 8, 9, 10, 11]

    # post_process 함수 호출
    results_post_process = post_process(cell_box, node_pred_tensor, pair_cell,
                                        pair_cell_pred_tensor, pair_cell_score_list_tensor,
                                        label_priority_list)

    print("\n--- post_process Results ---")
    for i, od_label in enumerate(results_post_process):
        print(f"Result {i + 1}:")
        print(f"  Label: {od_label['label']}")
        print(f"  Points (min_x, min_y, max_x, max_y): {od_label['points']}")
        print(f"  Node Set: {od_label['nodeSet']}")
        print("-" * 30)

    # group_node_by_edge_with_networkx_and_class_prior 함수를 위한 입력 준비
    num_nodes = len(cell_box)
    node_cls_array = node_pred_tensor.cpu().numpy()
    # node_score는 임의의 값으로 설정 (get_od_label에서는 사용되지 않지만, 함수 인자로 필요)
    # 실제 환경에서는 node_logits에서 파생된 실제 점수를 사용해야 함
    node_score_array = np.ones_like(node_cls_array, dtype=float) * 0.9  # 예시 값

    edge_index = pair_cell.T
    edge_matrix = get_edge_matrix(num_nodes, edge_index, pair_cell_pred_tensor)

    # group_node_by_edge_with_networkx_and_class_prior 함수 호출
    results_group_func = group_node_by_edge_with_networkx_and_class_prior(
        node_cls_array,
        node_score_array,  # 실제 node_score 값으로 대체해야 함
        edge_matrix,
        cell_box,  # bboxes는 cell_box와 동일
        label_priority_list
    )

    print("\n--- group_node_by_edge_with_networkx_and_class_prior Results ---")
    for i, group_data in enumerate(results_group_func):
        print(f"Result {i + 1}:")
        print(f"  Group Class: {group_data['group_class']}")
        print(f"  Group BBox: {group_data['group_bbox']}")
        print(f"  Indices: {group_data['indicies']}")
        print("-" * 30)

    # 두 함수의 결과 비교 (핵심 부분만)
    # 결과를 비교하기 전에 순서가 다를 수 있으므로 정렬이 필요
    def sort_results(res_list):
        # 각 결과 딕셔너리 내의 'nodeSet' 또는 'indicies'를 기준으로 정렬
        # 'nodeSet'은 set이므로 list로 변환 후 정렬하여 비교 가능하도록 함
        sorted_list = sorted(res_list, key=lambda x: sorted(list(x.get('nodeSet', x.get('indicies')))))
        return sorted_list

    sorted_results_post_process = sort_results(results_post_process)
    sorted_results_group_func = sort_results(results_group_func)

    # 길이 비교
    assert len(sorted_results_post_process) == len(sorted_results_group_func), \
        f"Number of groups mismatch: post_process={len(sorted_results_post_process)}, group_func={len(sorted_results_group_func)}"

    # 각 그룹의 핵심 정보 비교
    comparison_successful = True
    for i in range(len(sorted_results_post_process)):
        pp_res = sorted_results_post_process[i]
        gf_res = sorted_results_group_func[i]

        # 노드 집합 비교 (set으로 변환하여 순서에 무관하게 비교)
        if set(pp_res['nodeSet']) != set(gf_res['indicies']):
            print(f"Mismatch in Node Set for group {i}:")
            print(f"  post_process Node Set: {pp_res['nodeSet']}")
            print(f"  group_func Indices: {gf_res['indicies']}")
            comparison_successful = False

        # 레이블 비교
        if pp_res['label'] != gf_res['group_class']:
            print(f"Mismatch in Label for group {i}:")
            print(f"  post_process Label: {pp_res['label']}")
            print(f"  group_func Group Class: {gf_res['group_class']}")
            comparison_successful = False

        # 바운딩 박스 비교 (부동 소수점 오차 고려하여 근사치 비교)
        # np.allclose를 사용하거나, 특정 소수점 자리까지 비교
        if not np.allclose(pp_res['points'], gf_res['group_bbox'], atol=1e-5):  # atol은 허용 오차
            print(f"Mismatch in Bounding Box for group {i}:")
            print(f"  post_process Points: {pp_res['points']}")
            print(f"  group_func Group BBox: {gf_res['group_bbox']}")
            comparison_successful = False

    assert comparison_successful, "Comparison between post_process and group_node_by_edge_with_networkx_and_class_prior failed."

    print("\nTest completed. Both functions produced identical relevant outputs.")

if __name__ == "__main__":
    # test_edge_lists_equality()
    # test_relation_features()
    test_graph_post()
