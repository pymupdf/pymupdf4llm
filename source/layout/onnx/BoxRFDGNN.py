import os.path

import yaml
import numpy as np
import onnxruntime as ort

from pathlib import Path

from ..common_util import (get_boxes_transform, get_edge_by_directional_nn, get_edge_by_alignment, get_edge_by_knn,
                           get_edge_transform_bbox, get_edge_transform_bbox2,
                           get_text_pattern, get_edge_matrix, group_node_by_edge_with_networkx_and_class_prior,
                           extract_bbox_features_by_roi_pooling, extract_bbox_features_by_roi_align, resize_image, to_gray,
                           extract_bboxes_from_segmentation)
from ..pymupdf_util import create_input_data_from_page


def get_rf_features(custom_feature, rf_names):
    f = []
    for f_name in rf_names:
        if f_name == 'is_text':
            if 'box_type' not in custom_feature or custom_feature['box_type'] == 'text':
                f.append(1.0)
            else:
                f.append(0.0)
        elif f_name == 'is_image':
            if 'box_type' in custom_feature and custom_feature['box_type'] == 'image':
                f.append(1.0)
            else:
                f.append(0.0)

        elif f_name == 'is_vector':
            if 'box_type' in custom_feature and custom_feature['box_type'] == 'vector':
                f.append(1.0)
            else:
                f.append(0.0)

        elif f_name == 'is_hline_vector':
            if 'box_type' in custom_feature and custom_feature['box_type'] == 'h-line-vector':
                f.append(1.0)
            else:
                f.append(0.0)

        elif f_name == 'is_vline_vector':
            if 'box_type' in custom_feature and custom_feature['box_type'] == 'v-line-vector':
                f.append(1.0)
            else:
                f.append(0.0)
        else:
            if f_name not in custom_feature:
                raise Exception(f"'{f_name}' does not exist in 'custom_features'")

            try:
                f.append(float(custom_feature[f_name]))
            except Exception as ex:
                print(f'{f_name} : {str(ex)}')
                raise ex
    return f


def get_nn_input_from_datadict(data_dict, cfg, return_nn_index=False,
                               edge_sampling='4D', save_crop_page=False):
    original_bboxes = data_dict['bboxes']
    x = get_boxes_transform(original_bboxes)

    bboxes = np.array(original_bboxes, dtype=np.float32)

    data_option = cfg['data']
    model_option = cfg['model']

    if return_nn_index:
        nn_k = model_option['sample_k']
        nn_index = get_edge_by_knn(bboxes, k=nn_k)
        nn_attr = get_edge_transform_bbox(bboxes, nn_index)
        nn_index = np.array(nn_index).T
    else:
        nn_index = None
        nn_attr = None

    edge_type = ''
    if 'edge_type' in model_option['option']:
        edge_type = model_option['option']['edge_type']

    # One input
    if len(bboxes) == 1:
        dummy_bboxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0]
        ], dtype=np.float32)
        dummy_edge_index = [[0, 1], [1, 0]]
        if edge_type == 'SEO':
            dummy_edge_attr = get_edge_transform_bbox2(dummy_bboxes, dummy_edge_index)
        else:
            dummy_edge_attr = get_edge_transform_bbox(dummy_bboxes, dummy_edge_index)

        edge_dim = dummy_edge_attr.shape[1]
        edge_index = np.zeros(shape=[2, 1], dtype=np.int64)
        edge_attr = np.zeros(shape=[1, edge_dim], dtype=np.float32)
        nn_index  = np.zeros(shape=[2, 1], dtype=np.int64)
        nn_attr = np.zeros(shape=[1, edge_dim], dtype=np.float32)
    else:
        if edge_sampling == '4D+AL':
            edge_index1, _ = get_edge_by_directional_nn(bboxes, 50000, vertical_gap=0.3)
            edge_index2 = get_edge_by_alignment(bboxes, dist_threshold=0)
            edge_index = list(set(edge_index1 + edge_index2))
            edge_index = sorted(edge_index)
        else:
            edge_index, _ = get_edge_by_directional_nn(bboxes, 50000, vertical_gap=0.3)

        if edge_type == 'SEO':
            edge_attr = get_edge_transform_bbox2(bboxes, edge_index)
        else:
            edge_attr = get_edge_transform_bbox(bboxes, edge_index)
        edge_index = np.array(edge_index).T

    rf_names = data_option['rf_names']
    rf_feature = []
    for row_idx, custom_feature in enumerate(data_dict['custom_features']):
        f = get_rf_features(custom_feature, rf_names)
        rf_feature.append(f)
    rf_feature = np.array(rf_feature, dtype=np.float32)

    text_feature = []
    for text in data_dict['text']:
        text_feature.append(get_text_pattern(text, return_vector=True))
    text_feature = np.array(text_feature, dtype=np.float32)

    # Cropped page image by bboxes
    crop_img = None
    if save_crop_page:
        page_img = data_dict['image']
        x1_min = int(np.floor(np.min(bboxes[:, 0])))
        y1_min = int(np.floor(np.min(bboxes[:, 1])))
        x2_max = int(np.ceil(np.max(bboxes[:, 2])))
        y2_max = int(np.ceil(np.max(bboxes[:, 3])))

        h, w = page_img.shape[:2]
        x1_min = max(0, x1_min)
        y1_min = max(0, y1_min)
        x2_max = min(w, x2_max)
        y2_max = min(h, y2_max)

        crop_img = page_img[y1_min:y2_max, x1_min:x2_max]
        crop_img = resize_image(crop_img, (500, 500))

    feature_map = None
    if 'feature_map' in data_dict:
        feature_map = data_dict['feature_map']

    # Image feature extractor
    if feature_map is not None:
        page_img = data_dict['image']
        page_h, page_w, _ = page_img.shape

        apply_softmax = False
        add_uncertainty = False

        image_features = extract_bbox_features_by_roi_align(feature_map, original_bboxes, page_w, page_h,
                                                            apply_softmax=apply_softmax, concat_mean_max=True,
                                                            add_uncertainty=add_uncertainty)

        edge_bboxes = []
        row, col = edge_index
        for node_idx_1, node_idx_2 in zip(row, col):
            bbox1 = original_bboxes[node_idx_1]
            bbox2 = original_bboxes[node_idx_2]

            x1 = min(bbox1[0], bbox2[0])
            y1 = min(bbox1[1], bbox2[1])
            x2 = max(bbox1[2], bbox2[2])
            y2 = max(bbox1[3], bbox2[3])
            edge_bboxes.append([x1, y1, x2, y2])

        edge_features = extract_bbox_features_by_roi_align(feature_map, edge_bboxes, page_w, page_h,
                                                           apply_softmax=apply_softmax, concat_mean_max=True,
                                                           add_uncertainty=add_uncertainty)
        edge_attr = np.concatenate([edge_attr, edge_features], axis=1)
    else:
        image_features = None

    return x, edge_index, edge_attr, nn_index, nn_attr, rf_feature, text_feature, image_features, crop_img



class BoxRFDGNN:
    def __init__(self, config_path, model_path, imf_model_path, feature_set_name='imf+rf'):
        script_dir = Path(__file__).resolve().parent.parent

        self.feature_set_name = feature_set_name
        if self.feature_set_name not in ['imf', 'imf+rf']:
            raise "feature_set_name must be one of 'imf', 'imf+rf'"

        if config_path is None or model_path is None:
            if self.feature_set_name == 'imf':
                config_path = f'{script_dir}/resources/onnx/layout_imf1.1.yaml'
                model_path = f'{script_dir}/resources/onnx/layout_imf1.1.onnx'
            elif self.feature_set_name == 'imf+rf':
                config_path = f'{script_dir}/resources/onnx/layout_imf1.1+rf2.1.yaml'
                model_path = f'{script_dir}/resources/onnx/layout_imf1.1+rf2.1.onnx'
        if imf_model_path is None:
            imf_model_path = f'{script_dir}/resources/onnx/feature_imf1.1.onnx'

        self.config_path = config_path
        with open(self.config_path, "rb"
                  ) as f:
            self.cfg = yaml.safe_load(f)

        self.data_class_names = self.cfg['data']['class_list']
        self.data_class_map = {}
        for i in range(len(self.data_class_names)):
            self.data_class_map[self.data_class_names[i]] = i
        self.class_priority_list = self.cfg['data']['class_priority']

        self.model_path = model_path
        self.session = None
        self.load_onnx_model(self.model_path)

        if os.path.exists(imf_model_path):
            self.feature_extractor = ort.InferenceSession(imf_model_path, providers=['CPUExecutionProvider'])
        else:
            self.feature_extractor = None

    def load_onnx_model(self, model_path):
        ort.set_default_logger_severity(3)
        self.session = ort.InferenceSession(model_path)


    def predict(self, page, input_type=('text',), verbose=False):
        import numpy as np

        data_dict = create_input_data_from_page(page, input_type=input_type,
                                                feature_set_name=self.feature_set_name,
                                                feature_extractor=self.feature_extractor)
        bboxes = np.array(data_dict['bboxes'], dtype=np.float32)

        # Empty input
        if len(bboxes) == 0:
            return []

        model_type = self.cfg['model']['option']['conv_type']

        # Print model type when verbose is enabled
        if verbose:
            print(">>> model_type:", model_type)

        input_names = ["x", "edge_index", "edge_attr", "rf_features", "text_patterns", "image_features", "k", "batch"]
        if model_type in ['GAT', 'NNConv']:
            onnx_input_names = ["x", "edge_index", "edge_attr", "rf_features", "text_patterns"]
        elif model_type in ['CustomDGC']:
            onnx_input_names = ["x", "edge_index", "edge_attr", "text_patterns", "image_features", 'k', 'batch']
            if 'rf' in self.cfg['model']['option']['feature_types']:
                onnx_input_names.append('rf_features')
        else:
            raise Exception(f'Not supported model_type = {model_type}!')

        # Prepare neural network inputs from data_dict.
        # get_nn_input_from_datadict should return tensors/arrays in numpy-compatible formats.
        x, edge_index, edge_attr, nn_index, nn_attr, rf_feature, text_feature, image_feature, image_data = \
            get_nn_input_from_datadict(data_dict, self.cfg, return_nn_index=('nn_index' in onnx_input_names))

        # Build input data dict with explicit dtypes that are commonly expected by ONNX models.
        input_data_dict = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'rf_features': rf_feature,
            # Ensure k is an int64 scalar/array as many ONNX models expect integer type
            'k': np.array(min(len(bboxes), 20), dtype=np.int64),
            'text_patterns': text_feature,
            'image_features': image_feature,
            # Use int64 for batch indices for ONNX compatibility
            'batch': np.zeros(len(bboxes), dtype=np.int64)
        }

        # Build the actual dict of inputs the ONNX session will receive
        onnx_inputs = {}
        for input_name in onnx_input_names:
            onnx_inputs[input_name] = input_data_dict[input_name]

        # Verbose: print shapes and dtypes of onnx_inputs
        if verbose:
            print(">>> onnx_inputs:")
            for name, val in onnx_inputs.items():
                arr = np.asarray(val)
                try:
                    dtype = arr.dtype
                except Exception:
                    # fallback if the object doesn't expose dtype
                    dtype = type(val)
                print(f"  - {name}: shape={np.shape(arr)}, dtype={dtype}")

        # Verbose: print ONNX session input metadata if session exists
        if verbose and hasattr(self, 'session') and self.session is not None:
            try:
                print(">>> ONNX Runtime session inputs metadata:")
                for inp in self.session.get_inputs():
                    print(f"  - name={inp.name}, shape={inp.shape}, type={inp.type}")
                print(">>> ONNX Runtime session outputs metadata:")
                for out in self.session.get_outputs():
                    print(f"  - name={out.name}, shape={out.shape}, type={out.type}")
            except Exception as e:
                print("  (Failed to read session metadata):", e)

        # Run the ONNX model
        ort_outputs = self.session.run(None, onnx_inputs)
        onnx_node_logits, onnx_edge_logits = ort_outputs

        # Convert node logits to probabilities by applying softmax
        exp_node_logits = np.exp(onnx_node_logits - np.max(onnx_node_logits, axis=1, keepdims=True))
        node_probs = exp_node_logits / np.sum(exp_node_logits, axis=1, keepdims=True)

        # Predicted node labels and scores
        predicted_node_label = np.argmax(node_probs, axis=1)
        predicted_node_score = node_probs[np.arange(node_probs.shape[0]), predicted_node_label]

        # Edge prediction (assuming binary classification and using threshold on positive class)
        edge_threshold = 0.55
        if onnx_edge_logits.size > 0:
            exp_edge_logits = np.exp(onnx_edge_logits - np.max(onnx_edge_logits, axis=1, keepdims=True))
            edge_probs = exp_edge_logits / np.sum(exp_edge_logits, axis=1, keepdims=True)
            # Consider the second column (index 1) as positive-class probability
            predicted_edge_labels = (edge_probs[:, 1] > edge_threshold).astype(int)
        else:
            predicted_edge_labels = np.empty(0, dtype=np.int64)

        det_result = []
        num_nodes = len(predicted_node_label)
        edge_matrix = get_edge_matrix(num_nodes, edge_index, predicted_edge_labels)
        groups = group_node_by_edge_with_networkx_and_class_prior(predicted_node_label, predicted_node_score,
                                                                  edge_matrix,
                                                                  bboxes, self.class_priority_list)

        for group_idx, group in enumerate(groups):
            g_bbox = group['group_bbox']
            cls_name = self.data_class_names[group['group_class']]
            g_bbox.append(cls_name)
            det_result.append(g_bbox)

        return det_result
