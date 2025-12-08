
import torch
import anyconfig
import numpy as np

import onnxruntime as ort

from train.core.model.ModelFactory import get_model
from train.core.common.model_util import load_model_and_optimizer

from train.infer.common_util import get_edge_matrix, group_node_by_edge_with_networkx_and_class_prior
from train.infer.onnx.BoxRFDGNN import get_nn_input_from_datadict


class TorchDetModelWrapper:
    def __init__(self, config_path, model_path, feature_extractor_path, device):
        self.config_path = config_path
        self.model_path = model_path
        self.device = device

        with open(config_path, "rb") as f:
            self.model_config = anyconfig.load(f)
        self.data_class_names = self.model_config['data']['class_list']
        self.class_priority_list = self.model_config['data']['class_priority']

        self.model = get_model(self.model_config, self.data_class_names)
        self.model.to(device)

        load_model_and_optimizer(self.model, model_path)
        self.model.eval()
        self.feature_extractor = ort.InferenceSession(
            feature_extractor_path, providers=['CPUExecutionProvider']
        )

    def predict(self, data_dict):
        x, edge_index, edge_attr, nn_index, nn_attr, rf_feature, text_feature, image_features = \
            get_nn_input_from_datadict(data_dict, self.model_config, feature_extractor=self.feature_extractor)

        bboxes = np.array(data_dict['bboxes'], dtype=np.float32)
        k = min(len(bboxes), 20)

        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(self.device)
        rf_feature = torch.tensor(rf_feature, dtype=torch.float32).to(self.device)
        text_feature = torch.tensor(text_feature, dtype=torch.float32).to(self.device)
        image_features = torch.tensor(image_features, dtype=torch.float32).to(self.device)

        node_logits, edge_logits = self.model.forward(x, edge_index, edge_attr, rf_feature, text_feature, image_features,
                                                      k, None)
        node_probs = torch.softmax(node_logits, dim=1)
        predicted_node_label = torch.argmax(node_probs, dim=1).cpu().numpy()
        predicted_node_score = node_probs[torch.arange(node_probs.size(0)), predicted_node_label]

        edge_threshold = 0.55
        if edge_logits.numel() > 0:
            edge_probs = torch.softmax(edge_logits, dim=1)
            predicted_edge_labels = (edge_probs[:, 1] > edge_threshold).cpu().numpy().astype(int)
        else:
            predicted_edge_labels = torch.empty(0, dtype=torch.long).numpy()  # 엣지가 없으면 빈 텐서

        num_nodes = len(predicted_node_label)
        edge_matrix = get_edge_matrix(num_nodes, edge_index, predicted_edge_labels)
        groups = group_node_by_edge_with_networkx_and_class_prior(predicted_node_label, predicted_node_score,
                                                                  edge_matrix, bboxes, self.class_priority_list)
        pred_results = []
        for group_idx, group in enumerate(groups):
            g_bbox = group['group_bbox']
            cls_name = self.data_class_names[group['group_class']]
            if cls_name == 'unlabelled':
                continue
            g_bbox.append(cls_name)
            pred_results.append(g_bbox)

        return pred_results
