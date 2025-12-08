from train.core.model.GCNN import GCNN
from train.core.model.BoxDGCNN import BoxDGCNN
from train.core.model.BoxNNDGCNN import BoxNNDGCNN
from train.core.model.BoxRFDGNN import BoxRFDGCNN
from train.core.model.BoxIMFDGCNN import BoxIMFDGCNN


def get_version():
    return '20251124.0'

def get_model(cfg, data_class_names):
    num_node_classes = len(data_class_names)  # 텍스트, 제목, 이미지, 표 등
    num_edge_classes = 2  # 0: 연결 없음, 1: 연결 있음

    if cfg['model']['type'] == 'GCN':
        model_hidden_dim = cfg['model']['hidden_dim']
        node_feature_dim = cfg['model']['node_feature_dim']
        edge_feature_dim = cfg['model']['edge_feature_dim']

        model = GCNN(node_feature_dim, edge_feature_dim, num_node_classes, num_edge_classes,
                     hidden_dim=model_hidden_dim)
    elif cfg['model']['type'] == 'DGCNN':
        model_hidden_dim = cfg['model']['hidden_dim']
        sample_k = cfg['model']['sample_k']
        dgcn_dim = cfg['model']['dgcn_dim']
        dgcn_aggr = cfg['model']['dgcn_aggr']

        if cfg['model']['name'] == 'BoxDGCNN':
            node_feature_dim = cfg['model']['node_feature_dim']
            edge_feature_dim = cfg['model']['edge_feature_dim']

            model = BoxDGCNN(node_feature_dim, edge_feature_dim, num_node_classes, num_edge_classes,
                             hidden_dim=model_hidden_dim, dgcn_dim=dgcn_dim, k=sample_k, aggr=dgcn_aggr)
        elif cfg['model']['name'] == 'BoxNNDGCNN':
            node_feature_dim = cfg['model']['node_feature_dim']
            edge_feature_dim = cfg['model']['edge_feature_dim']

            nn_feature_dim = cfg['model']['nn_feature_dim']
            model = BoxNNDGCNN(node_feature_dim, edge_feature_dim, nn_feature_dim, num_node_classes, num_edge_classes,
                             hidden_dim=model_hidden_dim, dgcn_dim=dgcn_dim, k=sample_k)

        elif cfg['model']['name'] == 'BoxRFDGCNN':
            rf_names = cfg['data']['rf_names']
            node_input_dim = cfg['model']['node_input_dim']
            node_output_dim = cfg['model']['node_output_dim']
            edge_input_dim = cfg['model']['edge_input_dim']
            rf_output_dim = cfg['model']['rf_output_dim']
            txp_input_dim = cfg['model']['txp_input_dim']
            txp_output_dim = cfg['model']['txp_output_dim']
            imf_input_dim = cfg['model']['imf_input_dim']
            imf_output_dim = cfg['model']['imf_output_dim']
            option = cfg['model']['option']

            if option is None:
                option = []
            model = BoxRFDGCNN(node_input_dim=node_input_dim, node_output_dim=node_output_dim,
                               edge_input_dim=edge_input_dim,
                               rf_input_dim=len(rf_names), rf_output_dim=rf_output_dim,
                               txp_input_dim=txp_input_dim, txp_output_dim=txp_output_dim,
                               imf_input_dim=imf_input_dim, imf_output_dim=imf_output_dim,
                               num_node_classes=num_node_classes, num_edge_classes=num_edge_classes,
                               hidden_dim=model_hidden_dim, dgcn_dim=dgcn_dim, k=sample_k, aggr=dgcn_aggr,
                               option=option)

        elif cfg['model']['name'] == 'BoxIMFDGCNN':
            rf_names = cfg['data']['rf_names']
            node_input_dim = cfg['model']['node_input_dim']
            node_output_dim = cfg['model']['node_output_dim']
            edge_input_dim = cfg['model']['edge_input_dim']
            rf_output_dim = cfg['model']['rf_output_dim']
            txp_input_dim = cfg['model']['txp_input_dim']
            txp_output_dim = cfg['model']['txp_output_dim']
            imf_input_dim = cfg['model']['imf_input_dim']
            imf_output_dim = cfg['model']['imf_output_dim']
            option = cfg['model']['option']

            if option is None:
                option = []
            model = BoxIMFDGCNN(node_input_dim=node_input_dim, node_output_dim=node_output_dim,
                                edge_input_dim=edge_input_dim,
                                rf_input_dim=len(rf_names), rf_output_dim=rf_output_dim,
                                txp_input_dim=txp_input_dim, txp_output_dim=txp_output_dim,
                                imf_input_dim=imf_input_dim, imf_output_dim=imf_output_dim,
                                num_node_classes=num_node_classes, num_edge_classes=num_edge_classes,
                                hidden_dim=model_hidden_dim, dgcn_dim=dgcn_dim, k=sample_k, aggr=dgcn_aggr,
                                option=option)
        else:
            raise Exception(f'Not supported model - %s' % cfg['model']['type'])
    else:
        raise Exception(f'Not supported model - %s' % cfg['model']['type'])

    return model
