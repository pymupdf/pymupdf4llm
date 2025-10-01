from . import onnx

def get_model(model_name='BoxRFDGNN', feature_set_name='imf',
              config_path=None, model_path=None, imf_model_path=None):

    if model_name == 'BoxRFDGNN':
        return onnx.BoxRFDGNN(config_path, model_path, imf_model_path, feature_set_name=feature_set_name)
    else:
        raise Exception(f'Invalid model name = {model_name}')
