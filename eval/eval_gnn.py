import argparse

import pymupdf

import train.infer.DocumentLayoutAnalyzer as DocumentLayoutAnalyzer
from train.infer.pymupdf_util import create_input_data_from_page

from eval_util import evaluate_detection, print_eval_result


da = None

def det_func(pdf_path):
    doc = pymupdf.open(pdf_path)
    data_dict = create_input_data_from_page(doc[0])
    det_result = da.predict(data_dict)
    return det_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN detection model evaluation script.')
    parser.add_argument('--pdf_dir', type=str, required=True,
                        help='Path to the directory containing PDF files for evaluation')
    parser.add_argument('--result_csv_path', type=str, default=None,
                        help='Optional: Path to save evaluation results as a CSV file. Default is None (results not saved to CSV).')
    parser.add_argument('--model_name', type=str, default='BoxRFDGNN',
                        help='Model name to use. Default is BoxRFDGNN.')
    parser.add_argument('--feature_set_name', type=str, default='imf+rf',
                        help='Feature set name to use. Default is imf+rf.')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Optional: Path to model configuration file. Default is None.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Optional: Path to trained model onnx file. Default is None.')
    parser.add_argument('--imf_model_path', type=str, default=None,
                        help='Optional: Path to image featur extractor onnx model file. Default is None.')

    args = parser.parse_args()

    da = DocumentLayoutAnalyzer.get_model(
        model_name=args.model_name,
        feature_set_name=args.feature_set_name,
        config_path=args.config_path,
        model_path=args.model_path,
        imf_model_path=args.imf_model_path
    )
    ret = evaluate_detection(det_func, pdf_dirs=[args.pdf_dir], result_csv_path=args.result_csv_path)
    print_eval_result(ret)
