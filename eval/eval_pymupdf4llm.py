import argparse

import json

import pymupdf4llm._layout
import pymupdf4llm
from eval_util import evaluate_detection, print_eval_result

def det_func(pdf_path):
    json_result = pymupdf4llm.to_json(pdf_path)
    json_data = json.loads(json_result)
    boxes = json_data['pages'][0]['boxes']
    det = []
    for b in boxes:
        det.append([
            b['x0'], b['y0'], b['x1'], b['y1'], b['boxclass']
        ])
    return det


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PDF document detection evaluation script.')
    parser.add_argument('--pdf_dir', type=str, required=True,
                        help='Path to the directory containing PDF files for evaluation')
    parser.add_argument('--result_csv_path', type=str, default=None,
                        help='Optional: Path to save evaluation results as a CSV file. Default is None (results not saved to CSV).')

    args = parser.parse_args()



    ret = evaluate_detection(det_func, pdf_dirs=[args.pdf_dir], result_csv_path=args.result_csv_path)
    print_eval_result(ret)
