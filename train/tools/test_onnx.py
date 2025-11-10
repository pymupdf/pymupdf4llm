
import os
import json
import shutil
import pickle
import time

import blosc

import numpy as np

import cv2
import anyconfig

from collections import defaultdict

from train.core.common.util import get_iou

import pymupdf
import train.infer.DocumentLayoutAnalyzer as DocumentLayoutAnalyzer
from train.infer.pymupdf_util import create_input_data_from_page

def dump_json_data(cfg):
    json_dir = cfg['test_onnx']['json_dir']
    pdf_dir = cfg['test_onnx']['pdf_dir']

    config_path = cfg['test_onnx']['config_path']
    onnx_path = cfg['test_onnx']['onnx_path']

    da = DocumentLayoutAnalyzer.get_model(config_path=config_path, model_path=onnx_path)

    for json_file in os.listdir(json_dir):
        json_path = f'{json_dir}/{json_file}'
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        data_dict = {
            'bboxes': [],
            'text': [],
            'custom_features': []
        }

        for data_idx, data in enumerate(json_data['img_data_list']):
            data_dict['bboxes'].append(data['text_coor'])
            data_dict['text'].append(data['content'])
            data_dict['custom_features'].append(data['custom_features'])

        print(f'{json_path} -----------------------------------')

        det_result = da.predict(data_dict)

        doc = pymupdf.open(f'{pdf_dir}/{json_file[:-5]}.pdf')
        page = doc[0]

        pix = page.get_pixmap()
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = bytes.reshape(pix.height, pix.width, pix.n)
        page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)

        for bbox in data_dict['bboxes']:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 1)

        for bbox in det_result:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            cv2.putText(page_img, bbox[-1], (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
            cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        cv2.imshow('PDF Page', page_img)
        key = cv2.waitKey()
        if key == ord('y'):
            with open('test_input.json', 'w') as f:
                json.dump(data_dict, f, indent=4)
            shutil.copy(f'{pdf_dir}/{json_file[:-5]}.pdf', f'test_input.pdf')


def visualize_result(features_path, pdf_path):
    doc = pymupdf.open(pdf_path)
    da = DocumentLayoutAnalyzer.get_model()

    for page in doc:
        data_dict = create_input_data_from_page(page)
        det_result = da.predict(data_dict)

        pix = page.get_pixmap()
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = bytes.reshape(pix.height, pix.width, pix.n)
        page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)

        for bbox in data_dict['bboxes']:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 1)

        for bbox in det_result:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            cv2.putText(page_img, bbox[-1], (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
            cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        cv2.imshow('PDF Page', page_img)
        key = cv2.waitKey()


def robins_features_extraction(feature_path, pdf_dir, filename, rect_list):
    import tempfile
    import subprocess

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write("PDF,page,x0,y0,x1,y1,class,score,order\n")
            n = 0
            for r in rect_list:
                line = f'{filename},0,{r[0]},{r[1]},{r[2]},{r[3]},0,0,{n}'
                n = n + 1
                tmp.write(line + '\n')
        command = '%s -d "%s" "%s"' % (feature_path, pdf_dir, path)
        result = subprocess.run(command, text=True, capture_output=True, shell=True)
        if result.returncode:
            print('Command failed:\n' + command + '\n')
        lines = result.stdout.splitlines()
        feature_rect_list = []
        feature_header = []

        for line_idx, line in enumerate(lines):
            if line_idx == 0:
                feature_header = line.split(',')
                continue
            features = []
            for x in line.split(','):
                features.append(x)
            feature_rect_list.append(features)
        return feature_header, feature_rect_list
    except Exception as ex:
        print('%s: %s' % (filename, ex))
    finally:
        os.remove(path)
    return None, None


def eval_det_performance(cfg):
    from train.infer.pymupdf_util import create_input_data_by_pymupdf

    pdf_dir = cfg['test_onnx']['pdf_dir']
    pkl_dir = cfg['test_onnx']['pkl_dir']
    config_path = cfg['test_onnx']['config_path']
    model_path = cfg['test_onnx']['model_path']
    imf_model_path = cfg['test_onnx']['imf_model_path']
    features_path = cfg['test_onnx']['features_path']
    iou_threshold = cfg['test_onnx']['iou_threshold']
    show_data = cfg['test_onnx']['show_data']

    model = DocumentLayoutAnalyzer.get_model(
        config_path=config_path,
        model_path=model_path,
        imf_model_path=imf_model_path,
    )

    # Initialize statistics for each class
    stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    all_classes = set()

    total_elapsed_time = []
    fe_elapsed_time = []
    if_elapsed_time = []

    file_list = os.listdir(pkl_dir)
    for file_idx, file_name in enumerate(file_list):
        # if file_idx > 10:
        #     break

        pkl_file = os.path.join(pkl_dir, file_name)
        pdf_file = os.path.join(pdf_dir, file_name.replace('.pkl', '.pdf'))

        with open(pkl_file, 'rb') as f:
            compressed_pickle = f.read()
        decompressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(decompressed_pickle)
        gt_labels = pkl_data['label']  # [x1, y1, x2, y2, class_id, class_label]
        cv_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)
        doc = pymupdf.open(pdf_file)
        page = doc[0]

        pix = page.get_pixmap()
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = bytes.reshape(pix.height, pix.width, pix.n)
        page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)
        doc.close()

        for ant in gt_labels:
            ant[0] *= page_img.shape[1] / cv_img.shape[1]
            ant[1] *= page_img.shape[0] / cv_img.shape[0]
            ant[2] *= page_img.shape[1] / cv_img.shape[1]
            ant[3] *= page_img.shape[0] / cv_img.shape[0]

        st_time = time.time()
        fe_time = time.time()
        data_dict = create_input_data_by_pymupdf(pdf_file, features_path=features_path)
        fe_time = time.time() - fe_time

        infer_time = time.time()
        pred_results = model.predict(data_dict)  # [x1, y1, x2, y2, class_label]
        infer_time = time.time() - infer_time

        total_time = time.time() - st_time
        fe_elapsed_time.append(fe_time)
        if_elapsed_time.append(infer_time)
        total_elapsed_time.append(total_time)

        print(f'[{file_idx}/{len(file_list)}] {file_name[:-4]} ({len(data_dict["bboxes"])} bboxes, '
              f'Total: {total_time:.2f} sec / FE: {fe_time:.2f}, IF: {infer_time:.2f})...')

        # Extract box coordinates and labels
        gt_boxes = [(x1, y1, x2, y2, label) for x1, y1, x2, y2, _, label in gt_labels]
        pred_boxes = [(x1, y1, x2, y2, label) for x1, y1, x2, y2, label in pred_results]

        matched_gt = set()
        matched_pred = set()

        if show_data:
            for bbox in gt_boxes:
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]
                cv2.putText(page_img, bbox[-1], (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)
                cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

            for bbox in data_dict['bboxes']:
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]
                cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 1)

            for bbox in pred_boxes:
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]
                cv2.putText(page_img, bbox[-1], (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
                cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

            cv2.imshow('PDF Page', page_img)
            cv2.waitKey()

        for pred_idx, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt[4] != pred[4] or gt_idx in matched_gt:
                    continue
                iou = get_iou(gt[:4], pred[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                stats[pred[4]]['tp'] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
            else:
                stats[pred[4]]['fp'] += 1

            all_classes.add(pred[4])

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                stats[gt[4]]['fn'] += 1
                all_classes.add(gt[4])

    # Print evaluation results
    print(f"\nDetection Evaluation (IOU threshold = {iou_threshold}):")
    print("{:<20} {:>10} {:>10} {:>10}".format("Class", "Precision", "Recall", "F1"))

    total_tp, total_fp, total_fn = 0, 0, 0

    for cls in sorted(all_classes):
        tp = stats[cls]['tp']
        fp = stats[cls]['fp']
        fn = stats[cls]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{cls:<20} {precision:10.3f} {recall:10.3f} {f1:10.3f}")

        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    print(f"{'Overall':<20} {overall_precision:10.3f} {overall_recall:10.3f} {overall_f1:10.3f}")
    print('Total - %d sec, mean %.2f sec, std %.2f sec' % (np.sum(total_elapsed_time), np.mean(total_elapsed_time), np.std(total_elapsed_time)))
    print('FE - %d sec, mean %.2f sec, std %.2f sec' % (np.sum(fe_elapsed_time), np.mean(fe_elapsed_time), np.std(fe_elapsed_time)))
    print('IF - %d sec, mean %.2f sec, std %.2f sec' % (np.sum(if_elapsed_time), np.mean(if_elapsed_time), np.std(if_elapsed_time)))


def compare_input_data(cfg):
    import onnxruntime as ort

    from train.tools.data.layout.DocumentLMDBDataset import DocumentLMDBDataset
    from train.infer.onnx.BoxRFDGNN import get_nn_input_from_datadict
    from train.infer.pymupdf_util import create_input_data_by_pymupdf

    pdf_dir = cfg['test']['pdf_dir']
    pkl_dir = cfg['test']['pkl_dir']
    data_class_names = cfg['data']['class_list']
    data_class_map = {}
    for i in range(len(data_class_names)):
        data_class_map[data_class_names[i]] = i
    class_priority_list = cfg['data']['class_priority']
    data_rf_names = cfg['data']['rf_names']
    val_lmdb_path = cfg['train']['val_dataset']['lmdb_path']

    val_data = DocumentLMDBDataset(lmdb_path=val_lmdb_path, cache_size=0, readahead=True, keep_raw_data=True,
                                   rf_names=data_rf_names)

    feature_extractor_path = cfg['test_onnx']['imf_model_path']
    feature_extractor = ort.InferenceSession(
        feature_extractor_path, providers=['CPUExecutionProvider']
    )

    matched_count = 0
    not_matched_count = 0
    for data_idx, data in enumerate(val_data):
        if data_idx < 0:
            continue

        print(f"\n--- Comparing data_idx: {data_idx} ---")

        # Data from LMDB (torch.tensor)
        bboxes, edge_index, edge_attr, rf_feature, text_patterns, image_feature = \
            (data.x, data.edge_index, data.edge_attr, data.rf_features, data.text_patterns,
             data.img_features)
        raw_data = data.raw_data
        file_name = raw_data['file_name'][:-5]  # Adjusted slicing for '.pdf'

        # Data from create_pdf_input_data() and get_nn_input_from_datadict() (numpy.ndarray)
        pdf_path = f'{pdf_dir}/{file_name}.pdf'
        data_dict = create_input_data_by_pymupdf(pdf_path, features_path=None)
        bboxes_2, edge_index_2, edge_attr_2, _, _, rf_feature_2, text_patterns_2, image_feature_2 = \
            get_nn_input_from_datadict(data_dict, cfg, feature_extractor=feature_extractor)

        # Convert torch.tensor to numpy.ndarray for comparison
        # .cpu() ensures it's on CPU before converting to numpy
        bboxes_np = bboxes.cpu().numpy()
        edge_index_np = edge_index.cpu().numpy()
        edge_attr_np = edge_attr.cpu().numpy()
        rf_feature_np = rf_feature.cpu().numpy()
        text_patterns_np = text_patterns.cpu().numpy()
        image_feature_np = image_feature.cpu().numpy()

        # disp = data_dict['image'].copy()
        # if len(bboxes_np) != len(bboxes_2):
        #     for bbox in bboxes_np:
        #         x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        #         cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 1)
        #
        #     cv2.imshow('Page', disp)
        #     cv2.waitKey()

        # Define variables to compare
        vars_to_compare = [
            ("bboxes", bboxes_np, bboxes_2),
            ("edge_index", edge_index_np, edge_index_2),
            ("edge_attr", edge_attr_np, edge_attr_2),
            ("rf_feature", rf_feature_np, rf_feature_2),
            ("text_patterns", text_patterns_np, text_patterns_2),
            ("image_feature", image_feature_np, image_feature_2),
        ]

        # Perform comparison
        all_values_match = True
        for name, val1, val2 in vars_to_compare:
            if val1.shape != val2.shape:
                not_matched_count += 1
                continue

            # For integer arrays (like indices), use np.array_equal
            # For floating-point arrays, use np.allclose to account for precision differences
            if name in ["edge_index"]:  # Add other integer arrays here if applicable
                comparison_result = np.array_equal(val1, val2)
                print(f"  Comparison for {name}: {'Match' if comparison_result else 'Mismatch (array_equal)'}")
            else:  # Assumed to be floating-point arrays
                # rtol and atol can be adjusted based on desired precision
                comparison_result = np.allclose(val1, val2, rtol=1e-05, atol=1e-08)
                print(f"  Comparison for {name}: {'Match' if comparison_result else 'Mismatch (allclose)'}")

            if not comparison_result:
                all_values_match = False
                # Optionally print more detailed diff for debugging
                print(f"    - Max absolute difference: {np.max(np.abs(val1 - val2))}")
                print(f"    - Number of differing elements: {np.sum(~np.isclose(val1, val2, rtol=1e-05, atol=1e-08))}")

        if all_values_match:
            matched_count += 1
            print(f"\nAll compared values for data_idx {data_idx} match successfully.")
        else:
            not_matched_count += 1
            print(f"\nThere are mismatches for data_idx {data_idx}. Please check the output above.")

    total_count = matched_count + not_matched_count
    print('Matched: %d / %d (%.4f)' % (matched_count, total_count, matched_count / total_count))


def test_onnx_gpu_infer():
    import onnxruntime as ort
    print(ort.get_available_providers())
    feature_extractor_path = "/media/win/Dataset/DocumentlayoutGNN/directional_nn/imf_thin_seg-all/0.9218_all_172d.onnx"

    feature_extractor = ort.InferenceSession(
        feature_extractor_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    exit(0)

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    with open('tools/config.yaml', "rb") as f:
        cfg = anyconfig.load(f)

    test_onnx_gpu_infer()

    task = cfg['test_onnx']['task']
    if task == 'dump_json_data':
        dump_json_data(cfg)
    elif task == 'visualize_result':
        visualize_result(cfg['test_onnx']['features_path'], cfg['test_onnx']['test_pdf_file'])
    elif task == 'eval_det_performance':
        eval_det_performance(cfg)
    elif task == 'compare_input_data':
        compare_input_data(cfg)
