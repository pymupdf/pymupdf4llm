import os
import sys

import shutil
import time
import cv2
import blosc
import pickle
import torch
import numpy as np

import fitz
import pymupdf
import anyconfig
import threading
import traceback
import multiprocessing as mp

import onnxruntime as ort

from tqdm import tqdm

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from train.core.model.ModelFactory import get_model
from train.core.common.util import get_iou
from train.core.common.model_util import calculate_f1_score, load_model_and_optimizer

from train.infer.common_util import get_edge_matrix, group_node_by_edge_with_networkx_and_class_prior

from train.tools.data.layout.DocumentLMDBDataset import DocumentLMDBDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

global_lock = None


def eval_det_model_task(gt_node_label, predicted_node_label, predicted_node_score,
                        gt_edge_label, predicted_edge_labels, edge_index, input_bboxes, data_class_names,
                        pdf_dir, pkl_dir, file_name, result_map, exclude_class, label_priority_list,
                        vis_pdf_dir):
    try:
        det_result = []
        if len(gt_node_label) > 1:
            num_nodes = len(predicted_node_label)
            node_f1 = calculate_f1_score(gt_node_label, predicted_node_label)
            edge_f1 = calculate_f1_score(gt_edge_label, predicted_edge_labels)

            edge_matrix = get_edge_matrix(num_nodes, edge_index, predicted_edge_labels)
            # groups = group_node_by_edge_with_networkx(predicted_node_label, predicted_node_score, edge_matrix, input_bboxes)
            groups = group_node_by_edge_with_networkx_and_class_prior(predicted_node_label, predicted_node_score, edge_matrix,
                                                                      input_bboxes, label_priority_list)
            for group_idx, group in enumerate(groups):
                g_bbox = group['group_bbox']
                cls_name = data_class_names[group['group_class']]
                if cls_name == 'unlabelled':
                    continue
                g_bbox.append(cls_name)
                det_result.append(g_bbox)
        file_name = os.path.splitext(file_name)[0]
        doc = pymupdf.open(f'{pdf_dir}/{file_name}.pdf')
        page = doc[0]

        pix = page.get_pixmap()
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = bytes.reshape(pix.height, pix.width, pix.n)
        page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)

        pkl_file = f'{pkl_dir}/{file_name}.pkl'
        with open(pkl_file, 'rb') as f:
            compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(depressed_pickle)
        cv_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)

        to_pdf_resize_x = page.rect[2] / cv_img.shape[1]
        to_pdf_resize_y = page.rect[3] / cv_img.shape[0]
        label = pkl_data['label']
        for ant in label:
            # Rescale GT to PDF page
            ant[0] *= to_pdf_resize_x
            ant[1] *= to_pdf_resize_y
            ant[2] *= to_pdf_resize_x
            ant[3] *= to_pdf_resize_y
            if ant[-1] in exclude_class:
                continue
            result_map['%s-C' % ant[-1].lower()] += 1
            result_map['total-C'] += 1

        # Add Ground-Truth
        gt_layer = doc.add_ocg("GT", on=0)
        for ant in label:
            x1 = ant[0]
            y1 = ant[1]
            x2 = ant[2]
            y2 = ant[3]
            if vis_pdf_dir is not None:
                rect = fitz.Rect(x1, y1, x2, y2)
                page.draw_rect(rect, color=pymupdf.pdfcolor["blue"], width=0.1, oc=gt_layer)
                page.insert_text((x1, y1 - 5), ant[-1], fontsize=8, color=pymupdf.pdfcolor["blue"],
                                 oc=gt_layer)
            # cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

        def get_max_iou(dt_label, gt_labels, multi_class=True):
            boxA = dt_label[:4]
            boxA_cls = dt_label[4].lower()
            max_iou = 0
            max_idx = -1
            for gt_idx, gt_label in enumerate(gt_labels):
                boxB = gt_label[:4]
                boxB_cls = gt_label[5].lower()
                if multi_class and boxA_cls != boxB_cls:
                    continue
                iou_val = get_iou(boxA, boxB)
                if iou_val > max_iou:
                    max_iou = iou_val
                    max_idx = gt_idx
            return max_idx, max_iou

        # Add Node Result
        node_layer = doc.add_ocg("Node", on=0)
        for bbox in input_bboxes:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            if vis_pdf_dir is not None:
                rect = fitz.Rect(x1, y1, x2, y2)
                page.draw_rect(rect, color=pymupdf.pdfcolor["gray"], width=0.1, oc=node_layer)
                # cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 1)

        # Add Detection Result
        error_count = 0

        gt_labels = label[:]
        dt_layer = doc.add_ocg("DT", on=1)
        for ant_idx, ant in enumerate(det_result):
            x1 = round(ant[0], 0)
            y1 = round(ant[1], 0)
            x2 = round(ant[2], 0)
            y2 = round(ant[3], 0)

            # cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            cls_val = ant[-1].lower()
            if cls_val in exclude_class:
                continue

            max_idx, max_iou = get_max_iou(ant, gt_labels, multi_class=False)
            gt_cls_val = ''
            if max_idx >= 0:
                gt_cls_val = gt_labels[max_idx][-1].lower()

            if gt_cls_val != cls_val and max_iou >= 0.6:
                tie_class = groups[ant_idx]['tie_class']
                if tie_class >= 0 and max_iou >= 0.6:
                    result_map['tied_class_error'].append({
                        'cls_val': cls_val,
                        'gt_cls_cal': gt_cls_val,
                        'max_iou': max_iou,
                    })

            txt = '%s (%.4f)' % (ant[4], max_iou)
            if gt_cls_val == cls_val and max_iou >= 0.6:
                del gt_labels[max_idx]
                if vis_pdf_dir is not None:
                    rect = fitz.Rect(x1, y1, x2, y2)
                    page.draw_rect(rect, color=pymupdf.pdfcolor["green"], width=0.1, oc=dt_layer)
                    page.insert_text((x1, y1 - 5), txt, fontsize=8, color=pymupdf.pdfcolor["green"],
                                     oc=dt_layer)
                with global_lock:
                    result_map['total-T'] += 1
                    result_map['%s-T' % cls_val] += 1
            else:
                # if ant[4].lower() in ['table', 'picture']:
                #     error_count += 1
                error_count += 1
                if vis_pdf_dir is not None:
                    rect = fitz.Rect(x1, y1, x2, y2)
                    page.draw_rect(rect, color=pymupdf.pdfcolor["red"], width=0.1, oc=dt_layer)
                    page.insert_text((x1, y1 - 5), txt, fontsize=8, color=pymupdf.pdfcolor["red"],
                                     oc=dt_layer)
                with global_lock:
                    result_map['total-F'] += 1
                    result_map['%s-F' % cls_val] += 1

        if '' == 'D':
            cv2.imshow('Page', page_img)
            cv2.waitKey()

        if vis_pdf_dir is not None and error_count > 0:
            doc.save(f'{vis_pdf_dir}/{file_name[:-5]}.pdf')
        doc.close()

    except Exception as ex:
        traceback.print_exc()


def eval_det_model(cfg, load_from=None, verbose=True):
    global global_lock
    global_lock = threading.Lock()

    exclude_class = []

    pdf_dir = cfg['test']['pdf_dir']
    pkl_dir = cfg['test']['pkl_dir']
    vis_pdf_dir = cfg['test']['vis_pdf_dir']
    device = cfg['test']['device']

    max_thread_num = 40
    val_batch_size = cfg['test']['eval_batch_size']

    if load_from is None:
        load_from = cfg['test']['load_from']

    checkpoint_name = load_from.split('/')[-1]
    model_cfg = load_from.replace(checkpoint_name, 'model.yaml')
    with open(model_cfg, "rb") as f:
        model_cfg = anyconfig.load(f)

    data_class_names = cfg['data']['class_list']
    data_class_map = {}
    for i in range(len(data_class_names)):
        data_class_map[data_class_names[i]] = i
    class_priority_list = cfg['data']['class_priority']
    data_rf_names = cfg['data']['rf_names']

    result_map = {
        'total-T': 0,
        'total-F': 0,
        'total-C': 0,
        'tied_class_error': []
    }
    for cat_name in data_class_names:
        cat_name = cat_name.lower()
        result_map['%s-T' % cat_name] = 0
        result_map['%s-F' % cat_name] = 0
        result_map['%s-C' % cat_name] = 0

    val_lmdb_path = cfg['train']['val_dataset']['lmdb_path']
    val_data = DocumentLMDBDataset(lmdb_path=val_lmdb_path, cache_size=0, readahead=True, keep_raw_data=True,
                                   rf_names=data_rf_names)

    val_dataloader = GeometricDataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=0)

    # 모델 인스턴스 생성
    model = get_model(model_cfg, data_class_names)
    model = model.to(device)

    # 모델 로드
    if not os.path.exists(load_from):
        print(f"Error: Model checkpoint not found at '{load_from}'. Aborting inference.")
        return [], []

    print(f"Loading model from {load_from} for inference...")
    load_model_and_optimizer(model, load_from)
    model.eval()  # 추론 모드로 전환)

    # Verification
    # _, _, _, node_f1, edge_f1 = evaluate_model(model, val_dataloader, None, None, device)
    # val_dataloader = GeometricDataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=0)
    # print('Node-F1: %.4f, Edge-F1: %.4f' % (node_f1, edge_f1))

    if vis_pdf_dir is not None:
        shutil.rmtree(vis_pdf_dir, ignore_errors=True)
        os.makedirs(vis_pdf_dir)

    elapsed_time = []
    futures = []
    print("Starting inference...")
    with ThreadPoolExecutor(max_workers=max_thread_num) as executor:
        with tqdm(total=len(val_dataloader), desc="Processing") as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    batch = batch.to(device)
                    batch_node_logits, batch_edge_logits = model.forward_with_batch(batch)

                    batch_list = batch.to_data_list()
                    batch_indices = batch.batch
                    node_counts = torch.bincount(batch_indices)
                    split_node_logits = torch.split(batch_node_logits, node_counts.tolist())

                    edge_counts = [data.num_edges for data in batch_list]
                    split_edge_logits = torch.split(batch_edge_logits, edge_counts)

                    for sample_idx in range(len(batch_list)):
                        data = batch_list[sample_idx]
                        node_logits = split_node_logits[sample_idx]
                        edge_logits = split_edge_logits[sample_idx]

                        raw_data = data.raw_data
                        input_bboxes = raw_data['bboxes']
                        file_name = raw_data['file_name']

                        gt_node_label = data.y.cpu().numpy()
                        gt_edge_label = data.edge_label.cpu().numpy()
                        edge_index = data.edge_index.cpu().numpy()

                        # print('gt_node_label ---------------')
                        # print(gt_node_label)
                        # print('gt_edge_label ---------------')
                        # print(gt_edge_label)

                        # 결과 CPU로 이동 및 예측값 변환
                        node_probs = torch.softmax(node_logits, dim=1)
                        predicted_node_label = torch.argmax(node_probs, dim=1).cpu().numpy()
                        predicted_node_score = node_probs[torch.arange(node_probs.size(0)), predicted_node_label]

                        # 엣지 예측 (엣지 로짓을 확률로 변환하고 임계값 0.5 이상이면 연결로 판단)
                        # 엣지가 없는 경우 edge_logits가 비어있을 수 있으므로 확인
                        edge_threshold = 0.55
                        if edge_logits.numel() > 0:
                            edge_probs = torch.softmax(edge_logits, dim=1)
                            predicted_edge_labels = (edge_probs[:, 1] > edge_threshold).cpu().numpy().astype(int)
                        else:
                            predicted_edge_labels = torch.empty(0, dtype=torch.long).numpy()  # 엣지가 없으면 빈 텐서

                        # eval_det_model_task(gt_node_label, predicted_node_label, predicted_node_score, gt_edge_label,
                        #                     predicted_edge_labels, edge_index, input_bboxes, data_class_names,
                        #                     pdf_dir, pkl_dir, file_name, result_map, exclude_class, class_priority_list,
                        #                     vis_pdf_dir)
                        future = executor.submit(eval_det_model_task,
                                                 gt_node_label, predicted_node_label, predicted_node_score,
                                                 gt_edge_label, predicted_edge_labels,
                                                 edge_index, input_bboxes, data_class_names,
                                                 pdf_dir, pkl_dir, file_name, result_map, exclude_class, class_priority_list,
                                                 vis_pdf_dir)
                        # futures.append(future)

                    pbar.update(1)

        for future in futures:
            future.result()
        cat_names = data_class_names[:]
        cat_names.append('total')

        total_f1 = 0.0
        table_f1 = 0.0
        eval_result_str = ''
        for cat_name in cat_names:
            cat_name = cat_name.lower()
            t_val = result_map['%s-T' % cat_name]
            f_val = result_map['%s-F' % cat_name]
            c_val = result_map['%s-C' % cat_name]
            if t_val + f_val == 0:
                continue
            pr_val = t_val / (t_val + f_val)
            rc_val = t_val / c_val
            f1_val = 0.0
            if pr_val + rc_val > 0:
                f1_val = 2 * (pr_val * rc_val) / (pr_val + rc_val)
            if verbose:
                eval_result_str += '%s,%.4f,%.4f,%.4f,%d,%d,%d\n' % (cat_name, pr_val, rc_val, f1_val, t_val, f_val, c_val)
            if cat_name == 'total':
                total_f1 = f1_val
            elif cat_name == 'table':
                table_f1 = f1_val

        print(eval_result_str)
        if os.path.exists('temp/eval'):
            checkpoint_name = checkpoint_name.replace('0.0000_', '%.4f (%.4f)_' % (total_f1, table_f1))
            checkpoint_name = checkpoint_name.replace('.pt', '.csv')
            with open(os.path.join('temp/eval', checkpoint_name), 'w') as f:
                f.write(eval_result_str)
        return total_f1, table_f1


def run_eval_det_model(cfg, load_from, verbose, queue, eval_type):
    try:
        if eval_type == 'pdf':
            result = eval_det_model_with_pdf(cfg=cfg, load_from=load_from, verbose=False)
        else:
            result = eval_det_model(cfg=cfg, load_from=load_from, verbose=verbose)
        queue.put(("success", result))
    except Exception as e:
        error_trace = traceback.format_exc()
        queue.put(("error", error_trace))


def run_eval_det_with_retry(cfg, load_from, verbose, max_retries=3, wait_sec=3,
                            eval_type='val'):
    for attempt in range(1, max_retries + 1):
        queue = mp.Queue()
        proc = mp.Process(
            target=run_eval_det_model,
            args=(cfg, load_from, verbose, queue, eval_type)
        )
        proc.start()
        proc.join()

        if not queue.empty():
            status, payload = queue.get()
            if status == "success":
                print(f"[INFO] eval_det_model 성공 (시도 {attempt})")
                return payload
            else:
                print(f"[WARN] eval_det_model 예외 발생 (시도 {attempt}):\n{payload}")
        else:
            print(f"[ERROR] 프로세스가 종료됐지만 queue가 비어있음 (시도 {attempt})")

        time.sleep(wait_sec)

    raise RuntimeError(f"eval_det_model 실패: 최대 {max_retries}회 시도 후에도 실패")

def eval_loop(cfg):
    import glob

    target_dirs = cfg['test']['eval_loop_dir']
    eval_type = cfg['test']['eval_type']
    eval_score_thr = cfg['test']['eval_score_thr']
    loop_sleep_time = cfg['test']['sleep_time']

    do_rename = True
    pkl_dir = cfg['test']['pkl_dir']
    pdf_dir = cfg['test']['pdf_dir']

    if 'test_onnx' in cfg:
        cache_dir = cfg['test_onnx']['cache_dir']

        # Remove eval cache
        if cache_dir is not None:
            if cache_dir.startswith('_'):
                cache_dir = cache_dir[1:]
                print('Remove eval caches... ')
                shutil.rmtree(cache_dir, ignore_errors=True)
                os.makedirs(cache_dir, exist_ok=True)

    print('--------------------------------------------------')
    print('Target Dir: %s' % target_dirs)
    print('PKL Dir: %s' % pkl_dir)
    print('PDF Dir: %s' % pdf_dir)
    print('--------------------------------------------------')

    while True:
        for target_dir in target_dirs:
            model_path = None
            if not os.path.exists(target_dir):
                continue

            file_list = glob.glob(f"{target_dir}/*.pt")
            file_list.sort()
            file_list = file_list[::-1]

            for file_path in file_list:
                # Extract only the filename from the file path.
                file_name = os.path.basename(file_path)

                # Evaluated before or not a target
                if not file_name.startswith('0.0000_'):
                    continue

                try:
                    # Split the filename by '_' and convert the second element to a float.
                    # Example: extract '0.9527' from '0.0000_0.9527_...'
                    parts = file_name.split('_')

                    # Check if parts[1] exists and can be converted to a number.
                    if len(parts) > 1:
                        current_eval_score = float(parts[1])
                    else:
                        # If the filename format is unexpected, move to the next file.
                        continue

                except ValueError:
                    # If it cannot be converted to a number, move to the next file.
                    continue
                except IndexError:
                    # If parts[1] does not exist (e.g., format like '0.0000.pt'), move to the next file.
                    continue

                # Add condition to check if the extracted eval_score is greater than or equal to eval_score_thr.
                if current_eval_score >= eval_score_thr:
                    model_path = file_path
                    break  # Found a file that satisfies the condition, so exit the loop.

            if model_path is not None:
                print('Found: %s.' % model_path)
                total_f1_val, table_f1_val = run_eval_det_with_retry(cfg=cfg, load_from=model_path, verbose=True,
                                                                     eval_type=eval_type)
                f1_str = '/%.4f (%.4f)_' % (total_f1_val, table_f1_val)
                if do_rename:
                    os.rename(model_path, model_path.replace("/0.0000_", f1_str))

        if loop_sleep_time > 0:
            print('Wait %d sec... ' % loop_sleep_time)
            time.sleep(loop_sleep_time)


def print_class_priority():
    cls_names = ['text', 'title', 'picture', 'table', 'list-item', 'page-header', 'page-footer', 'section-header',
                 'footnote',
                 'caption', 'formula']
    cls_freq = ['text', 'section-header', 'list-item', 'formula', 'caption', 'table', 'page-header', 'page-footer',
                'footnote', 'picture', 'title']

    class_priority = []
    for cls_name in cls_freq:
        class_priority.append(cls_names.index(cls_name))
    print(class_priority)
    exit(0)


def export2onnx(cfg):
    device = 'cpu'
    val_batch_size = 1

    model_cfg_path = cfg['export2onnx']['model_cfg']
    checkpoint_path = cfg['export2onnx']['check_point']
    save_onnx_path = cfg['export2onnx']['save_path']
    verification_threshold = cfg['export2onnx']['verification_threshold']

    with open(model_cfg_path, "rb") as f:
        model_cfg = anyconfig.load(f)

    model_cfg['model']['option']['ExportONNX'] = True
    model_type = model_cfg['model']['option']['conv_type']
    feature_types = model_cfg['model']['option']['feature_types']

    data_rf_names = model_cfg['data']['rf_names']
    data_class_names = model_cfg['data']['class_list']
    data_class_map = {name: i for i, name in enumerate(data_class_names)}

    val_lmdb_path = cfg['train']['val_dataset']['lmdb_path']
    val_data = DocumentLMDBDataset(lmdb_path=val_lmdb_path, cache_size=0,
                                   rf_names=data_rf_names)
    val_dataloader = GeometricDataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0
    )

    # Instantiate the model
    model = get_model(model_cfg, data_class_names)
    model = model.to(device)

    # Load model weights
    if not os.path.exists(checkpoint_path):
        raise Exception(f"Error: Model checkpoint not found at '{checkpoint_path}'. Aborting.")
    print(f"Loading model from {checkpoint_path} for inference...")
    load_model_and_optimizer(model, checkpoint_path)
    model.eval()

    # Prepare input data for ONNX export and verification
    # The same data batch will be used for both to ensure consistency.
    data_batch = next(iter(val_dataloader))
    data_batch = data_batch.to(device)

    # Calculate original output from the PyTorch model
    with torch.no_grad():
        original_node_logits, original_edge_logits = model.forward_with_batch(data_batch)

    # Define all required input tensors for ONNX export
    # The order must match the model's forward_with_batch method
    k = min(data_batch.x.shape[0], 20)

    input_data_dict = {
        'x': data_batch.x,
        'edge_index': data_batch.edge_index,
        'edge_attr': data_batch.edge_attr,
        'rf_features': data_batch.rf_features,
        'k': k,
        'text_patterns': data_batch.text_patterns,
        'image_features': data_batch.img_features,
        'batch': data_batch.batch
    }

    input_names = ["x", "edge_index", "edge_attr", "rf_features", "text_patterns", "image_features", "k", "batch"]
    if model_type in ['GAT', 'NNConv']:
        onnx_input_names = ["x", "edge_index", "edge_attr", "rf_features", "text_patterns"]
    elif model_type in ['CustomDGC']:
        onnx_input_names = ["x", "edge_index", "edge_attr", "rf_features", "text_patterns", "image_features", 'k', 'batch']
    else:
        raise Exception(f'Not supported model_type = {model_type}!')

    example_inputs = []
    for input_name in input_names:
        example_inputs.append(input_data_dict[input_name])

    example_inputs = tuple(example_inputs)
    output_names = ["node_logits", "edge_logits"]

    dynamic_axes = {
        "x": {0: "num_nodes"},
        "edge_index": {1: "num_edges"},
        "edge_attr": {0: "num_edges"},
        "rf_features": {0: "num_nodes"},
        "text_patterns": {0: "num_nodes"},
        "image_features": {0: "num_nodes"},
        "batch": {0: "num_nodes"},
        "node_logits": {0: "num_nodes"},
        "edge_logits": {0: "num_edges"},
    }

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        example_inputs,
        save_onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,
        do_constant_folding=True,
        verbose=True
    )
    print(f"ONNX model has been exported to: {save_onnx_path}")


    # Calculate original output from the PyTorch model
    with torch.no_grad():
        original_node_logits, original_edge_logits = model.forward_with_batch(data_batch)

    # -------- Load and test the ONNX model --------
    print("\nStarting ONNX model verification...")

    # Load the ONNX model using onnxruntime
    ort.set_default_logger_severity(3)
    ort_session = ort.InferenceSession(save_onnx_path)

    total_count = 0
    error_count = 0

    for batch_idx, data_batch in enumerate(val_dataloader):
        print(f'{batch_idx} / {len(val_dataloader)} ...')
        total_count += 1
        data_batch = data_batch.to(device)

        input_data_dict = {
            'x': data_batch.x,
            'edge_index': data_batch.edge_index,
            'edge_attr': data_batch.edge_attr,
            'rf_features': data_batch.rf_features,
            'k': k,
            'text_patterns': data_batch.text_patterns,
            'batch': data_batch.batch
        }

        if 'image' in feature_types:
            input_data_dict['image_features'] = data_batch.img_features,

        # Prepare input dictionary for ONNX runtime
        # Convert PyTorch tensors to NumPy arrays
        k = min(data_batch.x.shape[0], 20)
        onnx_inputs = {}

        for input_name in onnx_input_names:
            if input_name == 'k':
                onnx_inputs['k'] = np.array(k)
            else:
                if input_name in input_data_dict:
                    onnx_inputs[input_name] = input_data_dict[input_name].detach().cpu().numpy()

        # Run the ONNX model
        with torch.no_grad():
            original_node_logits, original_edge_logits = model.forward_with_batch(data_batch)

        ort_outputs = ort_session.run(None, onnx_inputs)
        onnx_node_logits, onnx_edge_logits = ort_outputs

        # Compare ONNX outputs with original PyTorch outputs
        torch_node_logits = original_node_logits.detach().cpu().numpy()
        torch_edge_logits = original_edge_logits.detach().cpu().numpy()

        # Check output shapes
        assert onnx_node_logits.shape == torch_node_logits.shape, f"Node output shapes mismatch: {onnx_node_logits.shape} vs {torch_node_logits.shape}"
        assert onnx_edge_logits.shape == torch_edge_logits.shape, f"Edge output shapes mismatch: {onnx_edge_logits.shape} vs {torch_edge_logits.shape}"

        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True)

        onnx_node_probs = softmax(onnx_node_logits)
        torch_node_probs = softmax(torch_node_logits)

        onnx_edge_probs = softmax(onnx_edge_logits)
        torch_edge_probs = softmax(torch_edge_logits)

        try:
            np.testing.assert_allclose(onnx_node_probs, torch_node_probs, rtol=verification_threshold,
                                       atol=verification_threshold)
            np.testing.assert_allclose(onnx_edge_probs, torch_edge_probs, rtol=verification_threshold,
                                       atol=verification_threshold)
            print("\nONNX model outputs match PyTorch outputs. Verification successful!")
            print(f"ONNX node_logits shape: {onnx_node_logits.shape}, ONNX edge_logits shape: {onnx_edge_logits.shape}")
        except AssertionError as e:
            print(str(e))
            # print(f"\nONNX model verification failed: {e}")
            # # Print debug info to help with troubleshooting
            # print("\n--- Debug Info ---")
            # print("ONNX Inputs:")
            # for name, arr in onnx_inputs.items():
            #     print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
            # print("-" * 20)
            error_count += 1

    print('Total: %d, Error: %d (%.2f)' % (total_count, error_count, error_count / total_count))


def test_knn_function():
    import torch
    from train.core.common.model_util import custom_knn_batched
    from torch_cluster.knn import knn

    x = torch.load('./temp/dgcn_input_x.pt').to('cpu')
    batch = torch.load('./temp/dgcn_input_batch.pt').to('cpu')

    k = 5

    # Run custom_knn_batched
    edge_index_custom = custom_knn_batched(x, k, batch)

    # Run official knn
    edge_index_official = knn(x, x, k, batch_x=batch, batch_y=batch).flip([1])

    # Create dictionaries to store neighbors per node
    custom_neighbors = {i: set() for i in range(x.size(0))}
    official_neighbors = {i: set() for i in range(x.size(0))}

    for i in range(edge_index_custom.size(1)):
        src, tgt = edge_index_custom[0, i].item(), edge_index_custom[1, i].item()
        custom_neighbors[src].add(tgt)

    for i in range(edge_index_official.size(1)):
        src, tgt = edge_index_official[0, i].item(), edge_index_official[1, i].item()
        official_neighbors[src].add(tgt)

    # Compare neighbors
    all_match = True
    for i in range(x.size(0)):

        print(f"\n--- Node {i} ---")
        print(f"custom_neighbors ({len(custom_neighbors[i])}): {sorted(custom_neighbors[i])}")
        print(f"official_neighbors ({len(official_neighbors[i])}): {sorted(official_neighbors[i])}")

        if custom_neighbors[i] != official_neighbors[i]:
            all_match = False
            print(f"\nTest failed for node {i}:")

            diff_in_custom = custom_neighbors[i] - official_neighbors[i]
            diff_in_official = official_neighbors[i] - custom_neighbors[i]

            if diff_in_custom:
                print(f"  In custom but NOT in official: {[f'({i}, {d})' for d in sorted(diff_in_custom)]}")
            if diff_in_official:
                print(f"  In official but NOT in custom: {[f'({i}, {d})' for d in sorted(diff_in_official)]}")

            print(f"\nDistances and batch IDs for custom_neighbors of node {i}:")
            node_feat = x[i]
            for neighbor_idx in sorted(custom_neighbors[i]):
                neighbor_feat = x[neighbor_idx]
                distance = torch.norm(node_feat - neighbor_feat).item()
                batch_id = batch[neighbor_idx].item()
                print(f"  Node {neighbor_idx}: Distance={distance:.4f}, Batch ID={batch_id}")

    if all_match:
        print("\n? Test successful: custom_knn_batched and torch_cluster.knn produce matching neighbor sets.")
    else:
        raise AssertionError("\n? K-NN results do not match. See printed differences above.")


def test_torch_cluster_knn(k: int = 5, num_nodes: int = 100, feature_dim: int = 16):
    """
    Tests if torch_cluster.knn returns fewer than k neighbors
    when more than k neighbors are available.

    Args:
        k (int): The number of neighbors to find.
        num_nodes (int): The total number of nodes to generate.
        feature_dim (int): The dimension of node features.
    """
    print(f"--- Running test for torch_cluster.knn with k={k} ---")

    from torch_cluster.knn import knn

    # Generate random data with a single batch
    x = torch.randn(num_nodes, feature_dim)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Ensure there are more than k nodes to choose from in the batch
    if num_nodes <= k:
        print(f"Skipping test: num_nodes ({num_nodes}) must be greater than k ({k}) to be effective.")
        return

    # Run knn to find k nearest neighbors
    edge_index = knn(x, x, k, batch_x=batch, batch_y=batch).flip([0])

    # Store neighbor counts for each node
    neighbor_counts = torch.bincount(edge_index[1], minlength=num_nodes)

    # Check if any node has fewer than k neighbors
    has_fewer_neighbors = (neighbor_counts < k).any()

    print(f"\nTotal nodes generated: {num_nodes}")
    print(f"K-value: {k}")
    print(f"Total edges found: {edge_index.size(1)}")

    if has_fewer_neighbors:
        print("\n[Test Result] FAILED: Some nodes have fewer than k neighbors.")

        # Print a few examples of nodes with fewer than k neighbors
        nodes_with_fewer = torch.where(neighbor_counts < k)[0]
        print("Nodes with fewer than k neighbors:")
        for node_id in nodes_with_fewer[:5]:
            count = neighbor_counts[node_id].item()
            print(f"  - Node {node_id}: {count} neighbors found.")

    else:
        print("\n[Test Result] PASSED: All nodes successfully returned at least k neighbors.")

    # Final assertion to fail the test if the condition is met
    assert not has_fewer_neighbors, f"Test failed: Some nodes returned fewer than {k} neighbors."


def test_scatter_max_op(num_nodes: int, num_edges: int, num_features: int, tolerance: float = 1e-6):
    from torch_scatter import scatter_max
    from train.core.common.model_util import max_aggregate_numpy

    """
    Tests if torch_scatter.scatter_max and max_aggregate_numpy produce identical results.

    Args:
        num_nodes (int): The number of nodes in the graph.
        num_edges (int): The number of edges in the graph.
        num_features (int): The number of features per node/edge.
        tolerance (float): The numerical tolerance for comparison.
    """
    print(f"Testing with: Nodes={num_nodes}, Edges={num_edges}, Features={num_features}")

    # 1. Generate random input data
    # Create random edge indices (source to target)
    target_nodes_np = np.random.randint(0, num_nodes, size=num_edges, dtype=np.int64)
    # Create random messages/features
    messages_np = np.random.randn(num_edges, num_features).astype(np.float32)

    # 2. Run both aggregation methods

    # NumPy version
    output_numpy = max_aggregate_numpy(messages_np, target_nodes_np, num_nodes)

    # torch_scatter version
    messages_torch = torch.from_numpy(messages_np)
    target_nodes_torch = torch.from_numpy(target_nodes_np)

    # Initializing output with -inf to match scatter_max behavior
    # Note: torch_scatter handles the no-edge case by returning -inf, so we'll check later
    output_torch, _ = scatter_max(
        src=messages_torch,
        index=target_nodes_torch,
        dim=0,
        dim_size=num_nodes
    )

    # 3. Handle the -inf case for no-edge nodes in torch_scatter output
    # `torch_scatter` returns -inf for nodes with no incoming edges.
    # The NumPy implementation returns 0 for these.
    # We need to make them consistent for comparison.
    # We'll use a mask to check for -inf values and replace them with 0.
    is_inf_mask = torch.isinf(output_torch)
    output_torch = torch.where(is_inf_mask, torch.tensor(0.0), output_torch)

    # 4. Compare the results
    # Convert PyTorch tensor to NumPy array for comparison
    output_torch_np = output_torch.numpy()

    # `np.allclose` is ideal for comparing floating-point arrays with a tolerance
    if np.allclose(output_numpy, output_torch_np, atol=tolerance, rtol=tolerance):
        print("? Success: The outputs from both implementations are identical.")
        return True
    else:
        print("? Failure: The outputs are NOT identical.")
        # You can add more detailed logging here for debugging if needed
        # print("NumPy output:\n", output_numpy)
        # print("torch_scatter output:\n", output_torch_np)
        return False


def test_safe_max_aggregation():
    from torch_scatter import scatter_max
    from train.core.common.model_util import safe_max_aggregation

    torch.manual_seed(42)

    # Parameters
    num_nodes = 10
    num_edges = 50
    num_features = 8

    # Random edge connections and messages
    index = torch.randint(0, num_nodes, (num_edges,))
    message = torch.randn(num_edges, num_features)

    # Compute both versions
    custom_out = safe_max_aggregation(message, index, num_nodes)
    scatter_out, _ = scatter_max(message, index, dim=0, dim_size=num_nodes)

    # Compare
    if torch.allclose(custom_out, scatter_out, atol=1e-6):
        print("? Outputs match!")
    else:
        print("? Outputs do NOT match!")
        print("Diff:", (custom_out - scatter_out).abs().max())
        print("Custom out:\n", custom_out)
        print("Scatter out:\n", scatter_out)


def test_softmax_aggregation():
    """
    Tests if the custom manual softmax aggregation yields results
    similar to torch_scatter.scatter_softmax followed by scatter_add.
    """
    from torch_scatter import scatter_softmax, scatter_add
    from train.core.common.model_util import softmax_aggregation

    # Test Parameters
    num_nodes = 100
    num_edges = 500
    num_features = 32

    # Generate random input data
    # Ensure messages can sometimes be large to test stability (or lack thereof for manual non-max-trick)
    x = torch.randn(num_nodes,
                    64)  # Original node features (not directly used in aggregation result, but good for context)

    # target_nodes: Each edge connects to a target node
    target_nodes = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)

    # message: Features on the edges/messages
    # Introduce some potentially large values to test overflow
    message_small = torch.randn(num_edges, num_features) * 5
    message_large = torch.randn(num_edges, num_features) * 10

    # Combine messages, ensuring some large values are present
    message = torch.cat([message_small[:num_edges // 2], message_large[num_edges // 2:]], dim=0)
    # Ensure message is on the same device as x and has a floating-point type
    message = message.to(x.device).to(x.dtype)

    # Initial output tensor (usually zeros)
    out_initial = torch.zeros(num_nodes, num_features, device=x.device, dtype=message.dtype)

    print("\n--- Running Test ---")
    print(f"Test Parameters: num_nodes={num_nodes}, num_edges={num_edges}, num_features={num_features}")
    print(f"Message min: {message.min().item():.4f}, max: {message.max().item():.4f}")

    # --- Method 1: torch_scatter.scatter_softmax + scatter_add ---
    try:
        alpha_scatter = scatter_softmax(src=message, index=target_nodes, dim=0)
        weighted_msg_scatter = alpha_scatter * message
        out_scatter = scatter_add(src=weighted_msg_scatter, index=target_nodes, dim=0, dim_size=num_nodes)
        print(f"Scatter_softmax output max: {out_scatter.max().item():.4f}")
        print(f"Scatter_softmax output min: {out_scatter.min().item():.4f}")
        print(f"Scatter_softmax output has NaN: {torch.isnan(out_scatter).any().item()}")
    except Exception as e:
        print(f"Error in scatter_softmax path: {e}")
        out_scatter = None  # Indicate failure

    # --- Method 2: Manual softmax_aggregation (your proposed ONNX path) ---
    try:
        out_manual = softmax_aggregation(x, out_initial, target_nodes, num_features, num_nodes, message)
        print(f"Manual softmax output max: {out_manual.max().item():.4f}")
        print(f"Manual softmax output min: {out_manual.min().item():.4f}")
        print(f"Manual softmax output has NaN: {torch.isnan(out_manual).any().item()}")
    except Exception as e:
        print(f"Error in manual softmax path: {e}")
        out_manual = None  # Indicate failure

    # --- Comparison ---
    if out_scatter is not None and out_manual is not None:
        # Check for NaNs first, as allclose will fail if NaNs are present
        if torch.isnan(out_scatter).any() or torch.isnan(out_manual).any():
            print("Comparison failed: One or both outputs contain NaN.")
            assert False, "Outputs contain NaN, comparison cannot be meaningful."

        # Use torch.allclose for floating point comparison
        # atol (absolute tolerance): useful for values near zero
        # rtol (relative tolerance): useful for larger values
        is_close = torch.allclose(out_scatter, out_manual, rtol=1e-4, atol=1e-6)
        print(f"Outputs are close: {is_close}")

        if not is_close:
            # Print some statistics for debugging if not close
            diff = torch.abs(out_scatter - out_manual)
            print(f"Max absolute difference: {diff.max().item()}")
            print(f"Mean absolute difference: {diff.mean().item()}")

            # Show a few differing elements
            diff_indices = (diff > 1e-6).nonzero(as_tuple=True)
            if len(diff_indices[0]) > 0:
                print(f"First 5 differing elements (scatter, manual):")
                for i in range(min(5, len(diff_indices[0]))):
                    idx = (diff_indices[0][i], diff_indices[1][i])
                    print(f"  Index {idx}: ({out_scatter[idx].item():.6f}, {out_manual[idx].item():.6f})")

        assert is_close, "Softmax aggregation results do not match."
    else:
        assert False, "One or both aggregation methods failed."

    print("Test finished.")

def eval_det_model_with_pdf(cfg, load_from=None, verbose=True):
    from train.infer.pymupdf_util import create_input_data_by_pymupdf
    from train.infer.onnx.BoxRFDGNN import get_nn_input_from_datadict

    pdf_dir = cfg['test_onnx']['pdf_dir']
    pkl_dir = cfg['test_onnx']['pkl_dir']
    config_path = cfg['test_onnx']['config_path']
    features_path = cfg['test_onnx']['features_path']
    iou_threshold = cfg['test_onnx']['iou_threshold']
    cache_dir = cfg['test_onnx']['cache_dir']

    if cache_dir is None:
        cache_dir = './Not-Exist-Path'

    device = cfg['test_onnx']['device']
    feature_extractor_path = cfg['test_onnx']['imf_model_path']

    if load_from is None:
        load_from = cfg['test_onnx']['load_from']

    with open(config_path, "rb") as f:
        model_cfg = anyconfig.load(f)

    data_class_names = cfg['data']['class_list']
    data_class_map = {}
    for i in range(len(data_class_names)):
        data_class_map[data_class_names[i]] = i
    class_priority_list = cfg['data']['class_priority']

    model = get_model(model_cfg, data_class_names)
    model = model.to(device)

    # 모델 로드
    if not os.path.exists(load_from):
        print(f"Error: Model checkpoint not found at '{load_from}'. Aborting inference.")
        return [], []

    print(f"Loading model from {load_from} for inference...")
    load_model_and_optimizer(model, load_from)
    model.eval()

    feature_extractor = ort.InferenceSession(
        feature_extractor_path, providers=['CPUExecutionProvider']
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

        cache_file = f'{cache_dir}/{file_name}.pkl'
        st_time = time.time()
        nn_input = None
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            pkl_data = cache_data['pkl_data']
            gt_labels = cache_data['gt_labels']
            page_img = cache_data['page_img']
            data_dict = cache_data['data_dict']
            nn_input =  cache_data['nn_input']
            fe_time = 0.0
        else:
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

            fe_time = time.time()
            data_dict = create_input_data_by_pymupdf(pdf_file, features_path=features_path)
            fe_time = time.time() - fe_time

        infer_time = time.time()
        pred_results = []
        bboxes = np.array(data_dict['bboxes'], dtype=np.float32)
        if len(bboxes) > 0:
            if nn_input is None:
                x, edge_index, edge_attr, nn_index, nn_attr, rf_feature, text_feature, image_features = \
                    get_nn_input_from_datadict(data_dict, model_cfg, feature_extractor=feature_extractor)
                if os.path.exists(cache_dir):
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'pkl_data': pkl_data,
                            'gt_labels': gt_labels,
                            'page_img': page_img,
                            'data_dict': data_dict,
                            'nn_input': (x, edge_index, edge_attr, nn_index, nn_attr, rf_feature, text_feature,
                                         image_features),
                        }, f)
            else:
                x, edge_index, edge_attr, nn_index, nn_attr, rf_feature, text_feature, image_features = nn_input

            k = min(len(bboxes), 20)

            x = torch.tensor(x, dtype=torch.float32).to(device)
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
            edge_attr =  torch.tensor(edge_attr, dtype=torch.float32).to(device)
            rf_feature = torch.tensor(rf_feature, dtype=torch.float32).to(device)
            text_feature = torch.tensor(text_feature, dtype=torch.float32).to(device)
            image_features = torch.tensor(image_features, dtype=torch.float32).to(device)

            node_logits, edge_logits = model.forward(x, edge_index, edge_attr, rf_feature, text_feature, image_features,
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
                                                                      edge_matrix,
                                                                      bboxes, class_priority_list)
            for group_idx, group in enumerate(groups):
                g_bbox = group['group_bbox']
                cls_name = data_class_names[group['group_class']]
                if cls_name == 'unlabelled':
                    continue
                g_bbox.append(cls_name)
                pred_results.append(g_bbox)
            infer_time = time.time() - infer_time
        else:
            infer_time = 0

        total_time = time.time() - st_time
        fe_elapsed_time.append(fe_time)
        if_elapsed_time.append(infer_time)
        total_elapsed_time.append(total_time)

        if verbose:
            print(f'[{file_idx}/{len(file_list)}] {file_name[:-4]} ({len(data_dict["bboxes"])} bboxes, '
                  f'Total: {total_time:.2f} sec / FE: {fe_time:.2f}, IF: {infer_time:.2f})...')
        elif file_idx % 1000 == 0:
            print(f'{file_idx}/{len(file_list)} ...')

        # Extract box coordinates and labels
        gt_boxes = [(x1, y1, x2, y2, label) for x1, y1, x2, y2, _, label in gt_labels]
        pred_boxes = [(x1, y1, x2, y2, label) for x1, y1, x2, y2, label in pred_results]

        matched_gt = set()
        matched_pred = set()

        if '' == 'D':
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

    table_f1 = 0.0
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

        if cls == 'table':
            table_f1 = f1

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (
                                                                                                              overall_precision + overall_recall) > 0 else 0.0

    print(f"{'Overall':<20} {overall_precision:10.3f} {overall_recall:10.3f} {overall_f1:10.3f}")
    print('Total - %d sec, mean %.2f sec, std %.2f sec' % (np.sum(total_elapsed_time), np.mean(total_elapsed_time),
                                                           np.std(total_elapsed_time)))
    print('FE - %d sec, mean %.2f sec, std %.2f sec' % (np.sum(fe_elapsed_time), np.mean(fe_elapsed_time),
                                                        np.std(fe_elapsed_time)))
    print('IF - %d sec, mean %.2f sec, std %.2f sec' % (np.sum(if_elapsed_time), np.mean(if_elapsed_time),
                                                        np.std(if_elapsed_time)))
    return overall_f1, table_f1


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    if len(sys.argv) < 2:
        print("Usage: python test_gnn.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "rb") as f:
        cfg = anyconfig.load(f)

    task_name = cfg['test_task_name']
    if task_name == 'eval_det_model':
        eval_det_model(cfg=cfg)
    elif task_name == 'eval_det_model_with_pdf':
        eval_det_model_with_pdf(cfg)
    elif task_name == 'eval_loop':
        eval_loop(cfg)
    elif task_name == 'print_class_priority':
        print_class_priority()
    elif task_name == 'export2onnx':
        export2onnx(cfg)
    elif task_name == 'test_knn_function':
        test_knn_function()
    elif task_name == 'test_torch_cluster_knn':
        test_torch_cluster_knn()
    elif task_name == 'test_scatter_max_op':
        test_scatter_max_op(num_nodes=100, num_edges=500, num_features=64)
    elif task_name == 'test_safe_max_aggregation':
        test_safe_max_aggregation()
    elif task_name == 'test_softmax_aggregation':
        test_softmax_aggregation()
