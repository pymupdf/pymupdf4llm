import os
import shutil
import sys
import json
import csv
import time
import statistics
import fitz  # PyMuPDF
from collections import defaultdict
from tqdm import tqdm


def print_eval_result(eval_results):
    """
    Evaluation results from a detection model are formatted and printed in a human-readable way.

    Args:
        eval_results (list): A list containing dictionaries of evaluation metrics.
                             Each dictionary should have at least '__dataset_name__',
                             '__overall__', and various category metrics.
                             Example: [{'__dataset_name__': 'DoclayNet_val', '__overall__': {...}, 'caption': {...}}]
    """
    if not eval_results:
        print("No evaluation results to display.")
        return

    # Assuming eval_results is a list containing one main result dictionary
    for result_dict in eval_results:
        dataset_name = result_dict.get('__dataset_name__', 'Unknown Dataset')
        overall_metrics = result_dict.get('__overall__', {})

        print(f"\n--- Evaluation Results for Dataset: {dataset_name} ---")

        # Print Overall Metrics
        print("\nOverall Metrics:")
        print(f"  Precision: {overall_metrics.get('precision', 0.0):.4f}")
        print(f"  Recall:    {overall_metrics.get('recall', 0.0):.4f}")
        print(f"  F1 Score:  {overall_metrics.get('f1', 0.0):.4f}")
        print(f"  GT Count:  {overall_metrics.get('gt_count', 0):,d}")
        print(f"  Det Count: {overall_metrics.get('det_count', 0):,d}")
        print(f"  TP:        {overall_metrics.get('tp', 0):,d}")
        print(f"  FP:        {overall_metrics.get('fp', 0):,d}")
        print(f"  FN:        {overall_metrics.get('fn', 0):,d}")

        # Print Category Metrics
        print("\nCategory Metrics:")

        # Define table headers and their approximate widths for alignment
        headers = ["Category", "Precision", "Recall", "F1 Score", "GT Count", "Det Count", "TP", "FP", "FN"]
        col_widths = {
            "Category": 15,
            "Precision": 10,
            "Recall": 10,
            "F1 Score": 10,
            "GT Count": 10,
            "Det Count": 10,
            "TP": 10,
            "FP": 10,
            "FN": 10,
        }

        # Print table header
        header_line = "| " + " | ".join(f"{h:<{col_widths[h]}}" for h in headers) + " |"
        separator_line = "+-" + "-+-".join("-" * col_widths[h] for h in headers) + "-+"
        print(separator_line)
        print(header_line)
        print(separator_line)

        # Sort categories alphabetically, excluding special keys
        categories = sorted([k for k in result_dict.keys() if not k.startswith('__')])

        # Print metrics for each category
        for category in categories:
            metrics = result_dict.get(category, {})
            row = (
                f"| {category:<{col_widths['Category']}} | "
                f"{metrics.get('precision', 0.0):<{col_widths['Precision']}.4f} | "
                f"{metrics.get('recall', 0.0):<{col_widths['Recall']}.4f} | "
                f"{metrics.get('f1', 0.0):<{col_widths['F1 Score']}.4f} | "
                f"{metrics.get('gt_count', 0):<{col_widths['GT Count']},d} | "
                f"{metrics.get('det_count', 0):<{col_widths['Det Count']},d} | "
                f"{metrics.get('tp', 0):<{col_widths['TP']},d} | "
                f"{metrics.get('fp', 0):<{col_widths['FP']},d} | "
                f"{metrics.get('fn', 0):<{col_widths['FN']},d} |"
            )
            print(row)
        print(separator_line)


# --- 1. IoU Calculation Function ---

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) for two bounding boxes (x1, y1, x2, y2).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


# --- 2. Visualization Function ---

def make_vis_pdf(pdf_path, gt_bboxes_doc, det_bboxes_doc, output_dir, iou_threshold, class_names=None):
    """
    Visualizes detection results and text nodes on a PDF and saves the new PDF.

    Args:
        pdf_path (str): Path to the original PDF file.
        gt_bboxes_doc (list): List of GT bboxes (x1, y1, x2, y2, class_name) for the document.
        det_bboxes_doc (list): List of detection bboxes (x1, y1, x2, y2, class_name) for the document.
        output_dir (str): Directory to save the visualized PDF.
        iou_threshold (float): IoU threshold for determining TP/FP.
        class_names (list of str, optional): List of class names to visualize.
                                             If None, all classes will be visualized.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        tqdm.write(f"Error opening PDF {pdf_path} for visualization: {e}. Skipping visualization.")
        return

    if len(doc) == 0:
        tqdm.write(f"Warning: PDF {pdf_path} has no pages. Skipping visualization.")
        doc.close()
        return

    output_filepath = os.path.join(output_dir, os.path.basename(pdf_path))
    os.makedirs(output_dir, exist_ok=True)

    # Add OC layers for different visualizations
    ocg_gt = doc.add_ocg("GT", on=0)
    ocg_det = doc.add_ocg("DET", on=1)
    ocg_line = doc.add_ocg("node", on=0)

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # Visualize text lines on the "node" layer
        # Get detailed text information (including lines and spans)
        page_dict = page.get_text("dict")
        # Iterate through blocks, then lines within each block
        for block in page_dict['blocks']:
            if 'lines' in block:
                for line in block['lines']:
                    line_rect = fitz.Rect(line['bbox'])
                    page.draw_rect(line_rect, color=(0.5, 0.5, 0.5), width=1, oc=ocg_line)

        # For simplicity, this example only adds detection/GT for the first page
        if page_idx == 0:
            # Filter GT bboxes based on class_names
            filtered_gt_bboxes = [b for b in gt_bboxes_doc if class_names is None or b[4] in class_names]
            gt_bboxes_copy = [list(b) for b in filtered_gt_bboxes]
            matched_gt_indices_for_vis = set()

            # Step 1: Draw filtered GT bboxes on "GT Layer" (blue)
            for i, gt_box_with_class in enumerate(gt_bboxes_copy):
                x1, y1, x2, y2, class_name = gt_box_with_class
                rect = fitz.Rect(x1, y1, x2, y2)
                page.draw_rect(rect, color=(0, 0, 1), width=2, oc=ocg_gt)
                page.insert_text(fitz.Point(x1, y1 - 5), class_name, fontsize=8, color=(0, 0, 1), oc=ocg_gt)

            # Step 2: Determine and Draw TP/FP for filtered detections on "DET Layer"
            filtered_det_bboxes = [b for b in det_bboxes_doc if class_names is None or b[4] in class_names]

            for d_box_with_class in filtered_det_bboxes:
                d_box = d_box_with_class[:4]
                d_class = d_box_with_class[4]

                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt_box_with_class in enumerate(gt_bboxes_copy):
                    if gt_idx in matched_gt_indices_for_vis:
                        continue

                    gt_box = gt_box_with_class[:4]
                    gt_class = gt_box_with_class[4]

                    if d_class == gt_class:
                        iou = calculate_iou(d_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                if best_iou >= iou_threshold:
                    rect = fitz.Rect(d_box)
                    page.draw_rect(rect, color=(0, 1, 0), width=2, oc=ocg_det)
                    page.insert_text(fitz.Point(d_box[0], d_box[1] - 5), f"{d_class} ({best_iou:.2f})", fontsize=8,
                                     color=(0, 1, 0), oc=ocg_det)
                    matched_gt_indices_for_vis.add(best_gt_idx)
                else:
                    rect = fitz.Rect(d_box)
                    page.draw_rect(rect, color=(1, 0, 0), width=2, oc=ocg_det)
                    page.insert_text(fitz.Point(d_box[0], d_box[1] - 5), f"{d_class} ({best_iou:.2f})", fontsize=8,
                                     color=(1, 0, 0), oc=ocg_det)

    try:
        doc.save(output_filepath)
        doc.close()
    except Exception as e:
        tqdm.write(f"Error saving visualized PDF {output_filepath}: {e}. Skipping saving.")


# --- 3. Evaluation Function ---
def evaluate_detection(det_func, result_csv_path=None, gt_dir='eval/resources/gt', iou_threshold=0.6,
                       class_names=None, gt_names=('DoclayNet_val',),
                       pdf_dirs=None, vis_error_count=None, vis_pdf_dir=None, verbose=True):
    """
    Compares detection results (obtained solely via det_func) with ground truth to calculate
    Precision, Recall, and F1-score.

    Args:
        det_func (callable): Detection function to call for results.
                              It must accept (pdf_path) and return
                              a list of bboxes: [x1, y1, x2, y2, class_name].
        result_csv_path (str, optional): The full path including the filename where the results CSV
                                         should be saved. If None, the CSV is not saved. Defaults to None.
        gt_dir (str, optional): Path to the top-level directory containing ground truth JSON files.
                                Defaults to 'eval/resources/gt'.
        iou_threshold (float, optional): Minimum IoU threshold for True Positive. Defaults to 0.6.
        class_names (list of str, optional): List of classes to evaluate. Defaults to None (all classes).
        gt_names (list of str, optional): List of GT dataset names (subdirectories in gt_dir) to include.
                                          This is used only to filter which GT folders to process.
        pdf_dirs (list of str): List of root directories where original PDF files are located.
                                Must be a non-empty list of paths.
        vis_error_count (int, optional): If (FP + FN) for a file exceeds this, visualize the PDF. Defaults to None.
        vis_pdf_dir (str, optional): Root directory to save visualized PDFs. If provided, visualization PDFs
                                     are saved directly into this directory.
        verbose (bool, optional): If True, display progress bars via tqdm. Defaults to True.

    Returns:
        list: A list of evaluation results, broken down by dataset.
    """
    # --- Input Validation ---
    if not isinstance(pdf_dirs, list) or not pdf_dirs:
        print("Error: 'pdf_dirs' must be a non-empty list of paths to find PDF files. Terminating evaluation.")
        sys.exit(1)

    if det_func is None:
        print("Error: 'det_func' must be provided. Cannot proceed with evaluation.")
        sys.exit(1)

    # --- GT Directory Validation ---
    if not os.path.exists(gt_dir):
        print(f"Error: GT directory not found at '{gt_dir}'. Terminating evaluation.")
        sys.exit(1)
    if not os.path.isdir(gt_dir):
        print(f"Error: GT path '{gt_dir}' is not a directory. Terminating evaluation.")
        sys.exit(1)

    all_gt_datasets_in_dir = [d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))]

    if not all_gt_datasets_in_dir:
        print(f"Error: No subdirectories (datasets) found in GT directory '{gt_dir}'. Terminating evaluation.")
        sys.exit(1)

    # Determine which datasets to process based on gt_names
    if gt_names is None:
        datasets_to_process_gt_side = set(all_gt_datasets_in_dir)
    else:
        datasets_to_process_gt_side = set(all_gt_datasets_in_dir).intersection(gt_names)

    datasets_to_process_gt_side_sorted = sorted(list(datasets_to_process_gt_side))

    if not datasets_to_process_gt_side_sorted:
        print(
            f"Error: No datasets to process after filtering with 'gt_names' or no valid datasets found in '{gt_dir}'. Terminating evaluation.")
        sys.exit(1)

    # Check if any JSON files exist in the chosen datasets
    json_found = False
    for dataset_name in datasets_to_process_gt_side_sorted:
        gt_dataset_path = os.path.join(gt_dir, dataset_name)
        if any(f.endswith('.json') for f in os.listdir(gt_dataset_path)):
            json_found = True
            break

    if not json_found:
        print(
            f"Error: No JSON files found in the processed GT dataset directories within '{gt_dir}'. Terminating evaluation.")
        sys.exit(1)

    # --- End of Initial Validation ---

    if verbose:
        tqdm.write(f"\n--- Starting detection evaluation ---")

    det_func_times = []
    accumulated_results_for_csv_per_dataset = []

    if vis_pdf_dir and vis_error_count is not None:
        # Prepare output directory for visualization (vis_pdf_dir is used directly)
        shutil.rmtree(vis_pdf_dir, ignore_errors=True)
        os.makedirs(vis_pdf_dir, exist_ok=True)
        if verbose: tqdm.write(
            f"Note: Visualization enabled. Visualized PDFs will be saved directly under '{vis_pdf_dir}'.")

    vis_output_dir = vis_pdf_dir  # Use this variable for convenience in the loop

    for dataset_name in tqdm(datasets_to_process_gt_side_sorted, desc=f"Processing datasets",
                             leave=False, disable=not verbose):
        gt_dataset_path = os.path.join(gt_dir, dataset_name)
        json_files = [f for f in os.listdir(gt_dataset_path) if f.endswith('.json')]
        # json_files = json_files[:100]

        if not json_files:
            # This case should technically be caught by the initial validation, but included for robustness
            if verbose: tqdm.write(
                f"Warning: No JSON files found in GT directory for dataset '{dataset_name}'. Skipping evaluation.")
            continue

        if class_names is None:
            current_dataset_class_metrics = defaultdict(
                lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0, 'det_count': 0})
        else:
            current_dataset_class_metrics = {name: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0, 'det_count': 0} for
                                             name in class_names}

        current_dataset_overall_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
        current_dataset_total_gt_boxes_count = 0
        current_dataset_total_det_boxes_count = 0

        for json_file in tqdm(json_files, desc=f"  {dataset_name} files", leave=False,
                              disable=not verbose):
            gt_filepath = os.path.join(gt_dataset_path, json_file)
            pdf_filename_only = json_file.replace('.json', '.pdf')
            original_pdf_filepath = None

            # Find PDF path and exit if not found
            for single_pdf_input_dir in pdf_dirs:
                potential_pdf_path = os.path.join(single_pdf_input_dir, pdf_filename_only)
                if os.path.exists(potential_pdf_path):
                    original_pdf_filepath = potential_pdf_path
                    break

            if original_pdf_filepath is None:
                print(
                    f"\nError: Original PDF '{pdf_filename_only}' not found in any of the provided 'pdf_dirs': {pdf_dirs}. Terminating evaluation.")
                sys.exit(1)

            # Load GT data
            try:
                with open(gt_filepath, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                gt_bboxes = gt_data.get('bboxes', [])
            except Exception as e:
                if verbose: tqdm.write(
                    f"Error: Could not parse GT JSON file '{json_file}'. Error: {e}. Skipping this file.")
                continue

            # Get detection results via det_func and record time
            start_time = time.time()
            try:
                # det_func call: only pdf_path is passed
                det_bboxes = det_func(original_pdf_filepath)
            except Exception as e:
                if verbose: tqdm.write(
                    f"Error calling det_func for '{json_file}' (PDF: {original_pdf_filepath}): {e}. Proceeding with empty detections.")
                det_bboxes = []
            finally:
                end_time = time.time()
                det_func_times.append(end_time - start_time)

            # Check det_func return value format and exit if incorrect
            if not isinstance(det_bboxes, list):
                print(
                    f"\nError: det_func must return a list. Returned type: {type(det_bboxes)}. Terminating evaluation.")
                sys.exit(1)
            for i, bbox in enumerate(det_bboxes):
                if not (isinstance(bbox, list) and len(bbox) == 5 and all(
                        isinstance(coord, (int, float)) for coord in bbox[:4]) and isinstance(bbox[4], str)):
                    print(
                        f"\nError: det_func list element {i} ('{bbox}') format incorrect. Must be [x1, y1, x2, y2, class_name]. Terminating evaluation.")
                    sys.exit(1)

            # --- ACCUMULATION & METRICS CALCULATION LOGIC ---

            # Increment per-class GT and Detection counts
            for gt_box_with_class in gt_bboxes:
                gt_class = gt_box_with_class[4]
                if class_names is None or gt_class in class_names:
                    current_dataset_class_metrics[gt_class]['gt_count'] += 1

            for d_box_with_class in det_bboxes:
                d_class = d_box_with_class[4]
                if class_names is None or d_class in class_names:
                    current_dataset_class_metrics[d_class]['det_count'] += 1

            current_dataset_total_gt_boxes_count += len(gt_bboxes)
            current_dataset_total_det_boxes_count += len(det_bboxes)

            file_tp = 0
            file_fp = 0
            file_fn = 0

            # --- Calculate TP/FP/FN for current file (used for visualization check and overall metrics) ---
            filtered_gt_bboxes_for_eval = [b for b in gt_bboxes if class_names is None or b[4] in class_names]
            filtered_det_bboxes_for_eval = [b for b in det_bboxes if class_names is None or b[4] in class_names]

            matched_gt_indices_for_file = set()

            for d_bbox_with_class in filtered_det_bboxes_for_eval:
                d_box = d_bbox_with_class[:4]
                d_class = d_bbox_with_class[4]
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt_box_with_class in enumerate(filtered_gt_bboxes_for_eval):
                    if gt_idx in matched_gt_indices_for_file:
                        continue
                    gt_box = gt_box_with_class[:4]
                    gt_class = gt_box_with_class[4]

                    if d_class == gt_class:
                        iou = calculate_iou(d_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                if best_iou >= iou_threshold:
                    file_tp += 1
                    matched_gt_indices_for_file.add(best_gt_idx)
                else:
                    file_fp += 1

            file_fn = len(filtered_gt_bboxes_for_eval) - len(matched_gt_indices_for_file)

            # --- Visualization Logic ---
            if vis_output_dir and vis_error_count is not None:
                # Saves directly to vis_output_dir (vis_pdf_dir)
                make_vis_pdf(original_pdf_filepath, gt_bboxes, det_bboxes,
                             vis_output_dir, iou_threshold, class_names)

            # --- ACCUMULATION (Global Logic - Accumulate TP/FP/FN to overall metrics) ---
            current_dataset_overall_metrics['tp'] += file_tp
            current_dataset_overall_metrics['fp'] += file_fp
            current_dataset_overall_metrics['fn'] += file_fn

            # Per-class accumulation requires re-running the matching
            matched_gt_indices_global_acc = set()
            for d_bbox_with_class in det_bboxes:
                d_box = d_bbox_with_class[:4]
                d_class = d_bbox_with_class[4]
                if class_names is not None and d_class not in class_names:
                    continue

                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt_box_with_class in enumerate(gt_bboxes):
                    if gt_idx in matched_gt_indices_global_acc:
                        continue
                    gt_box = gt_box_with_class[:4]
                    gt_class = gt_box_with_class[4]

                    if (class_names is None or gt_class in class_names) and (d_class == gt_class):
                        iou = calculate_iou(d_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                if best_iou >= iou_threshold:
                    current_dataset_class_metrics[d_class]['tp'] += 1
                    matched_gt_indices_global_acc.add(best_gt_idx)
                else:
                    current_dataset_class_metrics[d_class]['fp'] += 1

            for gt_idx, gt_bbox_with_class in enumerate(gt_bboxes):
                if gt_idx not in matched_gt_indices_global_acc:
                    gt_class = gt_bbox_with_class[4]
                    if class_names is None or gt_class in class_names:
                        current_dataset_class_metrics[gt_class]['fn'] += 1

        # --- After processing all JSON files for a dataset (Format Results) ---
        current_dataset_formatted_results = {}
        classes_to_evaluate_sorted = sorted(list(class_names)) if class_names is not None else sorted(
            list(current_dataset_class_metrics.keys()))

        current_dataset_formatted_results['__dataset_name__'] = dataset_name
        overall_tp = current_dataset_overall_metrics['tp']
        overall_fp = current_dataset_overall_metrics['fp']
        overall_fn = current_dataset_overall_metrics['fn']

        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (
                                                                                                                overall_precision + overall_recall) > 0 else 0.0

        current_dataset_formatted_results['__overall__'] = {
            'precision': round(overall_precision, 4),
            'recall': round(overall_recall, 4),
            'f1': round(overall_f1, 4),
            'gt_count': current_dataset_total_gt_boxes_count,
            'det_count': current_dataset_total_det_boxes_count,
            'tp': overall_tp,
            'fp': overall_fp,
            'fn': overall_fn
        }

        for class_name in classes_to_evaluate_sorted:
            metrics = current_dataset_class_metrics.get(class_name,
                                                        {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0, 'det_count': 0})
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            gt_count = metrics['gt_count']
            det_count = metrics['det_count']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            current_dataset_formatted_results[class_name] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'gt_count': gt_count,
                'det_count': det_count,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        accumulated_results_for_csv_per_dataset.append(current_dataset_formatted_results)

    # --- Save CSV Results (Conditional Saving) ---
    if result_csv_path:
        os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
        try:
            with open(result_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                for dataset_data in accumulated_results_for_csv_per_dataset:
                    dataset_name = dataset_data['__dataset_name__']
                    writer.writerow([f"- {dataset_name} (IoU >= {iou_threshold:.2f})"])
                    writer.writerow([])
                    writer.writerow(
                        ['Class', 'Precision', 'Recall', 'F1', 'GT Count', 'Det Count', 'TP Count', 'FP Count',
                         'FN Count'])

                    class_keys_for_rows = sorted(
                        [k for k in dataset_data.keys() if k not in ['__dataset_name__', '__overall__']])
                    for class_name in class_keys_for_rows:
                        metrics = dataset_data[class_name]
                        writer.writerow([
                            class_name, metrics['precision'], metrics['recall'], metrics['f1'],
                            metrics['gt_count'], metrics['det_count'], metrics['tp'],
                            metrics['fp'], metrics['fn']
                        ])

                    overall_metrics_output = dataset_data['__overall__']
                    writer.writerow([
                        'overall', overall_metrics_output['precision'], overall_metrics_output['recall'],
                        overall_metrics_output['f1'], overall_metrics_output['gt_count'],
                        overall_metrics_output['det_count'], overall_metrics_output['tp'],
                        overall_metrics_output['fp'], overall_metrics_output['fn']
                    ])
                    writer.writerow([])

                writer.writerow([])
                writer.writerow(['Execution Time'])

                det_func_time_stats_str = ""
                if det_func_times:
                    det_func_time_stats_str += f"Number of calls: {len(det_func_times)}\n"
                    det_func_time_stats_str += f"Minimum: {min(det_func_times):.4f} seconds\n"
                    det_func_time_stats_str += f"Maximum: {max(det_func_times):.4f} seconds\n"
                    det_func_time_stats_str += f"Average: {statistics.mean(det_func_times):.4f} seconds\n"
                    det_func_time_stats_str += f"Median: {statistics.median(det_func_times):.4f} seconds\n"
                    if len(det_func_times) > 1:
                        det_func_time_stats_str += f"Standard Deviation: {statistics.stdev(det_func_times):.4f} seconds\n"
                    else:
                        det_func_time_stats_str += f"Standard Deviation: N/A (single call)\n"

                if det_func_time_stats_str:
                    for line in det_func_time_stats_str.splitlines():
                        writer.writerow([line])
                else:
                    writer.writerow(["No det_func execution recorded."])
            if verbose: tqdm.write(f"Results saved to '{result_csv_path}'")
        except Exception as e:
            if verbose: tqdm.write(f"Error saving result CSV to '{result_csv_path}': {e}. Skipping saving.")

    elif verbose:
        tqdm.write("Note: 'result_csv_path' was not provided. Skipping CSV results saving.")

    return accumulated_results_for_csv_per_dataset


