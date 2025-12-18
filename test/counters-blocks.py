"""
Script to evaluate block detection performance on DocLayNet validation set
using pymupdf4llm for block detection and IoU for matching with ground truth.

Note 1:
- Ground truth JSON files are expected to be in `gt_json_dir`.
- Corresponding PDF files are expected to be in `gt_pdf_dir`.
Adjust as necessary.

Note 2:
The execution time per file/page is practically equal to the time taken by
Markdown, plain text or JSON extraction times of pymupdf4llm.

Note 3:
The script creates a JSON file with counts and computes precision, recall,
and F1-score. It then calls `compute_scores` from `compute_scores.py` to display
the results as a nice table.
The script compute_scores.py can also be called standalone.
The "title" key in the json dictionary can be used to provide a description.
It will be used as the header line for the produced scoring table.
"""

import json
import os
import sys
import time
from pathlib import Path

import pymupdf.layout
import pymupdf4llm
from tqdm import tqdm

from compute_scores import compute_scores  # score table generator

print(f"{pymupdf.version=}, {pymupdf4llm.version=}")

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

gt_json_dir = get_script_path() + "/../eval/resources/gt//DoclayNet_val"
gt_pdf_dir = get_script_path() + "/../../datasets/DocLayNet/PDF"


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Each box is represented by [xmin, ymin, xmax, ymax].
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


def compare(my_bboxes, gt_bboxes, counts):
    """
    Compares detected bounding boxes with ground truth bounding boxes
    and updates counts for true positives (TP) and false positives (FP).
    """
    for box in my_bboxes:
        bclass = box[4]  # special adjustments for table bbox handling
        if bclass == "fallback":  # 4llm versions < 0.2.7
            bclass = "picture"
        elif bclass == "table-fallback":  # 4llm versions>=0.2.7
            bclass = "table"
        if any(
            compute_iou(box[:4], gt_box[:4]) >= 0.6 and bclass == gt_box[4]
            for gt_box in gt_bboxes
        ):
            counts[bclass]["TP"] += 1
        else:
            counts[bclass]["FP"] += 1
    return counts


# collect all counts herein
counts = {
    "text": {"P": 0, "TP": 0, "FP": 0},
    "picture": {"P": 0, "TP": 0, "FP": 0},
    "table": {"P": 0, "TP": 0, "FP": 0},
    "list-item": {"P": 0, "TP": 0, "FP": 0},
    "page-header": {"P": 0, "TP": 0, "FP": 0},
    "page-footer": {"P": 0, "TP": 0, "FP": 0},
    "section-header": {"P": 0, "TP": 0, "FP": 0},
    "footnote": {"P": 0, "TP": 0, "FP": 0},
    "caption": {"P": 0, "TP": 0, "FP": 0},
    "formula": {"P": 0, "TP": 0, "FP": 0},
    "title": {"P": 0, "TP": 0, "FP": 0},
}

files = os.listdir(gt_json_dir)
starttime = time.perf_counter()
for fname in tqdm(files):  # process the files with a progress bar
    js_file = os.path.join(gt_json_dir, fname)
    with open(js_file, "r") as f:
        gt_data = json.load(f)
    filename = os.path.join(gt_pdf_dir, gt_data["filename"])
    doc = pymupdf.open(filename)
    page = doc[0]
    gt_bboxes = gt_data["bboxes"]
    # prefill the GT total number counts
    for b in gt_bboxes:
        counts[b[4]]["P"] += 1
    # process the page and extract all information (text, images, tables, ...)
    parsed_doc = pymupdf4llm.parse_document(doc, use_ocr=False)
    # look at details for the first page only (DocLayNet: 1-page PDFs)
    parsed_page = parsed_doc.pages[0]
    my_bboxes = []

    # extract the layout boundary boxes resulting from 4llm's post-processing
    for box in parsed_page.boxes:
        my_bboxes.append([box.x0, box.y0, box.x1, box.y1, box.boxclass])
    counts = compare(my_bboxes, gt_bboxes, counts)

    # some housekeeping
    doc.close()
    page = None
    parsed_doc = None
    parsed_page = None

stoptime = time.perf_counter()
duration = stoptime - starttime
perfile = duration / len(files)
print(
    f"Processed {len(files)} files in {round(duration)} seconds ({perfile:.3f} s/file)"
)

# put any title in the JSON file
counts["Header"] = f"DocLayNet {len(files)} files, complete metrics O+O-revised"
js = json.dumps(counts)
json_path = __file__.replace(".py", "-O+O-revised.json")
# write JSON file and call table generating script.
Path(json_path).write_text(js)
compute_scores(json_path)
