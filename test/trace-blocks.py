"""
This script is a version of counters-blocks.py that includes performance
profiling using pyinstrument.
It processes a subset of the DocLayNet validation set, which can be specified
via command-line arguments. The profiling results are saved as HTML and JSON
files.
Make sure to only use only up to 200 files from the validation set, otherwise
the generated HTML report will be too large to be usable (dozens of GBs).

The profiled part of the code is wrapped between profiler.start() and
profiler.stop() calls.
"""

import json
import os
import sys
import time
from pathlib import Path

import pymupdf.layout
import pymupdf4llm
from pyinstrument import Profiler, renderers
from tqdm import tqdm

from compute_scores import compute_scores

profiler = Profiler()
json_renderer = renderers.JSONRenderer()

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
    for box in my_bboxes:
        bclass = box[4]
        if bclass == "table-fallback":
            bclass = "table"
        elif bclass == "fallback":
            bclass = "picture"
        if any(
            compute_iou(box[:4], gt_box[:4]) >= 0.6 and bclass == gt_box[4]
            for gt_box in gt_bboxes
        ):
            counts[bclass]["TP"] += 1
        else:
            counts[bclass]["FP"] += 1
    return counts


# this script should be run over a subset of the DocLayNet validation set
# because the generated HTML output of pyinstrument is too large otherwise.
def main(start=0, stop=200):
    counts = {
        "text": {"P": 0, "TP": 0, "FP": 0},
        "title": {"P": 0, "TP": 0, "FP": 0},
        "picture": {"P": 0, "TP": 0, "FP": 0},
        "table": {"P": 0, "TP": 0, "FP": 0},
        "list-item": {"P": 0, "TP": 0, "FP": 0},
        "page-header": {"P": 0, "TP": 0, "FP": 0},
        "page-footer": {"P": 0, "TP": 0, "FP": 0},
        "section-header": {"P": 0, "TP": 0, "FP": 0},
        "footnote": {"P": 0, "TP": 0, "FP": 0},
        "caption": {"P": 0, "TP": 0, "FP": 0},
        "formula": {"P": 0, "TP": 0, "FP": 0},
    }

    files = os.listdir(gt_json_dir)
    starttime = time.perf_counter()
    for fname in tqdm(files[start:stop]):
        js_file = os.path.join(gt_json_dir, fname)
        with open(js_file, "r") as f:
            gt_data = json.load(f)
        filename = os.path.join(gt_pdf_dir, gt_data["filename"])
        doc = pymupdf.open(filename)
        page = doc[0]
        gt_bboxes = gt_data["bboxes"]
        for b in gt_bboxes:
            counts[b[4]]["P"] += 1

        # only profile this statement
        profiler.start()
        parsed_doc = pymupdf4llm.parse_document(doc, use_ocr=False)
        profiler.stop()

        parsed_page = parsed_doc.pages[0]
        my_bboxes = []
        for box in parsed_page.boxes:
            my_bboxes.append([box.x0, box.y0, box.x1, box.y1, box.boxclass])
        counts = compare(my_bboxes, gt_bboxes, counts)
        doc.close()

    stoptime = time.perf_counter()
    duration = stoptime - starttime
    perfile = duration / (stop - start)
    print(
        f"Processed {stop - start} files in {round(duration)} seconds ({perfile:.3f} s/file)"
    )
    counts["Header"] = f"DocLayNet files [{start}:{stop}] seoul3-2"
    js = json.dumps(counts)
    json_path = __file__.replace(".py", f"-{start:03d}-{stop:03d}.json")
    Path(json_path).write_text(js)
    compute_scores(json_path)

    # Output the profiling results.abs
    # Caution: the HTML file can be very large if too many files are processed.
    profiler.write_html(Path(f"profile-report-{start:03d}-{stop:03d}-seoul3-2.html"))
    js = profiler.output(renderer=json_renderer)
    Path(f"profile-report-{start:03d}-{stop:03d}-seoul3-2.json").write_text(js)


if __name__ == "__main__":
    start = 0
    stop = 200

    if len(sys.argv) == 2:
        start = int(sys.argv[1])
        stop = start + 200
    elif len(sys.argv) >= 3:
        start = int(sys.argv[1])
        stop = int(sys.argv[2])
    if stop <= start:
        stop = start + 200
    main(start, stop)
