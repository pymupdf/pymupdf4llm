import os
import copy
import concurrent.futures

import numpy as np
import tempfile
import subprocess

import pymupdf
import pymupdf.features as rf_features

from .common_util import compute_iou, resize_image, to_gray, extract_bboxes_from_segmentation

def robins_features_extraction(feature_path, pdf_dir, filename, rect_list):
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


def get_vector_lines(page, omit_invisible=True):
    """Return horizontal and vertical vector lines on a page.

    Optionally, omit vectors that have the color of the background.
    The background color is determined as the most frequent color on a pixel
    level as determined from a rasterized image (pixmap) of the page.
    """
    # determine background color:
    # disable anti-aliasing for more accurate color count
    # We assume the most frequent color to be the background.
    pymupdf.TOOLS.set_aa_level(0)
    pix = page.get_pixmap()
    portion, color = pix.color_topusage()
    # sRGB integer represents an RGB color tripel.
    sRGB = color[0] << 16 | color[1] << 8 | color[2]
    # convert to a tripel of three floats in 0..1 range.
    bg_color = pymupdf.sRGB_to_pdf(sRGB)
    # print(f"Most used color: {bg_color=} in {portion*100:.2f}% of pixels")

    # all potentially significant paths
    paths = [
        p for p in page.get_drawings() if p["rect"].height > 5 or p["rect"].width > 5
    ]

    h_lines = []
    v_lines = []

    for p in paths:
        if (
            1
            and omit_invisible
            and (
                p["fill"] == bg_color
                and p["color"] in (None, bg_color)
                or p["color"] == bg_color
                and p["fill"] in (None, bg_color)
            )
        ):
            continue
        # total rectangle already is "line-like" ...
        if p["rect"].height < 3 and p["rect"].width > 30:
            h_lines.append(p["rect"])
            continue
        elif p["rect"].width < 3 and p["rect"].height > 30:
            v_lines.append(p["rect"])
            continue

        # cover other cases by looking at single draw commands (items)
        # ".normalize()" is required because invalid rectangles are not drawn
        for item in p["items"]:
            # Handle line items. Must be axis-parallel
            if item[0] == "l":  # a line
                p1, p2 = item[1:]
                if abs(p1.y - p2.y) <= 1 and abs(p2.x - p1.x) > 30:  # horizontal line
                    h_lines.append(pymupdf.Rect(p1, p2).normalize())
                elif abs(p1.x - p2.x) <= 1 and abs(p2.y - p1.y) > 30:  # vertical line
                    v_lines.append(pymupdf.Rect(p1, p2).normalize())
                continue
            if item[0] != "re":  # skip non-rectangles
                continue
            r = item[1]  # the rectangle of this item
            h_lines.append(pymupdf.Rect(r.tl, r.tr).normalize())  # top horizontal
            h_lines.append(pymupdf.Rect(r.bl, r.br).normalize())  # bottom horizontal
            v_lines.append(pymupdf.Rect(r.tl, r.bl).normalize())  # left vertical
            v_lines.append(pymupdf.Rect(r.tr, r.br).normalize())  # right vertical

    return h_lines, v_lines


def get_picture_clusters(page):
    """Identify and group connected vectors on a page.

    Notes:
        This function uses PyMuPDF's Page method cluster drawings to find
        neighbored vectors. Its main purpose is to limit the number of vectors
        that must be considered by cluster drawings as this currently has a
        performance of quadratic complexity O(n^2) in relation to the
        number of vectors.

    Args:
        page (pymupdf.Page): The page to analyze.

    Returns:
        A list of pymupdf.Rect objects representing bounding boxes of
        connected vector groups.
    """

    page_rect = page.rect

    # preprocess 1: exclude point-like paths and sort descending by area size:
    # t0 = mt()
    paths = [page_rect & p["rect"] for p in page.get_drawings()]
    # t1 = mt()
    # preprocess 2: extract images intersecting the page and make a common
    images = [page_rect & img["bbox"] for img in page.get_image_info()]
    # t2 = mt()

    # combine both and sort descending by area size:
    all_pictures = sorted(
        [{"rect": r} for r in paths + images if r.width >= 3 or r.height >= 3],
        key=lambda r: r["rect"].width * r["rect"].height,
        reverse=True,
    )
    # preprocess 3: swallow mutual containments, only looking at the largest.
    # t3 = mt()
    filtered_paths = []
    for p in all_pictures[:1000]:
        if not any(p["rect"] in f["rect"] for f in filtered_paths):
            filtered_paths.append(p)
    # t4 = mt()
    # now cluster the remaining rectangles -
    # hopefully avoiding performance hits
    # print(f"path extraction time: {t1 - t0:.3f} sec")
    # print(f"image extraction time: {t2 - t1:.3f} sec")
    # print(f"rectangle sorting/selecting time: {t3 - t2:.3f} sec")
    # print(f"containment filtering time: {t4 - t3:.3f} sec")
    return page.cluster_drawings(drawings=filtered_paths)


def extract_rf_features(data_dict, stext_page):
    for row_idx, bbox in enumerate(data_dict['bboxes']):
        region = pymupdf.mupdf.FzRect(bbox[0], bbox[1], bbox[2], bbox[3])
        features = rf_features.fz_features_for_region(stext_page, region, 0)
        for name in dir(features):
            if not name.startswith('_') and name not in ['this', 'thisown', 'last_char_rect']:
                data_dict['custom_features'][row_idx][name] = getattr(features, name)


def merge_lines(lines, orientation='h', tolerance=3):
    """
    Merge fragmented lines into longer continuous lines.

    lines: list of line objects with x0, y0, x1, y1 attributes
    orientation: 'h' for horizontal, 'v' for vertical
    tolerance: allowable gap or offset to consider lines as mergeable
    """
    # Sort lines by position: horizontal by y, vertical by x
    if orientation == 'h':
        lines.sort(key=lambda r: (r.y0, r.x0))
    else:
        lines.sort(key=lambda r: (r.x0, r.y0))

    merged = []
    for line in lines:
        x0, y0, x1, y1 = line.x0, line.y0, line.x1, line.y1
        matched = False

        for m in merged:
            if orientation == 'h':
                # Check if y is close and x ranges overlap or touch
                if abs(m.y0 - y0) <= tolerance and not (x1 < m.x0 - tolerance or x0 > m.x1 + tolerance):
                    m.x0 = min(m.x0, x0)
                    m.x1 = max(m.x1, x1)
                    matched = True
                    break
            else:
                # Check if x is close and y ranges overlap or touch
                if abs(m.x0 - x0) <= tolerance and not (y1 < m.y0 - tolerance or y0 > m.y1 + tolerance):
                    m.y0 = min(m.y0, y0)
                    m.y1 = max(m.y1, y1)
                    matched = True
                    break

        if not matched:
            # No match found, add as new merged line
            merged.append(copy.deepcopy(line))

    return merged


def merge_boxes(boxes, iou_threshold=0.5):
    merged = []
    while boxes:
        base = boxes.pop(0)
        has_merged = False
        for i, other in enumerate(boxes):
            iou = compute_iou(base, other)
            if iou > iou_threshold:
                new_box = [
                    min(base[0], other[0]),
                    min(base[1], other[1]),
                    max(base[2], other[2]),
                    max(base[3], other[3])
                ]
                boxes[i] = new_box
                has_merged = True
                break
        if not has_merged:
            merged.append(base)
    return merged


def image_feature_extraction_task(page_img, feature_extractor, input_type):
    page_h, page_w, _ = page_img.shape

    input_shape = feature_extractor.get_inputs()[0].shape
    target_height = input_shape[2]
    target_width = input_shape[3]

    img_resized = resize_image(page_img, (target_width, target_height))
    img_gray = to_gray(img_resized)

    img_gray = img_gray.astype(np.float32)
    min_val, max_val = img_gray.min(), img_gray.max()
    if max_val > min_val:
        img_gray = (img_gray - min_val) / (max_val - min_val)
    else:
        img_gray = np.zeros_like(img_gray, dtype=np.float32)

    nn_input = img_gray.astype(np.float32)
    nn_input = np.expand_dims(nn_input, axis=0)
    nn_input = np.expand_dims(nn_input, axis=0)

    ort_inputs = {feature_extractor.get_inputs()[0].name: nn_input}
    ort_outputs = feature_extractor.run(None, ort_inputs)[0]

    feature_map = ort_outputs
    bboxes_to_add = []
    box_types_to_add = []

    if 'seg-image' in input_type:
        class_names = ['background', 'text', 'title', 'picture', 'table', 'list-item', 'page-header', 'page-footer',
                       'section-header', 'footnote', 'caption', 'formula']

        prior_det = extract_bboxes_from_segmentation(ort_outputs, class_names=class_names,
                                                     min_component_area=50, morphology_kernel_size=3)
        resize_x = page_w / target_width
        resize_y = page_h / target_height
        for det in prior_det:
            if det['class'] == 'picture' and det['score'] > 0.8:
                bbox = [
                    det['bbox'][0] * resize_x, det['bbox'][1] * resize_y,
                    det['bbox'][2] * resize_x, det['bbox'][3] * resize_y
                ]
                bboxes_to_add.append(bbox)
                box_types_to_add.append('image')

    return feature_map, bboxes_to_add, box_types_to_add


def create_input_data_from_page(page, input_type=('text',), feature_set_name='rf+imf',
                                max_image_num=500, max_vec_line_num=200,
                                feature_extractor=None):
    data_dict = {
        'bboxes': [],
        'text': [],
        'custom_features': [],
        'page_width': 0,
        'page_height': 0,
    }
    box_type = []
    default_flags = (
            0
            | pymupdf.TEXT_PRESERVE_WHITESPACE
            | pymupdf.TEXT_PRESERVE_LIGATURES
            | pymupdf.TEXT_INHIBIT_SPACES
            | pymupdf.TEXT_ACCURATE_BBOXES
    )
    page_width, page_height = page.rect[2], page.rect[3]
    data_dict['page_width'] = page_width
    data_dict['page_height'] = page_height

    pix = page.get_pixmap()
    bytes = np.frombuffer(pix.samples, dtype=np.uint8)
    page_img = bytes.reshape(pix.height, pix.width, pix.n)
    data_dict['image'] = page_img

    if feature_extractor is not None:
        try:
            feature_map, bboxes_to_add, box_types_to_add = image_feature_extraction_task(page_img,
                                                                                         feature_extractor,
                                                                                         input_type)
            data_dict['feature_map'] = feature_map
            for bbox, type_val in zip(bboxes_to_add, box_types_to_add):
                if bbox not in data_dict['bboxes']:
                    data_dict['bboxes'].append(bbox)
                    data_dict['text'].append('')
                    box_type.append(type_val)

        except Exception as exc:
            print(f"Error in image feature extraction : {exc}")

    if 'image' in input_type:
        img_bboxes = [itm["bbox"] for itm in page.get_image_info()]
        img_bboxes = merge_boxes(img_bboxes, iou_threshold=0.5)
        img_bboxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
        img_bboxes = img_bboxes[:max_image_num]
        for rect in img_bboxes:
            x1 = max(0, rect[0])
            y1 = max(0, rect[1])
            x2 = max(0, rect[2])
            y2 = max(0, rect[3])
            if 0 <= x1 < x2 <= page_width and 0 <= y1 <= y2 <= page_height:
                bbox = [x1, y1, x2, y2]
                if bbox not in data_dict['bboxes']:
                    data_dict['bboxes'].append(bbox)
                    type_val = 'image'
                    data_dict['text'].append('')
                    box_type.append(type_val)

    if 'picture_clusters' in input_type:
        pic_bboxes = get_picture_clusters(page)
        pic_bboxes = sorted(
            pic_bboxes,
            key=lambda r: r.width * r.height,
            reverse=True
        )[:max_image_num]

        for rect in pic_bboxes:
            x1 = rect.x0
            y1 = rect.y0
            x2 = rect.x1
            y2 = rect.y1
            if 0 <= x1 < x2 <= page_width and 0 <= y1 <= y2 <= page_height:
                if y1 == y2:
                    y2 = y1 + 1
                bbox = [x1, y1, x2, y2]
                if bbox not in data_dict['bboxes']:
                    data_dict['bboxes'].append(bbox)
                    type_val = 'image'
                    data_dict['text'].append('')
                    box_type.append(type_val)

    if 'vec_line' in input_type:
        # all potentially significant paths
        h_lines, v_lines = get_vector_lines(page, omit_invisible=True)
        h_lines = merge_lines(h_lines, orientation='h', tolerance=3)
        v_lines = merge_lines(v_lines, orientation='v', tolerance=3)

        h_lines = sorted(
            h_lines,
            key=lambda r: r.width * r.height,
            reverse=True
        )[:max_vec_line_num]

        for rect in h_lines:
            x1 = rect.x0
            y1 = rect.y0
            x2 = rect.x1
            y2 = rect.y1
            if 0 <= x1 < x2 <= page_width and 0 <= y1 <= y2 <= page_height:
                if y1 == y2:
                    y2 = y1 + 1
                bbox = [x1, y1, x2, y2]
                if bbox not in data_dict['bboxes']:
                    data_dict['bboxes'].append(bbox)
                    type_val = 'h-line-vector'
                    data_dict['text'].append('')
                    box_type.append(type_val)

        v_lines = sorted(
            v_lines,
            key=lambda r: r.width * r.height,
            reverse=True
        )[:max_vec_line_num]
        for rect in v_lines:
            x1 = rect.x0
            y1 = rect.y0
            x2 = rect.x1
            y2 = rect.y1
            if 0 <= x1 <= x2 <= page_width and 0 <= y1 < y2 <= page_height:
                if x1 == x2:
                    x2 = x1 + 1
                bbox = [x1, y1, x2, y2]
                if bbox not in data_dict['bboxes']:
                    data_dict['bboxes'].append(bbox)
                    type_val = 'v-line-vector'
                    data_dict['text'].append('')
                    box_type.append(type_val)

    if 'text' in input_type:
        blocks = page.get_text("dict", flags=default_flags)["blocks"]
        for block in blocks:
            for line in block["lines"]:
                x1 = line['bbox'][0]
                y1 = line['bbox'][1]
                x2 = line['bbox'][2]
                y2 = line['bbox'][3]

                txt = []
                for span in line['spans']:
                    txt.append(span['text'])

                txt = ' '.join(txt).strip()

                # Filter Empty text
                if txt != '' and 0 <= x1 < x2 <= page_width and 0 <= y1 < y2 <= page_height:
                    bbox = [x1, y1, x2, y2]
                    if bbox not in data_dict['bboxes']:
                        data_dict['bboxes'].append(bbox)
                        data_dict['text'].append(txt)
                        box_type.append('text')

    checkboxes = [(widget.rect, widget.field_value) for widget in page.widgets() if
                  widget.field_type == pymupdf.PDF_WIDGET_TYPE_CHECKBOX]
    for bbox, value in checkboxes:
        x1 = bbox.x0
        y1 = bbox.y0
        x2 = bbox.x1
        y2 = bbox.y1
        if 0 <= x1 < x2 <= page_width and 0 <= y1 < y2 <= page_height:
            bbox = [x1, y1, x2, y2]
            if bbox not in data_dict['bboxes']:
                data_dict['bboxes'].append(bbox)
                type_val = 'check-box'
                data_dict['text'].append('')
                box_type.append(type_val)

    # Add 'custom-features'
    for row_idx in range(len(data_dict['bboxes'])):
        custom_feature = {
            'box_type': box_type[row_idx]
        }
        num_count = 0
        text = data_dict['text'][row_idx]
        if len(text) > 0:
            for c in text:
                if c.isdigit():
                    num_count += 1
            num_ratio = num_count / len(text)
        else:
            num_ratio = 0.0
        custom_feature['num_ratio'] = num_ratio

        is_text = is_image = is_hline_vector = is_vline_vector = 0
        if box_type[row_idx] == 'text':
            is_text = 1
        elif box_type[row_idx] == 'image':
            is_image = 1
        elif box_type[row_idx] == 'h-line-vector':
            is_hline_vector = 1
        elif box_type[row_idx] == 'v-line-vector':
            is_vline_vector = 1

        custom_feature['is_text'] = is_text
        custom_feature['is_image'] = is_image
        custom_feature['is_hline_vector'] = is_vline_vector
        custom_feature['is_vline_vector'] = is_vline_vector

        custom_feature['is_line'] = is_text  # old name
        custom_feature['is_vector'] = is_hline_vector + is_vline_vector  # old name
        data_dict['custom_features'].append(custom_feature)

    if 'rf' in feature_set_name:
        stext_flags = (
                0
                | pymupdf.TEXT_PRESERVE_WHITESPACE
                | pymupdf.TEXT_PRESERVE_LIGATURES
                | pymupdf.TEXT_INHIBIT_SPACES
                | pymupdf.TEXT_ACCURATE_BBOXES
                | pymupdf.TEXT_COLLECT_VECTORS
        )
        stext_page = page.get_textpage(flags=stext_flags)
        extract_rf_features(data_dict, stext_page)

    for row_idx in range(len(data_dict['bboxes'])):
        data_dict['custom_features'][row_idx]['box_type'] = box_type[row_idx]
        num_count = 0
        text = data_dict['text'][row_idx]
        if len(text) > 0:
            for c in text:
                if c.isdigit():
                    num_count += 1
            num_ratio = num_count / len(text)
        else:
            num_ratio = 0.0
        data_dict['custom_features'][row_idx]['num_ratio'] = num_ratio

    return data_dict


def create_input_data_by_pymupdf(pdf_path=None, document=None, page_no=0,
                                 input_type=('text',), feature_extractor=None):
    if document is None:
        if not os.path.exists(pdf_path):
            raise Exception(f'{pdf_path} is not exist!')
        doc = pymupdf.open(pdf_path)
        page = doc[page_no]
    else:
        doc = document
        page = doc[page_no]

    data_dict = create_input_data_from_page(page, input_type, feature_extractor=feature_extractor)
    data_dict['file_path'] = pdf_path

    if document is None:
        doc.close()
    return data_dict
