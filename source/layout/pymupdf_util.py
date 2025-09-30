import os
import numpy as np
import tempfile
import subprocess

import pymupdf

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



def create_input_data_from_page(page, input_type='pymupdf_text+imf', rbf_names=None,
                                features_path=None):
    data_dict = {
        'bboxes': [],
        'text': [],
        'custom_features': []
    }
    box_type = []

    if 'image' in input_type:
        rects = [itm["bbox"] for itm in page.get_image_info()]
        for rect in rects:
            x1 = max(0, rect[0])
            y1 = max(0, rect[1])
            x2 = max(0, rect[2])
            y2 = max(0, rect[3])
            data_dict['bboxes'].append([x1, y1, x2, y2])
            data_dict['text'].append('')
            box_type.append('image')

    if 'vector' in input_type:
        paths = [
            p for p in page.get_drawings() if p["rect"].width > 3 and p["rect"].height > 3
        ]
        vector_rects = page.cluster_drawings(drawings=paths)
        for vrect in vector_rects:
            x1 = vrect[0]
            y1 = vrect[1]
            x2 = vrect[2]
            y2 = vrect[3]
            data_dict['bboxes'].append([x1, y1, x2, y2])
            data_dict['text'].append('')
            box_type.append('vector')

    blocks = page.get_text("dict", flags=11)["blocks"]
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
            if txt != '':
                data_dict['bboxes'].append([x1, y1, x2, y2])
                data_dict['text'].append(txt)
                box_type.append('line')


    if rbf_names is None:
        rb_feat_value = []
        rb_feat_name = []
        rbf_names = []
    else:
        pass

    for row_idx in range(len(data_dict['bboxes'])):
        custom_feature = {}
        for f_idx, f_name in enumerate(rbf_names):
            if f_idx < 9:
                continue
            custom_feature[f_name] = rb_feat_value[row_idx][f_idx]

        is_vector = box_type[row_idx] == 'vector'
        is_image = box_type[row_idx] == 'image'

        custom_feature['is_vector'] = int(is_vector)
        custom_feature['is_image'] = int(is_image)
        custom_feature['is_line'] = int(not is_vector and not is_image)
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

        data_dict['custom_features'].append(custom_feature)

    if '+imf' in input_type:
        pix = page.get_pixmap()
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = bytes.reshape(pix.height, pix.width, pix.n)
        data_dict['image'] = page_img

    return data_dict


def create_input_data_by_pymupdf(pdf_path=None, document=None, page_no=0,
                                 input_type='pymupdf_text+imf', rbf_names=None, features_path=None):

    data_dict = {
        'bboxes': [],
        'text': [],
        'custom_features': []
    }
    box_type = []

    if document is None:
        if not os.path.exists(pdf_path):
            raise Exception(f'{pdf_path} is not exist!')
        doc = pymupdf.open(pdf_path)
        page = doc[page_no]
    else:
        doc = document
        page = doc[page_no]

    page_width, page_height = page.rect[2], page.rect[3]

    if 'image' in input_type:
        rects = [itm["bbox"] for itm in page.get_image_info()]
        for rect in rects:
            x1 = max(0, rect[0])
            y1 = max(0, rect[1])
            x2 = max(0, rect[2])
            y2 = max(0, rect[3])
            data_dict['bboxes'].append([x1, y1, x2, y2])
            data_dict['text'].append('')
            box_type.append('image')

    if 'vector' in input_type:
        paths = [
            p for p in page.get_drawings() if p["rect"].width > 3 and p["rect"].height > 3
        ]
        vector_rects = page.cluster_drawings(drawings=paths)
        for vrect in vector_rects:
            x1 = vrect[0]
            y1 = vrect[1]
            x2 = vrect[2]
            y2 = vrect[3]
            data_dict['bboxes'].append([x1, y1, x2, y2])
            data_dict['text'].append('')
            box_type.append('vector')

    blocks = page.get_text("dict", flags=11)["blocks"]
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

            if txt != '' and 0 <= x1 < x2 <= page_width and 0 <= y1 < y2 <= page_height:
                data_dict['bboxes'].append([x1, y1, x2, y2])
                data_dict['text'].append(txt)
                box_type.append('line')

    pdf_dir = os.path.dirname(pdf_path)
    file_name = os.path.basename(pdf_path)

    if features_path is not None:
        rb_feat_name, rb_feat_value = robins_features_extraction(features_path, pdf_dir, file_name, data_dict['bboxes'])
    else:
        rb_feat_name = []
        rb_feat_value = []

    for row_idx in range(len(data_dict['bboxes'])):
        custom_feature = {}
        for f_idx, f_name in enumerate(rb_feat_name):
            if f_idx < 9:
                continue
            custom_feature[f_name] = rb_feat_value[row_idx][f_idx]

        is_vector = box_type[row_idx] == 'vector'
        is_image = box_type[row_idx] == 'image'

        custom_feature['is_vector'] = int(is_vector)
        custom_feature['is_image'] = int(is_image)
        custom_feature['is_line'] = int(not is_vector and not is_image)

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
        data_dict['custom_features'].append(custom_feature)

    if '+imf' in input_type:
        page = doc[page_no]
        pix = page.get_pixmap()
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = bytes.reshape(pix.height, pix.width, pix.n)
        data_dict['image'] = page_img

    if document is None:
        doc.close()
    return data_dict
