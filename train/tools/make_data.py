import pickle

import os
import sys
import json
import time
import random
import hashlib

import blosc
import tqdm

import cv2
import re
import numpy as np
import shutil
import xml.etree.ElementTree as ET
import subprocess
import multiprocessing as mp
import tempfile

import lmdb
import torch
import pymupdf
import pymongo
import gridfs


import fitz
import anyconfig

import threading

from pymupdf import mupdf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import onnxruntime as ort
from train.tools.data.layout.DocumentJsonDataset import DocumentJsonDataset

_illegal_xml_chars_RE = re.compile(u'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]')

def log(msg):
    time_str = datetime.now().strftime("%y-%m-%d %H:%M:%S")
    with open('ymp/result/test.log', 'a') as f:
        f.write(f'[{time_str}] {msg}\n')

def escape_xml_illegal_chars(val, replacement='?'):
    return _illegal_xml_chars_RE.sub(replacement, val)


def make_mupdf_xml(cfg):
    pdf_dir = cfg['make_mupdf_xml']['pdf_dir']
    save_dir = cfg['make_mupdf_xml']['save_dir']
    pkl_dir = cfg['make_mupdf_xml']['pkl_dir']
    mutool_path = cfg['make_mupdf_xml']['mutool_path']

    valid_names = []
    if pkl_dir is not None:
        for file_name in os.listdir(pkl_dir):
            valid_names.append(file_name[:-4])

    # shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    file_list = []
    if pdf_dir.endswith('/'):
        for sub_idx, sub_dir in enumerate(os.listdir(pdf_dir)):
            for file_idx, file_name in enumerate(os.listdir(f'{pdf_dir}/{sub_dir}')):
                file_list.append([f'{pdf_dir}{sub_dir}', file_name])
    else:
        for file_idx, file_name in enumerate(os.listdir(pdf_dir)):
            file_list.append([pdf_dir, file_name])

    # file_list = file_list[:100]
    for file_idx, (pdf_sub_dir, file_name) in enumerate(file_list):
        if file_idx % 100 == 0:
            print('%d/%d ...' % (file_idx, len(file_list)))

        if not file_name.lower().endswith('.pdf'):
            print('Invalid file - %s' % file_name)
            continue

        if len(valid_names) > 0 and file_name[:-4] not in valid_names:
            continue

        # if file_name != '0c213fb95d00df08ea8d29f498d6ed414521eb6ad4207d77deba3e53412bb7d2.pdf':
        #     continue

        save_path = '%s/%s.stext' % (save_dir, file_name[:-4])

        if os.path.exists(save_path):
            continue

        file_path = '%s/%s' % (pdf_sub_dir, file_name)
        # cmd = f'mutool convert -o {save_path} -Ostructured,vectors,segment,table-hunt,collect-style,clip,accurate-bboxes {file_path}'
        # cmd = f'/home/youbit/bin/mupdf-master-250317/build/release/mutool convert -o {save_path} -Ostructured,segment,collect-styles,clip,accurate-bboxes {file_path}'
        cmd = f'{mutool_path} convert -o {save_path} -Ostructured,segment,vectors,images,collect-styles,clip {file_path}'
        # print(cmd)
        os.system(cmd)


def make_doclaynet_json(cfg):
    prefix = cfg['make_doclaynet_json']['prefix']
    pdf_dir = cfg['make_doclaynet_json'][prefix + '_pdf_dir']
    img_dir = cfg['make_doclaynet_json'][prefix + '_img_dir']
    json_dir = cfg['make_doclaynet_json'][prefix + '_json_dir']

    xml_dir = cfg['make_doclaynet_json']['mupdf_xml_dir']
    ts_dir = cfg['make_doclaynet_json']['textsense_json_dir']
    save_dir = cfg['make_doclaynet_json']['pdf_json_save_dir']

    st_idx = cfg['make_doclaynet_json']['start_index']
    ed_idx = cfg['make_doclaynet_json']['end_index']
    empty_save_dir = cfg['make_doclaynet_json']['do_empty_dir']

    show_result = cfg['make_doclaynet_json']['show_result']

    task = cfg['make_doclaynet_json']['task_type']
    task_option = cfg['make_doclaynet_json']['task_option']
    errors = []

    print('Task: %s' % task)
    print('Task Option : %s' % str(task_option))

    if empty_save_dir and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    file_list = os.listdir(pdf_dir)
    file_list.sort()

    if ed_idx == 0:
        ed_idx = len(file_list)

    for file_idx, file_name in enumerate(file_list):
        if file_idx < st_idx:
            continue
        if file_idx >= ed_idx:
            break
        if file_idx % 100 == 0:
            print('%d/%d...' % (file_idx, ed_idx))

        # print(file_name)
        # if file_name[:-4] != '4b676e935d31ab4ad44c52fb613a92c1bddcd2dcbca2aaf24e967c5aabb62974':
        #     continue

        xml_file = '%s/%s.stext' % (xml_dir, file_name[:-4])
        if task.startswith('mutool') and not os.path.exists(xml_file):
            # errors.append([file_name, 'not exist'])
            print('[%s] is not exist' % xml_file)
            continue

        if json_dir is not None:
            with open(f'{json_dir}/{file_name[:-4]}.json', 'r') as f:
                json_data = json.load(f)
        else:
            json_data = {
                'metadata': {
                    'original_filename': 'unknown',
                    'original_width': 0,
                    'original_height': 0,
                    'coco_width': 0,
                    'coco_height': 0,
                    'collection': 'unknown',
                    'doc_category': 'paper'
                },
                'cells': [],
            }

        if '' == 'D' and img_dir != '':
            img = cv2.imread(f'{img_dir}/{file_name[:-4]}.png')
            img_h, img_w, _ = img.shape
            disp = img.copy()
            json_data['metadata']['original_filename'] = file_name
            for cell_idx, cell in enumerate(json_data['cells']):
                bbox = list(map(int, cell['bbox']))
                cv2.rectangle(disp, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 1)
            cv2.imshow('Original JSON', disp)

        json_data['metadata']['original_filename'] = file_name
        try:
            doc = pymupdf.open(f'{pdf_dir}/{file_name}')
            page = doc[0]
            json_data['metadata']['original_width'] = page.rect.width
            json_data['metadata']['original_height'] = page.rect.height
            # resize_x = img.shape[1] / page.rect.width
            # resize_y = img.shape[0] / page.rect.height
            pix = page.get_pixmap()
            bytes = np.frombuffer(pix.samples, dtype=np.uint8)
            page_img = bytes.reshape(pix.height, pix.width, pix.n)
            page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)

        except Exception as ex:
            print(ex)
            continue
        resize_x = 1.0
        resize_y = 1.0

        new_cells = []
        # PyMuPDF extraction result
        if task == 'PyMuPDF':
            disp = page_img.copy()

            # Image bbox extraction
            if 'image' in task_option:
                rects = [itm["bbox"] for itm in page.get_image_info()]
                for rect in rects:
                    x1 = max(0, rect[0] * resize_x)
                    y1 = max(0, rect[1] * resize_y)
                    x2 = max(0, rect[2] * resize_x)
                    y2 = max(0, rect[3] * resize_y)
                    if 'exclude_text_in_image' in task_option:
                        image_rects.append([x1, y1, x2, y2])
                    new_cells.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'text': '',
                        'type': 'image',
                    })
                    cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

            # Vector bbox extraction
            if 'vector' in task_option:
                paths = [
                    p for p in page.get_drawings() if p["rect"].width > 3 and p["rect"].height > 3
                ]
                vector_rects = page.cluster_drawings(drawings=paths)
                for vrect in vector_rects:
                    x1 = vrect[0] * resize_x
                    y1 = vrect[1] * resize_y
                    x2 = vrect[2] * resize_x
                    y2 = vrect[3] * resize_y

                    if 'exclude_text_in_vector' in task_option:
                        image_rects.append([x1, y1, x2, y2])
                    new_cells.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'text': '',
                        'type': 'vector',
                    })
                    cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

            blocks = page.get_text("dict", flags=11)["blocks"]
            for block in blocks:
                for line in block["lines"]:
                    x1 = line['bbox'][0] * resize_x
                    y1 = line['bbox'][1] * resize_y
                    x2 = line['bbox'][2] * resize_x
                    y2 = line['bbox'][3] * resize_y

                    txt = []
                    for span in line['spans']:
                        txt.append(span['text'])
                    txt = ' '.join(txt).strip()
                    new_cells.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'text': txt
                    })
                    cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

            if show_result:
                disp = cv2.resize(disp, (0, 0), fx=0.7, fy=0.7)
                cv2.imshow('PyMuPDF JSON', disp)
                cv2.waitKey()

        # PyMuPDF, but with mupdf flags
        if task == 'MuPDF':
            flags = (
                    0
                    | mupdf.FZ_STEXT_ACCURATE_BBOXES
                    | mupdf.FZ_STEXT_SEGMENT
                    | mupdf.FZ_STEXT_COLLECT_STYLES
                    | mupdf.FZ_STEXT_COLLECT_STRUCTURE
                    | mupdf.FZ_STEXT_CLIP
            )

            disp = page_img.copy()
            target = 'line'

            try:
                front = """<?xml version="1.0"?>\n<document>\n"""
                back = """\n</document>"""
                xml_string = front + page.get_text("xml", flags=flags) + back
                xml_string = escape_xml_illegal_chars(xml_string, '')
                xml_string = xml_string.replace('c="&#', 'c="')
                doc_root = ET.fromstring(xml_string)

                page_nodes = doc_root.findall('page')
                for page_idx, page_node in enumerate(page_nodes):
                    page_width = float(page_node.get('width'))
                    page_height = float(page_node.get('height'))

                    image_rects = []
                    page = doc[page_idx]

                    # Image bbox extraction
                    if 'image' in task_option:
                        rects = [itm["bbox"] for itm in page.get_image_info()]
                        for rect in rects:
                            x1 = max(0, rect[0] * resize_x)
                            y1 = max(0, rect[1] * resize_y)
                            x2 = max(0, rect[2] * resize_x)
                            y2 = max(0, rect[3] * resize_y)
                            if 'exclude_text_in_image' in task_option:
                                image_rects.append([x1, y1, x2, y2])
                            new_cells.append({
                                'bbox': (x1, y1, x2 - x1, y2 - y1),
                                'text': '',
                                'type': 'image',
                            })
                            cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

                    # Vector bbox extraction
                    if 'vector' in task_option:
                        for vector in page_node.iter('vector'):
                            vector_bbox = vector.get('bbox').split()
                            vector_bbox = list(map(float, vector_bbox))
                            vector_txt = ''
                            x1 = vector_bbox[0] * resize_x
                            y1 = vector_bbox[1] * resize_y
                            x2 = vector_bbox[2] * resize_x
                            y2 = vector_bbox[3] * resize_y

                            if x2 - x1 < 1 or y2 - y1 < 1:
                                continue

                            if 0 <= x1 < x2 <= disp.shape[1] and 0 <= y1 < y2 <= disp.shape[0]:
                                new_cells.append({
                                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                                    'text': vector_txt,
                                    'type': 'vector',
                                })
                                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

                    for block in page_node.iter('block'):
                        block_bbox = block.get('bbox').split()
                        block_bbox = list(map(float, block_bbox))
                        block_txt = ''

                        for line in block.iter('line'):
                            line_bbox = line.get('bbox').split()
                            line_bbox = list(map(float, line_bbox))
                            line_txt = ''
                            for font in line.iter('font'):
                                for c in font.iter('char'):
                                    line_txt += c.get('c')

                            line_txt = line_txt.strip()
                            if line_txt == '':
                                continue

                            if target == 'line':
                                x1 = line_bbox[0] * resize_x
                                y1 = line_bbox[1] * resize_y
                                x2 = line_bbox[2] * resize_x
                                y2 = line_bbox[3] * resize_y

                                if x2 - x1 < 1 or y2 - y1 < 1:
                                    continue

                                # Exclude text elements in images
                                if 'exclude_text_in_image' in task_option:
                                    is_in_image = False
                                    for irect in image_rects:
                                        i_x1, i_y1, i_x2, i_y2 = irect
                                        if i_x1 <= x1 < x2 <= i_x2 and i_y1 <= y1 < y2 <= i_y2:
                                            is_in_image = True
                                            break
                                    if is_in_image:
                                        continue

                                if 0 <= x1 < x2 <= disp.shape[1] and 0 <= y1 < y2 <= disp.shape[0]:
                                    new_cells.append({
                                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                                        'text': line_txt,
                                        'type': 'line',
                                    })
                                    cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

                        if target == 'block':
                            x1 = block_bbox[0] * resize_x
                            y1 = block_bbox[1] * resize_y
                            x2 = block_bbox[2] * resize_x
                            y2 = block_bbox[3] * resize_y
                            if 0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h:
                                new_cells.append({
                                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                                    'text': block_txt,
                                    'type': 'block',
                                })
                                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

            except Exception as ex:
                errors.append([file_name, str(ex)])
                print('[%s] %s' % (file_name, str(ex)))
                # raise ex
                continue

            if show_result:
                disp = cv2.resize(disp, (0, 0), fx=1.0, fy=1.0)
                cv2.imshow('MuPDF JSON', disp)
                cv2.waitKey()

        # MuTool's XML
        if task in ['mutool']:
            disp = page_img.copy()
            target = 'line'

            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_string = f.read()
                xml_string = escape_xml_illegal_chars(xml_string, '')
                xml_string = xml_string.replace('c="&#', 'c="')
                # xml_string = html.escape(xml_string)
                doc_root = ET.fromstring(xml_string)

                page_nodes = doc_root.findall('page')
                for page_idx, page_node in enumerate(page_nodes):
                    page_width = float(page_node.get('width'))
                    page_height = float(page_node.get('height'))

                    image_rects = []
                    page = doc[page_idx]

                    # Image bbox extraction
                    # if 'image' in task_option:
                    #     rects = [itm["bbox"] for itm in page.get_image_info()]
                    #     for rect in rects:
                    #         x1 = max(0, rect[0] * resize_x)
                    #         y1 = max(0, rect[1] * resize_y)
                    #         x2 = max(0, rect[2] * resize_x)
                    #         y2 = max(0, rect[3] * resize_y)
                    #         if 'exclude_text_in_image' in task_option:
                    #             image_rects.append([x1, y1, x2, y2])
                    #         new_cells.append({
                    #             'bbox': (x1, y1, x2 - x1, y2 - y1),
                    #             'text': '',
                    #             'type': 'image',
                    #         })
                    #         cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

                    # Vector bbox extraction
                    if 'vector' in task_option:
                        for vector in page_node.iter('vector'):
                            vector_bbox = vector.get('bbox').split()
                            vector_bbox = list(map(float, vector_bbox))
                            vector_txt = ''
                            x1 = vector_bbox[0] * resize_x
                            y1 = vector_bbox[1] * resize_y
                            x2 = vector_bbox[2] * resize_x
                            y2 = vector_bbox[3] * resize_y

                            if x2 - x1 < 1 or y2 - y1 < 1:
                                continue

                            if 0 <= x1 < x2 <= disp.shape[1] and 0 <= y1 < y2 <= disp.shape[0]:
                                new_cells.append({
                                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                                    'text': vector_txt,
                                    'type': 'vector',
                                })
                                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

                    for block in page_node.iter('block'):
                        block_bbox = block.get('bbox').split()
                        block_bbox = list(map(float, block_bbox))
                        block_txt = ''

                        for line in block.iter('line'):
                            line_bbox = line.get('bbox').split()
                            line_bbox = list(map(float, line_bbox))
                            line_txt = ''
                            for font in line.iter('font'):
                                for c in font.iter('char'):
                                    line_txt += c.get('c')

                            line_txt = line_txt.strip()
                            if line_txt == '':
                                continue

                            if target == 'line':
                                x1 = line_bbox[0] * resize_x
                                y1 = line_bbox[1] * resize_y
                                x2 = line_bbox[2] * resize_x
                                y2 = line_bbox[3] * resize_y

                                if x2 - x1 < 1 or y2 - y1 < 1:
                                    continue

                                # Exclude text elements in images
                                if 'exclude_text_in_image' in task_option:
                                    is_in_image = False
                                    for irect in image_rects:
                                        i_x1, i_y1, i_x2, i_y2 = irect
                                        if i_x1 <= x1 < x2 <= i_x2 and i_y1 <= y1 < y2 <= i_y2:
                                            is_in_image = True
                                            break
                                    if is_in_image:
                                        continue

                                if 0 <= x1 < x2 <= disp.shape[1] and 0 <= y1 < y2 <= disp.shape[0]:
                                    new_cells.append({
                                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                                        'text': line_txt,
                                        'type': 'line',
                                    })
                                    cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

                        if target == 'block':
                            x1 = block_bbox[0] * resize_x
                            y1 = block_bbox[1] * resize_y
                            x2 = block_bbox[2] * resize_x
                            y2 = block_bbox[3] * resize_y
                            if 0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h:
                                new_cells.append({
                                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                                    'text': block_txt,
                                    'type': 'block',
                                })
                                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

            except Exception as ex:
                errors.append([file_name, str(ex)])
                print('[%s] %s' % (file_name, str(ex)))
                # raise ex
                continue

            if show_result:
                disp = cv2.resize(disp, (0, 0), fx=1.0, fy=1.0)
                cv2.imshow('Mutool JSON', disp)
                cv2.waitKey()

        # TextSense extraction result
        if task == 'TextSense':
            disp = img.copy()
            target = 'line'

            ts_file = '%s/%s0001-1.json' % (ts_dir, file_name[:-4])
            with open(ts_file, 'r') as f:
                ts_data = json.load(f)

            page_elems = ts_data['document']['pages'][0]['elements']
            resize_x = disp.shape[0] / ts_data['document']['pages'][0]['width']
            resize_y = disp.shape[1] / ts_data['document']['pages'][0]['height']
            for elem_idx, page_elem in enumerate(page_elems):
                if page_elem['type'] == 'word':
                    x1, y1 = page_elem['points'][0]
                    x2, y2 = page_elem['points'][1]

                    x1 = int(x1 * resize_x)
                    y1 = int(y1 * resize_y)
                    x2 = int(x2 * resize_x)
                    y2 = int(y2 * resize_y)
                    txt = page_elem['recognition']['content'].strip()
                    # if txt == '':
                    #     continue
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    new_cells.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'text': txt,
                    })
                elif page_elem['type'] == 'textline':
                    if target == 'line':
                        x1, y1 = page_elem['points'][0]
                        x2, y2 = page_elem['points'][1]

                        x1 = int(x1 * resize_x)
                        y1 = int(y1 * resize_y)
                        x2 = int(x2 * resize_x)
                        y2 = int(y2 * resize_y)
                        txt = page_elem['content'].strip()
                        # if txt == '':
                        #     continue
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        new_cells.append({
                            'bbox': (x1, y1, x2 - x1, y2 - y1),
                            'text': txt
                        })

                    elif target == 'word':
                        for elem_idx2, page_elem2 in enumerate(page_elem['texts']):
                            x1, y1 = page_elem2['points'][0]
                            x2, y2 = page_elem2['points'][1]

                            x1 = int(x1 * resize_x)
                            y1 = int(y1 * resize_y)
                            x2 = int(x2 * resize_x)
                            y2 = int(y2 * resize_y)
                            txt = page_elem2['recognition']['content'].strip()
                            # if txt == '':
                            #     continue
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            new_cells.append({
                                'bbox': (x1, y1, x2 - x1, y2 - y1),
                                'text': txt
                            })

            if '' == 'D':
                disp = cv2.resize(disp, (0, 0), fx=0.7, fy=0.7)
                cv2.imshow('TS JSON', disp)
                cv2.waitKey()

        doc.close()
        json_data['cells'] = new_cells
        with open(f'{save_dir}/{file_name[:-4]}.json', 'w') as f:
            json.dump(json_data, f)


def custom_collate_fn(batch):
    return batch[0]


def onnx_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    if dataset.feature_extractor is None and dataset.feature_extractor_path is not None:
        NUM_GPUS = int(os.environ.get('NUM_GPUS', 1))
        gpu_id = worker_id % NUM_GPUS
        try:
            torch.cuda.set_device(gpu_id)
            print(f"Worker {worker_id} assigned to GPU {gpu_id}")
        except Exception as e:
            print(f"Warning: Could not set CUDA device {gpu_id} for Worker {worker_id}. Error: {e}")

        try:
            session_options = ort.SessionOptions()
            print(f'Loading ... {dataset.feature_extractor_path} ...')
            dataset.feature_extractor = ort.InferenceSession(
                dataset.feature_extractor_path,
                sess_options=session_options,
                providers=[
                    ('CUDAExecutionProvider', {'device_id': gpu_id}),
                    'CPUExecutionProvider'
                ]
            )
            print(f"Worker {worker_id} - Feature Extractor Initialized on GPU {gpu_id}")
        except Exception as e:
            print(f"CUDA Initialization Failed for Worker {worker_id} on GPU {gpu_id}. Falling back to CPU. Error: {e}")
            dataset.feature_extractor = ort.InferenceSession(
                dataset.feature_extractor_path, providers=['CPUExecutionProvider']
            )


def safe_save_json_dataset_to_lmdb(cfg, max_retries=1000, wait_sec=60):
    for attempt in range(1, max_retries + 1):
        print(f"[INFO] Attempt {attempt} to run training...")
        proc = mp.Process(target=save_json_dataset_to_lmdb, args=(cfg,))
        proc.start()
        proc.join()

        if proc.exitcode == 0:
            print("[INFO] make_data() finished successfully.")
            break
        else:
            print(f"[WARN] make_data() crashed with exit code {proc.exitcode}. Retrying after {wait_sec} seconds...")
            time.sleep(wait_sec)
    else:
        print("[ERROR] make_data() failed after maximum retries.")

def save_json_dataset_to_lmdb(cfg):
    """
    Saves a JSON dataset to an LMDB database or MongoDB (GridFS) using multi-processing with DataLoader.

    Args:
        cfg (dict): Configuration dictionary containing all necessary parameters.
    """
    # Load configurations
    json_paths = cfg['save_json_dataset_to_lmdb']['json_paths']
    save_paths = cfg['save_json_dataset_to_lmdb']['save_paths']
    nn_type = cfg['save_json_dataset_to_lmdb']['nn_type']
    sample_k = cfg['save_json_dataset_to_lmdb']['sample_k']
    opt_save_type = cfg['save_json_dataset_to_lmdb']['opt_save_type']
    filter_class = cfg['save_json_dataset_to_lmdb']['filter_class']

    # 1. Add save_type configuration (Default: LMDB)
    # Options: 'LMDB', 'MongoDB'
    save_type = cfg['save_json_dataset_to_lmdb'].get('save_type', 'LMDB')

    # MongoDB Configuration (Used only if save_type == 'MongoDB')
    mongo_uri = cfg['save_json_dataset_to_lmdb'].get('mongo_uri', 'mongodb://localhost:27017/')
    mongo_db_name = cfg['save_json_dataset_to_lmdb'].get('mongo_db_name', 'dl_dataset_db')

    create_from_pdf = cfg['save_json_dataset_to_lmdb']['create_from_pdf']
    pdf_input_type = cfg['save_json_dataset_to_lmdb']['pdf_input_type']
    cfg_path = cfg['save_json_dataset_to_lmdb']['cfg_path']
    pkl_dirs = cfg['save_json_dataset_to_lmdb']['pkl_dirs']
    rf_executable = cfg['save_json_dataset_to_lmdb']['rf_executable']
    remove_unlabelled_data = cfg['save_json_dataset_to_lmdb']['remove_unlabelled_data']
    lmdb_size = cfg['save_json_dataset_to_lmdb']['lmdb_size']

    data_class_names = cfg['data']['class_list']
    data_class_map = {name: i for i, name in enumerate(data_class_names)}
    data_class_change_map = cfg['save_json_dataset_to_lmdb']['data_class_change_map']

    if data_class_change_map is not None:
        for key in data_class_change_map.keys():
            val = data_class_change_map[key]
            data_class_map[key] = val

    assert len(json_paths) == len(save_paths), "json_paths and save_paths must have the same length."

    feature_input_size = cfg['save_json_dataset_to_lmdb']['feature_input_size']
    feature_extractor_path = cfg['save_json_dataset_to_lmdb']['feature_extractor']
    feature_option = cfg['save_json_dataset_to_lmdb']['feature_option']

    pdf_dirs = cfg['save_json_dataset_to_lmdb']['pdf_dirs']
    data_names = [name.strip() for name in cfg['save_json_dataset_to_lmdb']['data_names'].split(',')]

    # Number of workers for DataLoader. Use all available CPU cores.
    num_workers = cfg['save_json_dataset_to_lmdb']['num_workers']
    print(f"Using {num_workers} workers for data loading. Save Type: {save_type}")

    for path_idx in range(len(json_paths)):
        if json_paths[path_idx] is None:
            continue

        for data_name in data_names:
            json_path = json_paths[path_idx]
            save_path = save_paths[path_idx].replace('[DATA_NAME]', data_name)

            if type(json_path) is list:
                json_path = json_path[0]

            pdf_dirs_temp = []
            for pdf_dir in pdf_dirs:
                pdf_dirs_temp.append(pdf_dir.replace('[DATA_NAME]', data_name))

            json_path = json_path.replace('[DATA_NAME]', data_name)

            # Initialize the dataset.
            if data_name in ['train', 'val']:
                dataset = DocumentJsonDataset(
                    data_path=json_path,
                    nn_type=nn_type,
                    nn_k=sample_k,
                    class_map=data_class_map,
                    is_train=(data_name == 'train'),
                    feature_extractor_path=feature_extractor_path,
                    feature_input_size=feature_input_size,
                    feature_option=feature_option,
                    lmdb_save_path=save_path,
                    pdf_dir=pdf_dirs_temp,
                    create_from_pdf=create_from_pdf,
                    pdf_input_type=pdf_input_type,
                    cfg_path=cfg_path,
                    pkl_dirs=pkl_dirs,
                    rf_executable=rf_executable,
                    remove_unlabelled_data=remove_unlabelled_data,
                    filter_class=filter_class,
                    opt_save_type=opt_save_type,
                    show_data=True
                )
            else:
                raise Exception(f'Invalid data_name = {data_name}')

            # Common DataLoader setup
            dataset.txn_keys = []

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,  # Process one sample at a time
                shuffle=False,
                num_workers=num_workers,
                collate_fn=custom_collate_fn,
                worker_init_fn=onnx_worker_init_fn,
            )

            os.makedirs(save_path, exist_ok=True)

            # ==========================================
            # Case 1: Save to LMDB
            # ==========================================
            if save_type == 'LMDB':
                env = lmdb.open(save_path, map_size=lmdb_size * 1024 ** 3)
                print(f'Saving data to {save_path} (LMDB) with {num_workers} workers.')

                # Load existing keys if appending
                read_txn = env.begin(write=False)
                keys = read_txn.get(b'__keys__')
                if keys is not None:
                    keys = pickle.loads(keys)
                else:
                    keys = []
                read_txn.abort()

                dataset.txn_keys = keys

                txn = env.begin(write=True)

                for idx, data in enumerate(dataloader):
                    file_name = 'None'
                    process_time = 0.0
                    if type(data) == str:
                        print(f'[{idx}/{len(dataloader)}] {data} ...')
                        continue

                    if data is not None:
                        file_name = data.raw_data['file_name']
                        process_time = data.raw_data['process_time']
                        del data.raw_data['process_time']

                    if '' == 'D':
                        with open(f'temp/{file_name}', 'wb') as f:
                            pickle.dump(data, f)

                    print(f'[{idx}/{len(dataloader)}] {file_name} ({process_time:.2f} sec) ...')

                    # Skip invalid data
                    if data is None or data.edge_index.shape[1] <= 0 or data.x.shape[0] >= 5000:
                        continue

                    file_name_base = os.path.splitext(data.raw_data['file_name'])[0]
                    key = hashlib.sha256(file_name_base.encode('utf-8')).digest()

                    try:
                        value = pickle.dumps(data)
                        compressed_value = blosc.compress(value, typesize=8, clevel=5, cname='zstd')
                        txn.put(key, compressed_value)
                        keys.append(key)
                    except Exception as ex:
                        print(ex)
                        continue

                    if len(keys) % 1000 == 0:
                        print('Committing transaction...')
                        txn.put(b'__keys__', pickle.dumps(keys))
                        txn.commit()
                        txn = env.begin(write=True)

                txn.put(b'__keys__', pickle.dumps(keys))
                txn.commit()
                env.sync()
                env.close()
                print(f"LMDB dataset saved successfully ({len(keys)} samples).")

            # ==========================================
            # Case 2: Save to MongoDB (GridFS)
            # ==========================================
            elif save_type == 'MongoDB':
                print(f'Connecting to MongoDB: {mongo_uri}, DB: {mongo_db_name}')
                try:
                    client = pymongo.MongoClient(mongo_uri)
                    db = client[mongo_db_name]
                    # Use GridFS for large object storage (replaces LMDB blobs)
                    fs = gridfs.GridFS(db, collection=data_name)
                except Exception as e:
                    print(f"Failed to connect to MongoDB: {e}")
                    return

                print(f'Saving data to MongoDB (GridFS: {data_name}) with {num_workers} workers.')

                valid_count = 0
                for idx, data in enumerate(dataloader):
                    file_name = 'None'

                    if type(data) == str:
                        print(f'[{idx}/{len(dataloader)}] {data} ...')
                        continue

                    if data is not None:
                        file_name = data.raw_data['file_name']

                    if '' == 'D':
                        with open(f'temp/{file_name}', 'wb') as f:
                            pickle.dump(data, f)

                    print(f'[{idx}/{len(dataloader)}] {file_name} ...')

                    # Skip invalid data
                    if data is None or data.edge_index.shape[1] <= 0 or data.x.shape[0] >= 5000:
                        continue

                    try:
                        # 1. Prepare Key (Filename based)
                        key_str = os.path.splitext(file_name)[0]
                        safe_key = key_str.replace('/', '_').replace('\\', '_')

                        # 2. Serialize & Compress (Optional but recommended for network transfer)
                        value = pickle.dumps(data)
                        compressed_value = blosc.compress(value, typesize=8, clevel=5, cname='zstd')

                        # 3. Check if file exists and delete (Simulate Upsert for GridFS)
                        # GridFS stores versions, but usually in datasets we want the latest unique file.
                        existing_file = fs.find_one({"filename": safe_key})
                        if existing_file:
                            fs.delete(existing_file._id)

                        # 4. Save to GridFS
                        # We store the compressed blob.
                        # Metadata can be added to 'metadata' field easily.
                        fs.put(
                            compressed_value,
                            filename=safe_key,
                            metadata={
                                "original_filename": file_name,
                                "compression": "blosc-zstd"
                            }
                        )
                        valid_count += 1

                    except Exception as ex:
                        print(f"Error saving to MongoDB: {ex}")
                        continue

                print(f"MongoDB dataset saved successfully ({valid_count} samples in GridFS collection '{data_name}').")

            else:
                raise ValueError(f"Unsupported save_type: {save_type}. Use 'LMDB' or 'MongoDB'.")


# A lock to manage console output to prevent interleaved print statements
print_lock = threading.Lock()


def get_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_file_path_list(root_path, extension_list):
    """
    Recursively find all files with given extensions in a directory.
    """
    file_path_list = []
    for (root, directories, files) in os.walk(root_path):
        for file in files:
            if file.split('.')[-1].lower() in extension_list:
                file_path = os.path.join(root, file)
                file_path_list.append(file_path)
    file_path_list.sort()
    return file_path_list


def make_graph_data_task(label_path, json_dir, pdf_dir, save_dir):
    cell_resize_x = 1.0
    cell_resize_y = 1.0
    margin = 5

    file_name = os.path.basename(label_path)[:-4]

    try:
        # Open PDF and get page image
        doc = pymupdf.open(f'{pdf_dir}/{file_name}.pdf')
        page = doc[0]
        pix = page.get_pixmap()
        bytes_img = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = cv2.cvtColor(bytes_img.reshape(pix.height, pix.width, pix.n), cv2.COLOR_BGR2RGB)
        doc.close()
    except Exception as ex:
        with print_lock:
            print(f'Error processing PDF {file_name}: {str(ex)}')
        return None

    try:
        # Load and decompress ground truth data
        with open(label_path, 'rb') as f:
            compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(depressed_pickle)
        gt_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)
        gt_label = pkl_data['label']
    except Exception as ex:
        with print_lock:
            print(f'Error loading pickle for {file_name}: {str(ex)}')
        return None

    if gt_img.shape != page_img.shape:
        gt_resize_x = page_img.shape[1] / gt_img.shape[1]
        gt_resize_y = page_img.shape[0] / gt_img.shape[0]
        for gt_idx in range(len(gt_label)):
            gt_label[gt_idx][0] *= gt_resize_x
            gt_label[gt_idx][1] *= gt_resize_y
            gt_label[gt_idx][2] *= gt_resize_x
            gt_label[gt_idx][3] *= gt_resize_y

    if not gt_label:
        return None

    json_path = os.path.join(json_dir, file_name + '.json')
    if not os.path.exists(json_path):
        with print_lock:
            print(f'[{json_path}] does not exist!')
        return None

    try:
        with open(json_path) as f:
            json_data = json.load(f)
    except Exception as ex:
        with print_lock:
            print(f'Error loading JSON for {file_name}: {str(ex)}')
        return None

    result_data = []
    for ocr_data in json_data.get('cells', []):
        if ocr_data['bbox'][2] <= 0 or ocr_data['bbox'][3] <= 0:
            continue

        iou_list = []
        bbox = ocr_data['bbox']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        img_scaled_bbox = [bbox[0] * cell_resize_x, bbox[1] * cell_resize_y,
                           bbox[2] * cell_resize_x, bbox[3] * cell_resize_y]

        for i, gt_data in enumerate(gt_label):
            if gt_data[2] - gt_data[0] <= 0 or gt_data[3] - gt_data[1] <= 0:
                iou_list.append(-1.0)
            else:
                label_bbox = gt_data[:4]
                if ((label_bbox[0] - margin <= img_scaled_bbox[0] and img_scaled_bbox[2] <= label_bbox[2] + margin) and
                        (label_bbox[1] - margin <= img_scaled_bbox[1] and img_scaled_bbox[3] <= label_bbox[
                            3] + margin)):
                    iou_list.append(1.0)
                else:
                    iou = max(get_iou(img_scaled_bbox, label_bbox), get_iou(label_bbox, img_scaled_bbox))
                    iou_list.append(iou)

        max_iou = max(iou_list)
        index = iou_list.index(max_iou)
        label = gt_label[index][-1]
        pdf_type = ocr_data.get('type', '')

        item = {
            'text_coor': bbox,
            'label': f'{label}_{index}',
            'content': ocr_data['text'],
            'pdf_type': pdf_type,
            'max_iou': max_iou
        }

        if max_iou < 0.5:
            item['label'] = f'no_label_{index}'

        result_data.append(item)

    os.makedirs(os.path.join(save_dir, 'graph_labels'), exist_ok=True)
    out_path = os.path.join(save_dir, 'graph_labels', file_name + '.json')
    try:
        with open(out_path, "w", encoding='utf-8') as f:
            json.dump({
                'img_data_list': result_data,
                'doc_category': json_data['metadata']['doc_category']
            }, f, ensure_ascii=False, indent=2)
    except Exception as ex:
        with print_lock:
            print(f'Error saving JSON for {file_name}: {str(ex)}')

    # Return data needed for visualization
    return page_img, gt_label, json_data, result_data, file_name, out_path


def make_graph_data(cfg):
    """
    Main function to process data using multithreading and handle visualization.
    """
    data_names = cfg['make_graph_data']['data_names']
    pkl_dir_0 = cfg['make_graph_data']['pkl_dir']
    json_dirs = cfg['make_graph_data']['json_dirs']
    save_dirs = cfg['make_graph_data']['save_dirs']
    empty_save_dir = cfg['make_graph_data']['empty_save_dir']
    num_workers = cfg['make_graph_data']['num_workers']
    show_pre_gt = cfg['make_graph_data']['show_pre_gt']
    show_post_gt = cfg['make_graph_data']['show_post_gt']

    assert len(json_dirs) == len(save_dirs)

    for dir_idx in range(len(json_dirs)):
        for data_name in data_names:
            st_idx = cfg['make_graph_data']['st_idx']
            ed_idx = cfg['make_graph_data']['ed_idx']

            pkl_dir = pkl_dir_0.replace('[DATA_NAME]', data_name)
            json_dir = json_dirs[dir_idx]
            save_dir = save_dirs[dir_idx].replace('[DATA_NAME]', data_name)
            pdf_dir = cfg['make_graph_data']['pdf_dir'].replace('[DATA_NAME]', data_name)

            print(f'PKL Dir [{pkl_dir}].')
            print(f'JSON Dir [{json_dir}].')

            if empty_save_dir:
                print(f'Removing [{save_dir}] ...')
                shutil.rmtree(save_dir, ignore_errors=True)
                os.makedirs(os.path.join(save_dir, 'graph_labels'), exist_ok=True)

            label_path_list = get_file_path_list(pkl_dir, ['pkl'])

            if ed_idx == 0:
                ed_idx = len(label_path_list)

            if show_pre_gt or show_post_gt or num_workers == 1:
                for file_idx in range(st_idx, ed_idx):
                    if file_idx % 100 == 0:
                        print('%d/%d ...' % (file_idx, ed_idx))

                    label_path = label_path_list[file_idx]
                    file_name = os.path.basename(label_path)[:-4]
                    out_path = os.path.join(save_dir, 'graph_labels', file_name + '.json')
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                        continue

                    result = make_graph_data_task(label_path, json_dir, pdf_dir, save_dir)
                    if result is None:
                        continue
                    page_img, gt_label, json_data, result_data, file_name, out_path = result

                    if show_pre_gt:
                        disp_pre = page_img.copy()
                        for label in gt_label:
                            bbox = list(map(int, label[:4]))
                            cv2.rectangle(disp_pre, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                        cv2.imshow('GT', disp_pre)

                    if show_post_gt:
                        cell_resize_x = 1.0
                        cell_resize_y = 1.0

                        disp_dt = page_img.copy()
                        for cell in json_data.get('cells', []):
                            bbox = cell['bbox']
                            x1 = int(bbox[0] * cell_resize_x)
                            y1 = int(bbox[1] * cell_resize_y)
                            x2 = int((bbox[0] + bbox[2]) * cell_resize_x)
                            y2 = int((bbox[1] + bbox[3]) * cell_resize_y)
                            cv2.rectangle(disp_dt, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.imshow('DT', disp_dt)

                        disp_post = page_img.copy()
                        for r in result_data:
                            bbox = r['text_coor']
                            x1 = int(bbox[0] * cell_resize_x)
                            y1 = int(bbox[1] * cell_resize_y)
                            x2 = int(bbox[2] * cell_resize_x)
                            y2 = int(bbox[3] * cell_resize_y)
                            cv2.putText(disp_post, r['label'], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
                            cv2.rectangle(disp_post, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.imshow('Post-GT', disp_post)
                        cv2.waitKey()

            else:
                futures = []
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    for file_idx in range(st_idx, ed_idx):
                        if file_idx % 100 == 0:
                            print('%d/%d ...' % (file_idx, ed_idx - st_idx))

                        if len(futures) < num_workers:
                            label_path = label_path_list[file_idx]

                            file_name = os.path.basename(label_path)[:-4]
                            out_path = os.path.join(save_dir, 'graph_labels', file_name + '.json')
                            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                                continue

                            future = executor.submit(make_graph_data_task, label_path, cfg, json_dir, save_dir)
                            futures.append(future)
                        else:
                            completed = []
                            for future in as_completed(futures):
                                completed.append(future)
                            for future in completed:
                                futures.remove(future)



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

        if os.path.exists('./temp/csv'):
            with open(f'temp/csv/{filename[:-4]}.csv', 'w') as f:
                for line in lines:
                    f.write(line + '\n')

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


def custom_feature_task(feature_path, pdf_dir, pkl_dir, file_name, json_path, show_data=False):
    from train.infer.common_util import get_text_pattern

    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        pkl_file = f'{pkl_dir}/{file_name[:-5]}.pkl'
        with open(pkl_file, 'rb') as f2:
            compressed_pickle = f2.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(depressed_pickle)
        gt_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)
        gt_label = pkl_data['label']

        # Resize image scale to page scale
        doc = pymupdf.open(f'{pdf_dir}/{file_name[:-5]}.pdf')
        page = doc[0]
        page_width = page.rect.width
        page_height = page.rect.height
        # resize_x = page.rect[2] / gt_img.shape[1]
        # resize_y = page.rect[3] / gt_img.shape[0]
        resize_x = 1.0
        resize_y = 1.0

        json_data['meta'] = {
            'page_width': page_width,
            'page_height': page_height,
        }

        pdf_rects = []
        for cell_idx, cell_data in enumerate(json_data['img_data_list']):
            text = cell_data['content'].strip()

            x1 = float(cell_data['text_coor'][0]) * resize_x
            y1 = float(cell_data['text_coor'][1]) * resize_y
            x2 = float(cell_data['text_coor'][2]) * resize_x
            y2 = float(cell_data['text_coor'][3]) * resize_y
            pdf_rects.append([x1, y1, x2, y2])

            custom_features = {}
            if 'custom_features' in cell_data:
                custom_features = cell_data['custom_features']
            else:
                cell_data['custom_features'] = custom_features

            is_line = 1     # default
            is_vector = 0
            is_image = 0

            # PDF data type
            if 'pdf_type' in cell_data:
                is_vector = int(cell_data['pdf_type'] == 'vector')
                is_image = int(cell_data['pdf_type'] == 'image')
                if is_vector == 1 or is_image == 1:
                    is_line = 0

            # Numeric ratio in content
            num_ratio = 0
            for c in text:
                if c.isdigit():
                    num_ratio += 1
            if len(text) == 0:
                num_ratio = 0.0
            else:
                num_ratio = num_ratio / len(text)

            custom_features['num_ratio'] = num_ratio
            custom_features['is_line'] = is_line
            custom_features['is_vector'] = is_vector
            custom_features['is_image'] = is_image
            custom_features['text_pattern'] = get_text_pattern(text, return_vector=False)

        # Robin's feature
        if feature_path is not None and len(json_data['img_data_list']) > 0:
            st_time = time.time()
            rb_feat_name, rb_feat_value = robins_features_extraction(feature_path, pdf_dir, f'{file_name[:-5]}.pdf', pdf_rects)
            elapsed_time = time.time() - st_time

            # Code for execution time analysis
            # with open('./temp/time_record.csv', 'a') as f:
            #     csv_writer = csv.writer(f)
            #     csv_writer.writerow([file_name[:-5] + '.pdf', len(pdf_rects), elapsed_time])

            # for f_name in rb_feat_name:
            #     print("'%s'" % f_name, end=',')
            for cell_idx, cell_data in enumerate(json_data['img_data_list']):
                for f_idx, f_name in enumerate(rb_feat_name):
                    if f_idx < 9:
                        continue
                    cell_data['custom_features'][f_name] = rb_feat_value[cell_idx][f_idx]

        if show_data:
            pix = page.get_pixmap()
            bytes = np.frombuffer(pix.samples, dtype=np.uint8)
            page_img = bytes.reshape(pix.height, pix.width, pix.n)
            page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = page_img.shape
            for r in pdf_rects:
                r = list(map(int, r))
                cv2.rectangle(page_img, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 1)
            cv2.imshow('PDF', page_img)
            cv2.waitKey()
        else:
            # Write final result to original file
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
            pass
        doc.close()

    except Exception as ex:
        print('Error : ' + str(ex))


def make_doclaynet_json_add_custom_feature(cfg):
    src_dirs = cfg['make_doclaynet_json_add_custom_feature']['src_dirs']
    pdf_dir = cfg['make_doclaynet_json_add_custom_feature']['pdf_dir']
    pkl_dir = cfg['make_doclaynet_json_add_custom_feature']['pkl_dir']
    data_names = cfg['make_doclaynet_json_add_custom_feature']['data_names']
    show_data = cfg['make_doclaynet_json_add_custom_feature']['show_data']

    max_workers = cfg['make_doclaynet_json_add_custom_feature']['max_workers']
    features_executable_path = cfg['make_doclaynet_json_add_custom_feature']['features_executable_path']

    for src_dir in src_dirs:
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for data_name in data_names:
                file_list = os.listdir(f'{src_dir}/{data_name}/graph_labels')

                src_dir_temp = src_dir.replace('[DATA_NAME]', data_name)
                pdf_dir_temp = pdf_dir.replace('[DATA_NAME]', data_name)
                pkl_dir_temp = pkl_dir.replace('[DATA_NAME]', data_name)

                for file_idx, file_name in enumerate(file_list):
                    json_path = f'{src_dir_temp}/{data_name}/graph_labels/{file_name}'
                    try:
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                    except Exception as ex:
                        print(ex)
                        continue

                    # if len(json_data['img_data_list']) > 10000 or \
                    #         (len(json_data['img_data_list']) > 0 and
                    #          'custom_features' in json_data['img_data_list'][0] and
                    #          'avg char space' in json_data['img_data_list'][0]['custom_features']):
                    #     print('[%s][%s][%d/%d] %s (%d) - Skip' % (src_dir, data_name, file_idx, len(file_list), file_name, len(json_data['img_data_list'])))
                    #     continue
                    if len(json_data['img_data_list']) > 10000:
                        continue

                    if len(json_data['img_data_list']) > 0:
                        print('[%s][%s][%d/%d] %s (%d)' % (src_dir, data_name, file_idx, len(file_list), file_name, len(json_data['img_data_list'])))

                    if show_data or max_workers == 1:
                        custom_feature_task(features_executable_path, pdf_dir_temp, pkl_dir_temp, file_name, json_path, show_data=show_data)
                    else:
                        while len(futures) > max_workers:
                            done_list = []
                            for f in futures:
                                if f.done():
                                    # ret = f.result(timeout=600)
                                    done_list.append(f)
                            for f in done_list:
                                futures.remove(f)
                        future = executor.submit(custom_feature_task, features_executable_path, pdf_dir_temp, pkl_dir_temp,
                                                 file_name, json_path, show_data)
                        futures.append(future)


def visualize_lmdb_dataset(cfg):
    import cv2
    import pymupdf
    from train.tools.data.layout.DocumentLMDBDataset import DocumentLMDBDataset

    pdf_dir = cfg['visualize_lmdb_dataset']['pdf_dir']
    lmdb_path = cfg['visualize_lmdb_dataset']['lmdb_path']

    val_data = DocumentLMDBDataset(lmdb_path=lmdb_path, cache_size=0, keep_raw_data=True)

    for data in val_data:
        raw_data = data.raw_data

        pdf_path = f'{pdf_dir}/{raw_data["file_name"][:-5]}.pdf'
        doc = pymupdf.open(pdf_path)
        page = doc[0]

        pix = page.get_pixmap()
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = bytes.reshape(pix.height, pix.width, pix.n)
        page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)

        for bbox in raw_data['bboxes']:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 200), 1)

        cv2.imshow('PDF Page', page_img)
        cv2.waitKey()

def make_table_modified_pkl_dataset(cfg):
    pdf_dir = cfg['make_table_modified_pkl_dataset']['pdf_dir']
    pkl_dir = cfg['make_table_modified_pkl_dataset']['pkl_dir']
    save_dir = cfg['make_table_modified_pkl_dataset']['save_dir']

    show_pre_gt = cfg['make_table_modified_pkl_dataset']['show_pre_gt']
    show_post_gt = cfg['make_table_modified_pkl_dataset']['show_post_gt']

    os.makedirs(save_dir, exist_ok=True)

    file_list = os.listdir(pkl_dir)
    for file_idx, pkl_file_name in enumerate(file_list):
        file_name = pkl_file_name[:-4]
        if file_idx % 10 == 0:
            print('[%d/%d] %s ...' % (file_idx, len(file_list), file_name))

        gt_file = f'{pkl_dir}/{pkl_file_name}'
        with open(gt_file, 'rb') as f:
            compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(depressed_pickle)
        gt_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)
        gt_label = pkl_data['label']

        pdf_file = f'{pdf_dir}/{file_name}.pdf'
        doc = pymupdf.open(pdf_file)
        page = doc[0]
        pix = page.get_pixmap()
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = bytes.reshape(pix.height, pix.width, pix.n)
        page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)

        gt_resize_x = page_img.shape[1] / gt_img.shape[1]
        gt_resize_y = page_img.shape[0] / gt_img.shape[0]
        inv_resize_x = gt_img.shape[1] / page_img.shape[1]
        inv_resize_y = gt_img.shape[0] / page_img.shape[0]

        table_count = 0
        for gt_idx in range(len(gt_label)):
            if gt_label[gt_idx][-1] == 'table':
                table_count += 1
            gt_label[gt_idx][0] *= gt_resize_x
            gt_label[gt_idx][1] *= gt_resize_y
            gt_label[gt_idx][2] *= gt_resize_x
            gt_label[gt_idx][3] *= gt_resize_y

        if (show_pre_gt or show_post_gt) and table_count == 0:
            continue

        if show_pre_gt:
            disp = page_img.copy()
            for ant in gt_label:
                if ant[-1] != 'table':
                    continue
                x1, y1, x2, y2 = list(map(int, ant[:4]))
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.imshow('Pre GT', disp)
            if not show_post_gt:
                cv2.waitKey()

        # Find texts in table and modify gt bbox to text enclosing bbox
        included_lines = []
        for ant in gt_label:
            if ant[-1] != 'table':
                continue

            table_x1, table_y1, table_x2, table_y2 = list(map(int, ant[:4]))
            blocks = page.get_text("dict", flags=11)["blocks"]
            min_new_x1, min_new_y1 = float('inf'), float('inf')
            max_new_x2, max_new_y2 = float('-inf'), float('-inf')
            found_any_line_in_area = False  # Flag to check if any line was included
            margin = 5
            for block_idx, block in enumerate(blocks):
                for line_idx, line in enumerate(block["lines"]):
                    lx1, ly1, lx2, ly2 = line['bbox']
                    if (lx1 >= table_x1 - margin and ly1 >= table_y1 - margin and
                            lx2 <= table_x2 + margin and ly2 <= table_y2 + margin):
                        included_lines.append([lx1, ly1, lx2, ly2])
                        found_any_line_in_area = True
                        min_new_x1 = min(min_new_x1, lx1)
                        min_new_y1 = min(min_new_y1, ly1)
                        max_new_x2 = max(max_new_x2, lx2)
                        max_new_y2 = max(max_new_y2, ly2)
            if found_any_line_in_area:
                ant[0] = min_new_x1
                ant[1] = min_new_y1
                ant[2] = max_new_x2
                ant[3] = max_new_y2
            else:
                print('No texts in table!')
                # show_post_gt = True

        if show_post_gt:
            disp = page_img.copy()
            for ant in included_lines:
                x1, y1, x2, y2 = list(map(int, ant[:4]))
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 0), 1)
            for ant in gt_label:
                if ant[-1] != 'table':
                    continue
                x1, y1, x2, y2 = list(map(int, ant[:4]))
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.imshow('Post GT', disp)
            cv2.waitKey()

        for gt_idx in range(len(gt_label)):
            if gt_label[gt_idx][-1] == 'table':
                table_count += 1
            gt_label[gt_idx][0] *= inv_resize_x
            gt_label[gt_idx][1] *= inv_resize_y
            gt_label[gt_idx][2] *= inv_resize_x
            gt_label[gt_idx][3] *= inv_resize_y

        pkl_data['label'] = gt_label
        pickled_data = pickle.dumps(pkl_data)
        compressed_pickle = blosc.compress(pickled_data)
        with open(f'{save_dir}/{pkl_file_name}', 'wb') as f:
            f.write(compressed_pickle)


# Change bbox dsize
def fix_bbox_size_of_graph_data():
    graph_json_dir = '/media/win/Dataset/Layout2Graph/DocLayNet_core_graph_labels_ts/train/graph_labels'
    file_list = os.listdir(graph_json_dir)

    for file_idx, filename in enumerate(file_list):
        if file_idx % 1000 == 0:
            print('%d/%d ...' % (file_idx, len(file_list)))

        json_path = f'{graph_json_dir}/{filename}'
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        page_width, page_height = int(json_data['meta']['page_width']), int(json_data['meta']['page_height'])

        # From DoclayNet COCO image size
        resize_x = page_width / 1025
        resize_y = page_height / 1025

        disp = np.zeros(shape=(page_height, page_width, 3), dtype=np.uint8)
        disp[:] = 255
        for data_idx, data in enumerate(json_data['img_data_list']):
            x1, y1, x2, y2 = data['text_coor']
            x1 *= resize_x
            y1 *= resize_y
            x2 *= resize_x
            y2 *= resize_y

            data['text_coor'] = [x1, y1, x2, y2]
            x1, y1, x2, y2 = data['text_coor']
            cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        if '' == 'D':
            cv2.imshow('Annotations', disp)
            cv2.waitKey(0)
        else:
            with open(json_path, 'w') as f:
                json.dump(json_data, f)


def modify_gt_dataset_by_docling(cfg):
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    from PIL import Image

    pdf_dir = cfg['make_gt_dataset_by_docling']['pdf_dir']
    pkl_dir = cfg['make_gt_dataset_by_docling']['pkl_dir']
    save_dir = cfg['make_gt_dataset_by_docling']['save_dir']

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_processor = RTDetrImageProcessor.from_pretrained("HuggingPanda/docling-layout")
    model = RTDetrForObjectDetection.from_pretrained("HuggingPanda/docling-layout")
    model.to('cuda:0')

    file_list = os.listdir(pkl_dir)
    for file_idx, pkl_file in enumerate(file_list):
        filename = os.path.splitext(pkl_file)[0]
        print(f'[{file_idx + 1}/{len(file_list)}] {filename} ...')

        pdf_path = f'{pdf_dir}/{filename}.pdf'
        pkl_path = f'{pkl_dir}/{pkl_file}'

        # Load existing ground truth data
        with open(pkl_path, 'rb') as f:
            compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(depressed_pickle)
        gt_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)
        gt_label = pkl_data['label']

        # Calculate scaling ratios and adjust GT bounding boxes
        try:
            doc = pymupdf.open(pdf_path)
            page = doc[0]
            pix = page.get_pixmap()
            bytes = np.frombuffer(pix.samples, dtype=np.uint8)
            page_img = bytes.reshape(pix.height, pix.width, pix.n)
            page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)
            doc.close()
        except Exception as ex:
            print(pdf_path, str(ex))
            continue

        gt_resize_x = page_img.shape[1] / gt_img.shape[1]
        gt_resize_y = page_img.shape[0] / gt_img.shape[0]

        table_count = 0
        for gt_idx in range(len(gt_label)):
            if gt_label[gt_idx][-1] == 'table':
                table_count += 1
            gt_label[gt_idx][0] *= gt_resize_x
            gt_label[gt_idx][1] *= gt_resize_y
            gt_label[gt_idx][2] *= gt_resize_x
            gt_label[gt_idx][3] *= gt_resize_y

        docling_result = []
        # Load an image
        image = Image.fromarray(page_img)

        # Preprocess the image
        resize = {"height": 1024, "width": 1024}
        inputs = image_processor(
            images=image,
            return_tensors="pt",
            size=resize,
        )
        inputs = inputs.to('cuda:0')

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]),
            threshold=0.5
        )

        for result in results:
            for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                score, label = score.item(), label_id.item()
                cls = model.config.id2label[label + 1].lower()
                bbox = [round(i, 2) for i in box.tolist()]
                docling_result.append([bbox[0], bbox[1], bbox[2], bbox[3], cls])

                # cv2.putText(page_img, cls, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)
                # cv2.rectangle(page_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)

            # cv2.imshow('Det Result', page_img)
            # cv2.waitKey()

        # Replace existing GT to docling result
        # if '' == '':
        #     # Create a new list to store updated labels
        #     new_gt_label = gt_label[:]
        #
        #     # Compare GT bboxes with Docling bboxes and replace if IoU is high
        #     for gt_idx, gt_bbox_info in enumerate(new_gt_label):
        #         # gt_bbox_info: (x1, y1, x2, y2, class_name)
        #         if gt_bbox_info[4] == 'table':  # Process only 'table' class bboxes
        #             gt_bbox = gt_bbox_info[:4]  # Extract bbox coordinates
        #             best_iou = 0
        #             best_docling_bbox = None
        #
        #             for docling_bbox in docling_tables:
        #                 iou = calculate_iou(gt_bbox, docling_bbox)
        #                 if iou > best_iou:
        #                     best_iou = iou
        #                     best_docling_bbox = docling_bbox
        #
        #             # If IoU is greater than or equal to the threshold, replace coordinates
        #             if best_iou >= iou_threshold:
        #                 new_gt_label[gt_idx][0] = best_docling_bbox[0]
        #                 new_gt_label[gt_idx][1] = best_docling_bbox[1]
        #                 new_gt_label[gt_idx][2] = best_docling_bbox[2]
        #                 new_gt_label[gt_idx][3] = best_docling_bbox[3]

        # Update pkl_data with the new gt_label
        pkl_data['label'] = docling_result

        # Save the new pkl file
        save_path = os.path.join(save_dir, f'{filename}.pkl')
        compressed_pickle = blosc.compress(pickle.dumps(pkl_data))
        with open(save_path, 'wb') as f:
            f.write(compressed_pickle)

        if table_count > 0:
            save_path = os.path.join('/media/win/Dataset/PubLayNet/train_pkl_docling_table', f'{filename}.pkl')
            compressed_pickle = blosc.compress(pickle.dumps(pkl_data))
            with open(save_path, 'wb') as f:
                f.write(compressed_pickle)


def make_gt_dataset_by_docling(cfg):
    from docling.document_converter import DocumentConverter

    pdf_dir = cfg['make_gt_dataset_by_docling']['pdf_dir']
    save_dir = cfg['make_gt_dataset_by_docling']['save_dir']

    if os.path.exists(save_dir + '/pkl'):
        exist_file_list = os.listdir(save_dir + '/pkl')
        for file_name in exist_file_list:
            file_name = file_name.split('_')[0]
            pdf_path = f'{pdf_dir}/{file_name}.pdf'
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    file_list = os.listdir(pdf_dir)
    os.makedirs(save_dir + '/PDF', exist_ok=True)
    os.makedirs(save_dir + '/pkl', exist_ok=True)

    da = DocumentConverter()

    file_count = 0
    for file_idx, pdf_file in enumerate(file_list):
        try:
            doc = pymupdf.open(f'{pdf_dir}/{pdf_file}')

            for page_idx, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                if len(blocks) < 10:
                    continue

                conv_res = da.convert(f'{pdf_dir}/{pdf_file}', page_range=(page_idx + 1, page_idx + 1))

                bboxes = []
                table_count = 0

                for doc_elem in conv_res.assembled.elements:
                    cls_label = str(doc_elem.label).replace('_', '-').strip()
                    x1, y1, x2, y2 = doc_elem.cluster.bbox.l, doc_elem.cluster.bbox.t, doc_elem.cluster.bbox.r, doc_elem.cluster.bbox.b

                    if cls_label == 'document-index':
                        cls_label = 'table'

                    if cls_label == 'table':
                        table_count += 1

                    bboxes.append([x1, y1, x2, y2, 0, cls_label])

                    if cls_label == 'table':
                        for cell in doc_elem.table_cells:
                            x1, y1, x2, y2 = cell.bbox.l, cell.bbox.t, cell.bbox.r, cell.bbox.b
                            bboxes.append([x1, y1, x2, y2, 0, 'table-cell', [cell.start_row_offset_idx, cell.start_row_offset_idx, cell.row_span, cell.col_span]])

                if table_count >= 0:
                    pix = page.get_pixmap()
                    bytes = np.frombuffer(pix.samples, dtype=np.uint8)
                    page_img = bytes.reshape(pix.height, pix.width, pix.n)
                    page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)

                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    enc_ret, encimg = cv2.imencode('.jpg', page_img, encode_param)
                    if not enc_ret:
                        raise Exception('Jpeg Encoding Error!')

                    pkl_data = {
                        'jpeg': encimg,
                        'label': bboxes,
                        'info': {},
                    }

                    pkl_path = os.path.join(save_dir, 'pkl', f'{pdf_file[:-4]}_{page_idx}.pkl')
                    compressed_pickle = blosc.compress(pickle.dumps(pkl_data))
                    with open(pkl_path, 'wb') as f:
                        f.write(compressed_pickle)

                    new_doc = fitz.open()
                    new_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                    pdf_path = os.path.join(save_dir, 'PDF', f'{pdf_file[:-4]}_{page_idx}.pdf')
                    new_doc.save(pdf_path)
                    new_doc.close()

                    file_count += 1
                    if 'D' == 'D' and file_count % 10 == 0:
                        print(f'[{file_idx}/{len(file_list)}]{pdf_file} / {page_idx} ...')
                        for ant in bboxes:
                            cv2.rectangle(page_img, (int(ant[0]), int(ant[1])), (int(ant[2]), int(ant[3])), (0, 0, 255), 1)
                            cv2.putText(page_img, ant[5], (int(ant[0]), int(ant[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 0), 1)
                        cv2.imshow('Result', page_img)
                        cv2.waitKey(10)
        except Exception as ex:
            print(f'{pdf_file} - {str(ex)}')

        os.remove(f'{pdf_dir}/{pdf_file}')


def sampling_docling_dataset(cfg):
    data_dir = cfg['sampling_docling_dataset']['data_dir']
    save_dir = cfg['sampling_docling_dataset']['save_dir']
    sample_num = cfg['sampling_docling_dataset']['sample_num']
    remove_original = cfg['sampling_docling_dataset'].get('remove_original', False)

    pdf_dir = os.path.join(data_dir, 'PDF')
    pkl_dir = os.path.join(data_dir, 'pkl')

    save_pdf_dir = os.path.join(save_dir, 'PDF')
    save_pkl_dir = os.path.join(save_dir, 'pkl')
    save_labelme_dir = os.path.join(save_dir, 'labelme')

    os.makedirs(save_pdf_dir, exist_ok=True)
    os.makedirs(save_pkl_dir, exist_ok=True)
    os.makedirs(save_labelme_dir, exist_ok=True)

    # Extract file names from PDF directory (without extension)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    file_names = [os.path.splitext(f)[0] for f in pdf_files]

    # Random sampling
    sampled_names = random.sample(file_names, sample_num)

    for name in sampled_names:
        src_pdf = os.path.join(pdf_dir, f"{name}.pdf")
        src_pkl = os.path.join(pkl_dir, f"{name}.pkl")

        dst_pdf = os.path.join(save_pdf_dir, f"{name}.pdf")
        dst_pkl = os.path.join(save_pkl_dir, f"{name}.pkl")
        dst_json = os.path.join(save_labelme_dir, f"{name}.json")
        dst_img = os.path.join(save_labelme_dir, f"{name}.jpg")  # Save image in same folder as JSON

        # Copy files
        shutil.copy2(src_pdf, dst_pdf)
        shutil.copy2(src_pkl, dst_pkl)

        # Read and decode pkl data
        with open(src_pkl, 'rb') as f:
            compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(depressed_pickle)

        gt_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)
        gt_label = pkl_data['label']

        height, width = gt_img.shape[:2]

        # Save image to labelme/{name}.jpg
        cv2.imwrite(dst_img, gt_img)

        # Convert to LabelMe format
        labelme_data = {
            "version": "4.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": f"{name}.jpg",
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        for item in gt_label:
            x1, y1, x2, y2, class_id, class_name = item[:6]

            if class_name == 'table-cell':
                continue
            if class_name in ['key-value-region', 'code', 'form', 'checkbox-selected', 'checkbox-unselected']:
                class_name = 'text'

            shape = {
                "label": class_name,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            labelme_data["shapes"].append(shape)

        # Save JSON file with UTF-8 BOM for Windows compatibility
        with open(dst_json, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

        # Optionally remove original files
        if remove_original:
            os.remove(src_pdf)
            os.remove(src_pkl)


def clean_docling_dataset(cfg):
    pkl_dir = cfg['clean_docling_dataset']['pkl_dir']
    pdf_dir = cfg['clean_docling_dataset']['pdf_dir']

    def is_contained(inner, outer):
        return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

    def get_tables_from_pdf(pdf_path, filename):
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            tables = page.find_tables()
            return {
                "filename": filename,
                "bboxes": [
                    list(fitz.IRect(table.bbox)) + [0, "table"] for table in tables.tables
                ]
            }
        except Exception as e:
            print(f"Error reading PDF {filename}: {e}")
            return {"filename": filename, "bboxes": []}

    pkl_file_list = os.listdir(pkl_dir)
    for file_idx, pkl_file in enumerate(pkl_file_list):
        if file_idx % 100 == 0:
            print(f'{file_idx} / {len(pkl_file_list)} ...')

        pkl_path = f'{pkl_dir}/{pkl_file}'
        pdf_path = f'{pdf_dir}/{pkl_file[:-4]}.pdf'

        # Load and decompress pkl data
        with open(pkl_path, 'rb') as f:
            compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(depressed_pickle)

        gt_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)
        gt_label = pkl_data['label']

        # Replace table labels with those found by pymupdf
        table_result = get_tables_from_pdf(pdf_path, pkl_file)
        if table_result['bboxes']:
            # Remove existing table labels
            gt_label = [label for label in gt_label if len(label) >= 6 and label[5] != 'table']
            # Add new table labels from PDF
            gt_label.extend(table_result['bboxes'])

        cleaned_labels = []
        picture_indices_to_remove = set()

        # Remove overlapping or nested picture labels
        for i, label_i in enumerate(gt_label):
            if len(label_i) < 6:
                continue
            box_i = label_i[:4]
            cls_name_i = label_i[5]

            if cls_name_i not in ['table', 'picture']:
                continue

            for j, label_j in enumerate(gt_label):
                if i == j or len(label_j) < 6:
                    continue
                box_j = label_j[:4]
                cls_name_j = label_j[5]

                if cls_name_j not in ['table', 'picture']:
                    continue

                iou = get_iou(box_i, box_j)
                if iou >= 0.5 or is_contained(box_i, box_j) or is_contained(box_j, box_i):
                    if cls_name_i == 'picture':
                        picture_indices_to_remove.add(i)
                    elif cls_name_j == 'picture':
                        picture_indices_to_remove.add(j)

        for idx, label in enumerate(gt_label):
            if idx not in picture_indices_to_remove:
                cleaned_labels.append(label)

        if '' == 'D':
            # Visualize cleaned labels using OpenCV
            for label in cleaned_labels:
                if len(label) < 6:
                    continue
                x1, y1, x2, y2 = map(int, label[:4])
                cls_name = label[5]
                color = (0, 255, 0) if cls_name == 'table' else (255, 0, 0)
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(gt_img, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Show the image with bounding boxes
            cv2.imshow('Data', gt_img)
            cv2.waitKey(10)
        else:
            # Save cleaned data back to pkl
            pkl_data['label'] = cleaned_labels
            compressed_pickle = blosc.compress(pickle.dumps(pkl_data))
            with open(pkl_path, 'wb') as f:
                f.write(compressed_pickle)




def convert_labelme_to_eval_gt(src_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for filename in os.listdir(src_dir):
        if not filename.endswith('.json'):
            continue

        src_path = os.path.join(src_dir, filename)
        with open(src_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        bboxes = []
        for shape in data.get('shapes', []):
            if shape.get('shape_type') != 'rectangle':
                continue

            label = shape.get('label', '')
            points = shape.get('points', [])
            if len(points) != 2:
                continue

            x1, y1 = points[0]
            x2, y2 = points[1]
            x_min, y_min = int(min(x1, x2)), int(min(y1, y2))
            x_max, y_max = int(max(x1, x2)), int(max(y1, y2))

            bboxes.append([x_min, y_min, x_max, y_max, label])

        image_path = data.get('imagePath', '')
        base_name = os.path.splitext(image_path)[0]
        output_filename = f"{base_name}.pdf"

        output_data = {
            "filename": output_filename,
            "bboxes": bboxes
        }

        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)


def make_lmdb_dataset_same_keys(cfg):
    """
    Synchronizes keys by removing keys from LMDB 2 that are NOT present in LMDB 1.
    Uses hex string conversion for stable comparison of bytes keys.
    """

    lmdb_dirs = cfg['make_lmdb_dataset_same_keys']['lmdb_dirs']
    if len(lmdb_dirs) != 2:
        raise ValueError("This function requires exactly two LMDB paths.")

    lmdb1_dir = lmdb_dirs[0]
    lmdb2_dir = lmdb_dirs[1]

    print(f"Base LMDB (LMDB 1): {lmdb1_dir}")
    print(f"Target LMDB (LMDB 2): {lmdb2_dir}")

    # --- Step 1: Read keys from LMDB 1 (Base Keys) ---
    print("\nStep 1: Reading base key set from LMDB 1...")

    env1 = lmdb.open(lmdb1_dir, readonly=True, lock=False)
    with env1.begin(write=False) as txn:
        keys_bytes = txn.get(b'__keys__')
        if keys_bytes is None:
            raise ValueError(f"LMDB 1 at {lmdb1_dir} does not contain '__keys__'")

        keys_loaded = pickle.loads(keys_bytes)

        # CRITICAL FIX: Convert bytes keys to stable hex strings for set comparison
        base_keys_hex_set = set(k.hex() if isinstance(k, bytes) else k for k in keys_loaded)

    env1.close()
    print(f"  > Found {len(base_keys_hex_set)} base keys in LMDB 1.")

    # --- Step 2: Identify keys in LMDB 2 that are not in LMDB 1 ---
    print("\nStep 2: Identifying non-shared keys in LMDB 2...")

    env2_ro = lmdb.open(lmdb2_dir, readonly=True, lock=False)
    with env2_ro.begin(write=False) as txn:
        keys_bytes = txn.get(b'__keys__')
        if keys_bytes is None:
            raise ValueError(f"LMDB 2 at {lmdb2_dir} does not contain '__keys__'")

        lmdb2_keys_list_raw = pickle.loads(keys_bytes)

        # Create a consistent list of byte keys for deletion and a list of hex keys for comparison
        lmdb2_keys_list_bytes = [k if isinstance(k, bytes) else k.encode('utf-8') for k in lmdb2_keys_list_raw]
        lmdb2_keys_list_hex = [k.hex() for k in lmdb2_keys_list_bytes]

    env2_ro.close()

    # Map hex key back to its original byte key for deletion
    hex_to_byte_map = {k.hex(): k for k in lmdb2_keys_list_bytes}

    # Identify keys to be removed (Comparison is now robust)
    keys_to_remove_hex = [k_hex for k_hex in lmdb2_keys_list_hex if k_hex not in base_keys_hex_set]

    # Identify keys to keep (in hex)
    keys_to_keep_hex = [k_hex for k_hex in lmdb2_keys_list_hex if k_hex in base_keys_hex_set]

    # Convert hex lists to byte lists for LMDB operations
    keys_to_remove_bytes = [hex_to_byte_map[k_hex] for k_hex in keys_to_remove_hex]
    keys_to_keep_bytes = [hex_to_byte_map[k_hex] for k_hex in keys_to_keep_hex]

    print(f"  > Total keys in LMDB 2: {len(lmdb2_keys_list_bytes)}")
    print(f"  > Keys to REMOVE from LMDB 2: {len(keys_to_remove_bytes)}")
    print(f"  > Keys to KEEP in LMDB 2: {len(keys_to_keep_bytes)}")

    if not keys_to_remove_bytes:
        print("\n? No keys need to be removed. LMDB 2 is a subset of LMDB 1.")
        return

    # --- Step 3: Remove non-shared keys and update '__keys__' in LMDB 2 ---
    print("\nStep 3: Deleting keys and updating '__keys__' in LMDB 2...")

    env2 = lmdb.open(lmdb2_dir, map_size=1099511627776)
    txn = env2.begin(write=True)
    removed_count = 0

    try:
        # Sequential deletion with progress bar
        print(f"  - Deleting {len(keys_to_remove_bytes)} data entries...")
        for key_byte in tqdm(keys_to_remove_bytes, desc="  -> Deleting"):
            # Delete data entry (key-value pair)
            if txn.delete(key_byte):
                removed_count += 1

        # Update the '__keys__' entry with only the kept keys (bytes list)
        print("  - Updating '__keys__' list...")
        txn.put(b'__keys__', pickle.dumps(keys_to_keep_bytes))

        # Commit the transaction to save changes
        txn.commit()

        print(f"\n? Synchronization Complete for LMDB 2!")
        print(f"    > Successfully removed {removed_count} keys and data entries.")
        print(f"    > Final key count in LMDB 2: {len(keys_to_keep_bytes)}.")

    except Exception as e:
        txn.abort()
        print(f"\n? Error processing LMDB 2: {e}")
    finally:
        env2.close()


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)

    if len(sys.argv) < 2:
        print("Usage: python make_data.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "rb") as f:
        cfg = anyconfig.load(f)

    task_names = cfg['task_name'].split(',')

    for task_name in task_names:
        task_name = task_name.strip()
        if task_name == 'make_doclaynet_json':
            make_doclaynet_json(cfg)
        elif task_name == 'make_graph_data':
            make_graph_data(cfg)
        elif task_name == 'make_doclaynet_json_add_custom_feature':
            make_doclaynet_json_add_custom_feature(cfg)
        elif task_name == 'save_json_dataset_to_lmdb':
            save_json_dataset_to_lmdb(cfg)
        elif task_name == 'safe_save_json_dataset_to_lmdb':
            safe_save_json_dataset_to_lmdb(cfg)
        elif task_name == 'make_mupdf_xml':
            make_mupdf_xml(cfg)
        elif task_name == 'visualize_lmdb_dataset':
            visualize_lmdb_dataset(cfg)
        elif task_name == 'make_table_modified_pkl_dataset':
            make_table_modified_pkl_dataset(cfg)
        elif task_name == 'fix_bbox_size_of_graph_data':
            fix_bbox_size_of_graph_data()
        elif task_name == 'modify_gt_dataset_by_docling':
            modify_gt_dataset_by_docling(cfg)
        elif task_name == 'make_gt_dataset_by_docling':
            make_gt_dataset_by_docling(cfg)
        elif task_name == 'sampling_docling_dataset':
            sampling_docling_dataset(cfg)
        elif task_name == 'clean_docling_dataset':
            clean_docling_dataset(cfg)
        elif task_name == 'make_lmdb_dataset_same_keys':
            make_lmdb_dataset_same_keys(cfg)
        else:
            raise Exception(f'Invalid task_name = {task_name}')
