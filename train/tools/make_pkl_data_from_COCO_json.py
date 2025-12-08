import os
import json
import blosc
import pickle

import cv2
import random


def save_COCO_data_to_pickle_format(coco_json_file, img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    print('Loading json ...')
    with open(coco_json_file, 'r') as f:
        data = json.load(f)

    img_info_list = {}

    for img_info in data['images']:
        img_info['annotation'] = []
        if img_info['id'] not in img_info_list:
            img_info_list[img_info['id']] = img_info

    for ant_info in data['annotations']:
        img_info = img_info_list[ant_info['image_id']]
        img_info['annotation'].append(ant_info)

    category_map = {}
    for c in data['categories']:
        category_map[c['id']] = c['name'].lower()

    category_colors = {}
    for cat_id in category_map:
        category_colors[cat_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    file_count = 0
    for img_id in img_info_list.keys():
        if file_count % 100 == 0:
            print('%d / %d...' % (file_count, len(img_info_list)))
        img_info = img_info_list[img_id]
        img_path = '%s/%s' % (img_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            print('Invalid Path - %s' % img_path)
            continue

        img = cv2.imread(img_path)

        if img is None:
            print('Invalid Image - %s' % img_path)
            continue

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        enc_ret, encimg = cv2.imencode('.jpg', img, encode_param)
        if not enc_ret:
            raise Exception('Jpeg Encoding Error!')

        ant_list = []
        for ant in img_info['annotation']:
            bbox = list(map(int, ant['bbox']))

            # Change x1, y1, x2, y2 format
            if 'D' == 'D':
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]

            cat_id = ant['category_id']
            bbox.append(cat_id)
            bbox.append(category_map[cat_id])
            ant_list.append(bbox)

            # Visualize data
            if file_count % 100 == 0:
                color = category_colors.get(cat_id, (0, 0, 255))
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

                max_dim = max(img.shape[0], img.shape[1])
                if max_dim > 900:
                    scale = 900 / max_dim
                    new_width = int(img.shape[1] * scale)
                    new_height = int(img.shape[0] * scale)
                    img_resized = cv2.resize(img, (new_width, new_height))
                else:
                    img_resized = img
                cv2.imshow('Result', img_resized)
                cv2.waitKey(10)

        pkl_data = {
            'jpeg': encimg,
            'label': ant_list,
            'info': img_info,
        }
        pickled_data = pickle.dumps(pkl_data)
        compressed_pickle = blosc.compress(pickled_data)
        with open('%s/%s.pkl' % (save_dir, img_info['file_name'][:-4]), 'wb') as f:
            f.write(compressed_pickle)
        file_count += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert COCO JSON to compressed pickle format')
    parser.add_argument('--json', type=str, required=True, help='Path to COCO JSON file')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save pickle files')

    args = parser.parse_args()
    save_COCO_data_to_pickle_format(args.json, args.img_dir, args.save_dir)
