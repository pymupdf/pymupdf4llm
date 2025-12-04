import time

import cv2
import numpy as np

import argparse
import pymupdf.features

def print_rf_features(pdf_path):
    default_flags = (
            0
            | pymupdf.TEXT_PRESERVE_WHITESPACE
            | pymupdf.TEXT_PRESERVE_LIGATURES
            | pymupdf.TEXT_INHIBIT_SPACES
            | pymupdf.TEXT_ACCURATE_BBOXES
    )

    doc = pymupdf.open(pdf_path)
    page = doc[0]
    blocks = page.get_text("dict", flags=default_flags)["blocks"]
    bboxes = []
    texts = []
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
            bboxes.append([x1, y1, x2, y2])
            texts.append(txt)

    stext_flags = (
            0
            | pymupdf.TEXT_PRESERVE_WHITESPACE
            | pymupdf.TEXT_PRESERVE_LIGATURES
            | pymupdf.TEXT_INHIBIT_SPACES
            | pymupdf.TEXT_ACCURATE_BBOXES
            | pymupdf.TEXT_COLLECT_VECTORS
    )
    stext_page = page.get_textpage(flags=stext_flags)
    st_time = time.time()
    for bbox_idx, bbox in enumerate(bboxes):
        print(f'[{bbox_idx+1}] {texts[bbox_idx]} {str(bbox)}'.encode('utf-8'))
        region = pymupdf.mupdf.FzRect(bbox[0], bbox[1], bbox[2], bbox[3])
        features = pymupdf.features.fz_features_for_region(stext_page, region, 0)
        fet_no = 1
        for name in dir(features):
            if not name.startswith('_') and name not in ['this', 'thisown', 'last_char_rect']:
                # print(f'    {name}: {getattr(features, name)!r}')
                # print(f"'{name}',", end='')
                print(f'\t[{fet_no}] {name} : {getattr(features, name)}')
                fet_no += 1
    print(f'Elapsed Time : {time.time() - st_time:.2f} sec')


def _test_layout_model(pdf_path):
    import pymupdf.layout.DocumentLayoutAnalyzer as DocumentLayoutAnalyzer
    from pymupdf.layout.pymupdf_util import create_input_data_from_page

    doc = pymupdf.open(pdf_path)
    da = DocumentLayoutAnalyzer.get_model(feature_set_name='rf')

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["show", "print_rf"])
    parser.add_argument("--file", help="Path to pdf file")
    args = parser.parse_args()

    if args.mode == 'show':
        _test_layout_model(args.file)
    elif args.mode == 'print_rf':
        print_rf_features(args.file)
