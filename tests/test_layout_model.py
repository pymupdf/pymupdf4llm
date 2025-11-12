
import cv2
import numpy as np

import argparse
import pymupdf.features

def print_rf_features():
    rect = pymupdf.mupdf.FzRect(pymupdf.mupdf.FzRect.Fixed_INFINITE)
    stext_page = pymupdf.mupdf.FzStextPage(rect)    # mediabox
    region = pymupdf.mupdf.FzRect(0, 0, 100, 100)
    features = pymupdf.features.fz_features_for_region(stext_page, region, 0)

    fet_no = 1
    print(f'{features=}:')
    for name in dir(features):
        if not name.startswith('_') and name != 'thisown':
            # print(f'    {name}: {getattr(features, name)!r}')
            # print(f"'{name}',", end='')
            print(f'{fet_no} : {name}')
            fet_no += 1


def test_layout_model(pdf_path):
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
    parser.add_argument("mode", choices=["show"])
    parser.add_argument("file", help="Path to pdf file")
    args = parser.parse_args()

    if args.mode == 'show':
        test_layout_model(args.file)
