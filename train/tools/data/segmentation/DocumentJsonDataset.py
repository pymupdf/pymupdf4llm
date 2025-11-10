import os

import blosc
import pickle
import random
import cv2
import numpy as np
import torch

from train.infer.pymupdf_util import create_input_data_by_pymupdf

from torch.utils.data import Dataset

def get_crop_n_pad_image(img, target_size):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    # Case 1: Image is smaller than target size ¡æ pad
    if w <= target_w and h <= target_h:
        pad_right = target_w - w
        pad_bottom = target_h - h
        padded_img = cv2.copyMakeBorder(
            img,
            0, pad_bottom,  # top, bottom
            0, pad_right,   # left, right
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # black padding
        )
        return padded_img

    # Case 2: Image is larger than target size ¡æ crop
    crop_x = np.random.randint(0, max(w - target_w + 1, 1))
    crop_y = np.random.randint(0, max(h - target_h + 1, 1))
    cropped_img = img[crop_y:crop_y + target_h, crop_x:crop_x + target_w]

    # If cropped image is smaller than target (edge case), pad
    ch, cw = cropped_img.shape[:2]
    pad_right = target_w - cw
    pad_bottom = target_h - ch
    padded_img = cv2.copyMakeBorder(
        cropped_img,
        0, pad_bottom,
        0, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return padded_img


def get_segmentation_data(img, pkl_data, pdf_boxes, input_size, output_mask_size,
                             class_map, crop_prob, crop_ratio, noise_prob, noise_ratio):
    src_h, src_w = img.shape[:2]

    # --- Resize image according to input_size type ---
    if input_size['type'] == 'Fix':
        # Direct resize to fixed (width, height)
        img_w, img_h = input_size['size']
        img_resized = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        transform_info = {
            "mode": "Fix",
            "scale_x": float(img_w) / src_w,
            "scale_y": float(img_h) / src_h
        }

    elif input_size['type'] == 'CropPad':
        img_w, img_h = input_size['size']
        h, w = img.shape[:2]

        # Initialize crop offsets
        crop_offset_x, crop_offset_y = 0, 0

        # If the image is larger than the target size, crop the center
        if h > img_h or w > img_w:
            start_y = max((h - img_h) // 2, 0)
            start_x = max((w - img_w) // 2, 0)
            end_y = start_y + min(img_h, h)
            end_x = start_x + min(img_w, w)
            img_cropped = img[start_y:end_y, start_x:end_x]

            crop_offset_x, crop_offset_y = start_x, start_y
        else:
            img_cropped = img

        # If the cropped image is smaller, pad it to target size
        h_c, w_c = img_cropped.shape[:2]
        pad_top = 0
        pad_left = 0
        pad_bottom = max(img_h - h_c - pad_top, 0)
        pad_right = max(img_w - w_c - pad_left, 0)

        img_resized = cv2.copyMakeBorder(
            img_cropped,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # black padding
        )

        if h_c < img_h or w_c < img_w:
            scale_x, scale_y = 1.0, 1.0
        else:
            # Note: If no padding occurred (h_c=img_h, w_c=img_w), scale is 1.0
            scale_x = float(img_w) / w_c
            scale_y = float(img_h) / h_c

        # Store transformation info for annotation adjustment
        transform_info = {
            "mode": "CropPad",
            "crop_offset_x": crop_offset_x,
            "crop_offset_y": crop_offset_y,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "scale_x": scale_x,
            "scale_y": scale_y
        }

    elif input_size['type'] == 'Original':
        img_resized = img
        img_h, img_w = img.shape[:2]
        transform_info = {
            "mode": "Original",
            "scale_x": 1.0,
            "scale_y": 1.0
        }
    else:
        raise Exception(f'Invalid input_size type = {input_size}')

    img_h, img_w, _ = img_resized.shape

    # --- Determine final mask size ---
    if output_mask_size['type'] == 'Original':
        mask_w_final = img_w
        mask_h_final = img_h
    elif output_mask_size['type'] == 'DownScale':
        mask_w_final = img_w // output_mask_size['size']
        mask_h_final = img_h // output_mask_size['size']
    else:
        raise Exception(f'Invalid output_mask_size type = {output_mask_size}')

    # --- Initialize intermediate mask for pkl_data annotations ---
    # Temporary mask to mark regions from pkl_data (before PDF boxes refinement)
    mask_pkl_intermediate = np.zeros((img_h, img_w), dtype=np.uint8)

    # --- Fill mask with annotations (pkl_data) ---
    for ant in pkl_data.get('label', []):
        x1, y1, x2, y2 = ant[:4]

        # Annotation coordinate transformation (Original logic)
        if transform_info["mode"] == "CropPad":
            # Adjust for crop offset
            x1 -= transform_info["crop_offset_x"]
            x2 -= transform_info["crop_offset_x"]
            y1 -= transform_info["crop_offset_y"]
            y2 -= transform_info["crop_offset_y"]

            # Adjust for padding
            x1 += transform_info["pad_left"]
            x2 += transform_info["pad_left"]
            y1 += transform_info["pad_top"]
            y2 += transform_info["pad_top"]

            # Apply scaling
            x1 = int(round(x1 * transform_info["scale_x"]))
            x2 = int(round(x2 * transform_info["scale_x"]))
            y1 = int(round(y1 * transform_info["scale_y"]))
            y2 = int(round(y2 * transform_info["scale_y"]))

        else:
            # Fix or Original mode: only apply scaling
            x1 = int(round(x1 * transform_info["scale_x"]))
            x2 = int(round(x2 * transform_info["scale_x"]))
            y1 = int(round(y1 * transform_info["scale_y"]))
            y2 = int(round(y2 * transform_info["scale_y"]))

        # Clip coordinates to valid range
        x1 = max(0, min(img_w - 1, x1))
        x2 = max(0, min(img_w, x2))
        y1 = max(0, min(img_h - 1, y1))
        y2 = max(0, min(img_h, y2))

        txt_label = ant[4]
        if isinstance(txt_label, int):
            txt_label = ant[5]

        if txt_label in class_map:
            cls_idx = int(class_map[txt_label] + 1)  # 0 reserved for background
            if cls_idx > 0 and x2 > x1 and y2 > y1:
                # Mark the region from pkl_data in mask_pkl_intermediate
                mask_pkl_intermediate[y1:y2, x1:x2] = cls_idx
        else:
            print(f'Warn: Invalid class name = {txt_label}')

    # ----------------------------------------------------------------------
    # --- Refine and Merge Mask using PDF Boxes ---
    # The refinement process is to only keep pkl_data GT regions that overlap with PDF boxes,
    # except for the 'picture' and 'table' classes, which are preserved regardless of PDF box overlap.
    mask_img_intermediate = np.zeros((img_h, img_w), dtype=np.uint8)  # Final intermediate mask

    text_cls_idx = int(
        class_map.get('text', -1) + 1)  # 'text' class index (for mask_pdf_intermediate creation)

    # Get class indices for classes to be excluded from PDF BBox filtering
    picture_cls_idx = class_map.get('picture', -1) + 1
    table_cls_idx = class_map.get('table', -1) + 1

    # Check if PDF boxes are present and 'text' class is defined
    if pdf_boxes is not None and len(pdf_boxes) > 0 and text_cls_idx > 0:
        # 1. Create a binary mask for all PDF text boxes
        mask_pdf_intermediate = np.zeros((img_h, img_w), dtype=np.uint8)

        # Transform and fill PDF boxes into mask_pdf_intermediate
        for bbox in pdf_boxes:
            x1, y1, x2, y2 = bbox

            # Apply the same coordinate transformation as for pkl_data
            if transform_info["mode"] == "CropPad":
                # Adjust for crop offset
                x1 -= transform_info["crop_offset_x"]
                x2 -= transform_info["crop_offset_x"]
                y1 -= transform_info["crop_offset_y"]
                y2 -= transform_info["crop_offset_y"]

                # Adjust for padding
                x1 += transform_info["pad_left"]
                x2 += transform_info["pad_left"]
                y1 += transform_info["pad_top"]
                y2 += transform_info["pad_top"]

                # Apply scaling
                x1 = int(round(x1 * transform_info["scale_x"]))
                x2 = int(round(x2 * transform_info["scale_x"]))
                y1 = int(round(y1 * transform_info["scale_y"]))
                y2 = int(round(y2 * transform_info["scale_y"]))

            else:
                # Fix or Original mode: only apply scaling
                x1 = int(round(x1 * transform_info["scale_x"]))
                x2 = int(round(x2 * transform_info["scale_x"]))
                y1 = int(round(y1 * transform_info["scale_y"]))
                y2 = int(round(y2 * transform_info["scale_y"]))

            # Clip coordinates
            x1 = max(0, min(img_w - 1, x1))
            x2 = max(0, min(img_w, x2))
            y1 = max(0, min(img_h - 1, y1))
            y2 = max(0, min(img_h, y2))

            if x2 > x1 and y2 > y1:
                # Mark the PDF box region
                mask_pdf_intermediate[y1:y2, x1:x2] = text_cls_idx

        # 2. Refine the mask using PDF boxes (Intersection logic)

        # Select pixels where pkl_data has a GT (value > 0)
        pkl_gt_pixels = mask_pkl_intermediate > 0

        # Pixels belonging to classes to be preserved (picture and table)
        preserved_pixels = np.zeros((img_h, img_w), dtype=bool)
        if picture_cls_idx > 0:
            preserved_pixels = preserved_pixels | (mask_pkl_intermediate == picture_cls_idx)
        if table_cls_idx > 0:
            preserved_pixels = preserved_pixels | (mask_pkl_intermediate == table_cls_idx)

        # All other GT pixels (excluding preserved classes)
        other_gt_pixels = pkl_gt_pixels & ~preserved_pixels

        # Check overlap condition for non-preserved GT pixels
        overlap_condition_other = mask_pdf_intermediate[other_gt_pixels] > 0

        # Fill the final mask for non-preserved areas
        mask_img_intermediate[other_gt_pixels] = np.where(
            overlap_condition_other,
            # If overlap, keep the original pkl_data GT class index
            mask_pkl_intermediate[other_gt_pixels],
            # Otherwise (no overlap with pdf box), set to background (0)
            0
        )

        # Preserve the 'picture' and 'table' GT areas regardless of PDF box overlap
        mask_img_intermediate[preserved_pixels] = mask_pkl_intermediate[preserved_pixels]

    else:
        # If no pdf_boxes or 'text' class is defined, use the pkl_data mask as is
        mask_img_intermediate = mask_pkl_intermediate

    # ----------------------------------------------------------------------
    # --- Apply random cropping augmentation ---
    if random.random() < crop_prob:
        min_ratio, max_ratio = crop_ratio
        # Calculate size based on area ratio (sqrt for linear dimension)
        target_area_ratio = random.uniform(min_ratio, max_ratio)

        new_w = int(img_w * np.sqrt(target_area_ratio))
        new_h = int(img_h * np.sqrt(target_area_ratio))

        if new_w < img_w and new_h < img_h:
            x_min = random.randint(0, img_w - new_w)
            y_min = random.randint(0, img_h - new_h)
            x_max = x_min + new_w
            y_max = y_min + new_h

            img_cropped = img_resized[y_min:y_max, x_min:x_max]
            mask_cropped = mask_img_intermediate[y_min:y_max, x_min:x_max]

            # Resize back to original target size (img_w, img_h)
            img_resized = cv2.resize(img_cropped, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            mask_img_intermediate = cv2.resize(mask_cropped, (img_w, img_h),
                                               interpolation=cv2.INTER_NEAREST)  # Use NEAREST for mask

    # --- Convert to grayscale and normalize ---
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    min_val, max_val = img_gray.min(), img_gray.max()
    if max_val > min_val:
        img_gray = (img_gray - min_val) / (max_val - min_val)
    else:
        img_gray = np.zeros_like(img_gray, dtype=np.float32)

    # --- Resize mask to final output size ---
    mask_img_final = cv2.resize(mask_img_intermediate, (mask_w_final, mask_h_final), # 오류 해결됨
                                interpolation=cv2.INTER_NEAREST)  # Use NEAREST for mask

    # --- Convert to tensors ---
    img_tensor = torch.from_numpy(img_gray).unsqueeze(0)  # (1, H, W)
    mask_tensor = torch.from_numpy(mask_img_final).long()  # (H, W)

    # --- Add Gaussian noise augmentation ---
    if random.random() < noise_prob:
        noise = torch.randn_like(img_tensor) * noise_ratio
        img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)

    return {'image': img_tensor, 'mask': mask_tensor}


class DocumentJsonDataset(Dataset):
    """
    - image_size: (width, height)  -> used by cv2.resize
    - output_mask_size: (height, width) -> mask shape (H, W)
    - class_map: dict of {label_str: class_idx}  where class_idx in [0, num_cls-1]
    """
    def __init__(self, pkl_dir, pdf_dir, class_map, image_size, output_mask_size, target_type='segmentation',
                 cache_size=0, cache_rate=0.0, crop_prob=0.0, crop_ratio=(0.0, 1.0), noise_prob=0.0, noise_ratio=0.0, pkl_prob=None):
        super().__init__()
        self.cache_size = cache_size
        self.cache_rate = cache_rate
        self.image_size = image_size
        self.output_mask_size = output_mask_size
        self.class_map = class_map
        self.num_cls = len(class_map)
        self.target_type = target_type

        self.data_file_list = []
        if pkl_prob is not None:
            if len(pkl_prob) != len(pdf_dir):
                raise ValueError("Length of data_prob must match number of LMDB paths")
            pkl_sum = sum(pkl_prob)
            self.pkl_prob = [x / pkl_sum for x in pkl_prob]
        else:
            self.pkl_prob = [1.0 / len(pdf_dir)] * len(pdf_dir)

        self.pkl_file_list = []
        self.data_cache = []
        self.crop_prob = crop_prob
        self.crop_ratio = crop_ratio
        self.noise_prob = noise_prob
        self.noise_ratio = noise_ratio

        for path_idx, pkl_path in enumerate(pkl_dir):
            sub_list = []
            for file_name in os.listdir(pkl_path):
                pkl_path = os.path.join(pkl_dir[path_idx], file_name)
                pdf_path = None
                for pdf_sub_dir in pdf_dir:
                    pdf_path = os.path.join(pdf_sub_dir, file_name[:-4] + '.pdf')
                    if os.path.exists(pdf_path):
                        break
                    else:
                        pdf_path = None
                if pdf_path is not None:
                    sub_list.append([pkl_path, pdf_path])

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                print('%s (%d samples)' % (pkl_path, len(sub_list)))
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
                if worker_info.id == 0:
                    print('%s (%d samples)' % (pkl_path, len(sub_list)))
            self.data_file_list.append(sub_list)


    def __len__(self):
        total = 0
        for i in range(len(self.data_file_list)):
            total += len(self.data_file_list[i])
        return total

    def get_reconstruction_data(self, img, to_gray=True):
        img_h, img_w = self.image_size
        h, w = img.shape[:2]

        if to_gray:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Randomly crop the image with a probability of self.crop_prob
        if np.random.rand() < self.crop_prob:
            # Determine crop dimensions based on the random ratio
            crop_ratio_h = np.random.uniform(self.crop_ratio[0], self.crop_ratio[1])
            crop_ratio_w = np.random.uniform(self.crop_ratio[0], self.crop_ratio[1])
            crop_h = int(h * crop_ratio_h)
            crop_w = int(w * crop_ratio_w)

            # Ensure crop dimensions are not zero
            if crop_h == 0: crop_h = 1
            if crop_w == 0: crop_w = 1

            # Calculate random starting coordinates for the crop
            x1 = np.random.randint(0, w - crop_w)
            y1 = np.random.randint(0, h - crop_h)
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            # Crop the image
            img_cropped = img[y1:y2, x1:x2]
            img_resized = cv2.resize(img_cropped, (img_w, img_h))
        else:
            # If not cropped, resize the original image
            img_resized = cv2.resize(img, (img_w, img_h))

        mask = img_resized.copy()
        grid_size = np.random.randint(5, 40)

        # Determine which masking pattern to apply based on a 40/30/30 probability
        mask_prob = np.random.rand()
        grid_h = img_h // grid_size
        grid_w = img_w // grid_size

        if mask_prob < 0.4:  # 40% probability for checkerboard
            for y_grid in range(grid_h):
                for x_grid in range(grid_w):
                    if (y_grid + x_grid) % 2 == 0:
                        x1 = x_grid * grid_size
                        y1 = y_grid * grid_size
                        x2 = x1 + grid_size
                        y2 = y1 + grid_size
                        if len(mask.shape) == 3:  # BGR image
                            cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)
                        else:  # Grayscale image
                            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        elif mask_prob < 0.7:  # 30% probability for vertical grid (0.4 <= mask_prob < 0.7)
            for x_grid in range(grid_w):
                if x_grid % 2 == 0:
                    x1 = x_grid * grid_size
                    y1 = 0
                    x2 = x1 + grid_size
                    y2 = img_h
                    if len(mask.shape) == 3:
                        cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    else:
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        else:  # 30% probability for horizontal grid (0.7 <= mask_prob < 1.0)
            for y_grid in range(grid_h):
                if y_grid % 2 == 0:
                    x1 = 0
                    y1 = y_grid * grid_size
                    x2 = img_w
                    y2 = y1 + grid_size
                    if len(mask.shape) == 3:
                        cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    else:
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        # Convert to tensor and change shape to (C, H, W)
        img_tensor = torch.from_numpy(img_resized).float().div(255.0)
        mask_tensor = torch.from_numpy(mask).float().div(255.0)

        # Permute dimensions if it's a BGR image (H, W, C) -> (C, H, W)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
            mask_tensor = mask_tensor.permute(2, 0, 1)
        # Grayscale images need a channel dimension (H, W) -> (1, H, W)
        elif len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.unsqueeze(0)

        data = {'image': mask_tensor, 'mask': img_tensor}
        return data


    def __getitem__(self, idx):
        # cache logic (optional)
        if 0 < self.cache_size == len(self.data_cache):
            if np.random.rand() < self.cache_rate:
                rnd = np.random.randint(0, self.cache_size)
                return self.data_cache[rnd]

        if self.pkl_prob is not None:
            # Uniformly sample a data file from each list.
            # 1. Randomly select one sub-list (file group) from the main list.
            db_idx = np.random.choice(len(self.data_file_list), p=self.pkl_prob)
            data_list = self.data_file_list[db_idx]
            # 2. Randomly select a file from the chosen sub-list.
            pkl_path, pdf_path = random.choice(data_list)
        else:
            # Find the file corresponding to the given index 'idx' in the entire dataset.
            data_count = []
            for i in range(len(self.data_file_list)):
                if i == 0:
                    data_count.append(len(self.data_file_list[i]))
                else:
                    data_count.append(data_count[-1] + len(self.data_file_list[i]))

            dataset_idx = 0
            for i in range(len(data_count)):
                if idx < data_count[i]:
                    dataset_idx = i
                    break

            data_list = self.data_file_list[dataset_idx]

            # Adjust the index to be relative to the selected sub-list.
            if dataset_idx > 0:
                idx = idx - data_count[dataset_idx - 1]
            pkl_path, pdf_path = data_list[idx]

        # load pkl (blosc compressed)
        with open(pkl_path, 'rb') as f:
            compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(depressed_pickle)

        # decode jpeg buffer -> cv image (BGR)
        cv_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)
        if cv_img is None:
            raise RuntimeError(f'Failed to decode image: {pkl_path}')

        # Use PDF BBox mask filtering
        if '' == 'D':
            try:
                # print(pdf_path)
                pdf_data_dict = create_input_data_by_pymupdf(pdf_path)
                pdf_bboxes = pdf_data_dict['bboxes']
            except Exception as ex:
                print(ex)
                return self.__getitem__(np.random.randint(0, len(self.data_file_list)))

            if len(pdf_bboxes) == 0:
                return self.__getitem__(np.random.randint(0, len(self.data_file_list)))


            pkl_resize_x = cv_img.shape[1] / pdf_data_dict['page_width']
            pkl_resize_y = cv_img.shape[0] / pdf_data_dict['page_height']
            for i in range(len(pdf_bboxes)):
                pdf_bboxes[i][0] *= pkl_resize_x
                pdf_bboxes[i][1] *= pkl_resize_y
                pdf_bboxes[i][2] *= pkl_resize_x
                pdf_bboxes[i][3] *= pkl_resize_y
        else:
            pdf_bboxes = None

        if self.target_type == 'segmentation':
            data = get_segmentation_data(cv_img, pkl_data, pdf_bboxes, self.image_size, self.output_mask_size, self.class_map,
                                         self.crop_prob, self.crop_ratio, self.noise_prob, self.noise_ratio)
        elif self.target_type == 'reconstruction':
            data = self.get_reconstruction_data(cv_img)
        else:
            raise Exception(f'Invalid target_type = {self.target_type}')

        # caching logic
        if self.cache_size > 0:
            if len(self.data_cache) < self.cache_size:
                self.data_cache.append(data)
            else:
                r = np.random.randint(0, self.cache_size)
                self.data_cache[r] = data
        return data


def _make_colormap(num_cls, seed=0):
    """Deterministic random colors for classes. returns (num_cls,3) uint8"""
    rng = np.random.RandomState(seed)
    colors = rng.randint(0, 256, size=(num_cls, 3), dtype=np.uint8)
    # ensure background (class 0) is dark / distinct
    if num_cls > 0:
        colors[0] = np.array([30, 30, 30], dtype=np.uint8)
    return colors

def _tensor_to_bgr_uint8(img_tensor, mean=None, std=None):
    """
    Convert image tensor (C,H,W) float in [0,1] (or normalized) to BGR uint8 HxWx3.
    If mean/std provided, performs un-normalization first.
    """
    if not torch.is_tensor(img_tensor):
        # assume numpy HxWxC or HxW (grayscale)
        arr = img_tensor
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        # if already uint8 assume BGR (cv2 style)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        # if shape is CHW convert to HWC above
        return arr

    img = img_tensor.detach().cpu().float()
    if img.ndim == 3:
        # C,H,W
        if img.shape[0] == 3:
            # optionally unnormalize
            if mean is not None and std is not None:
                mean = torch.tensor(mean, dtype=img.dtype, device=img.device)[:, None, None]
                std = torch.tensor(std, dtype=img.dtype, device=img.device)[:, None, None]
                img = img * std + mean
            img = img.permute(1, 2, 0).numpy()  # H,W,C (RGB)
        else:
            # unexpected channel count -> repeat/clip
            img = img.permute(1, 2, 0).numpy()
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
    else:
        raise ValueError("Unsupported image tensor shape: {}".format(img.shape))

    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    # convert RGB -> BGR for cv2 display
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr

def _mask_to_color(mask_np, colors):
    """
    mask_np: HxW, int class indices
    colors: (num_cls,3) uint8 in BGR order
    returns HxWx3 uint8 color image
    """
    h, w = mask_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    num_cls = colors.shape[0]
    # clip indices to valid range
    mask_clipped = np.clip(mask_np, 0, num_cls - 1)
    for cls in range(num_cls):
        color_img[mask_clipped == cls] = colors[cls]
    return color_img

def show_dataset(dataset, max_samples=20, overlay=True, wait_for_key=True, save_dir=None, mean=None, std=None):
    """
    Visualize dataset samples.
    - dataset: Dataset returning dict with keys 'image' and 'mask'
    - max_samples: stop after this many samples
    - overlay: show overlay image (image + colored mask)
    - wait_for_key: if True, wait for key press between samples (Esc to exit)
    - save_dir: if provided, saves Image/Mask/Overlay into this folder instead of or in addition to showing
    - mean/std: if images were normalized, provide mean/std (list of 3) to unnormalize before display
    """
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    # find num classes if dataset provides it, else infer from mask values on-the-fly
    num_cls = getattr(dataset, 'num_cls', None)
    # fallback colormap: will update dynamically if num_cls is None
    colors = None
    saved = 0

    for idx, item in enumerate(dataset):
        if idx >= max_samples:
            break

        img_item = item.get('image', None) if isinstance(item, dict) else None
        mask_item = item.get('mask', None) if isinstance(item, dict) else None

        # backward compatibility: if dataset returns tuple (img, mask)
        if img_item is None and isinstance(item, (tuple, list)) and len(item) >= 2:
            img_item, mask_item = item[0], item[1]

        if img_item is None or mask_item is None:
            print(f"[show_dataset] skip idx {idx}, invalid item keys.")
            continue

        # convert image tensor -> BGR uint8
        try:
            img_bgr = _tensor_to_bgr_uint8(img_item, mean=mean, std=std)
        except Exception as e:
            print(f"[show_dataset] image conversion failed at idx {idx}: {e}")
            continue

        # convert mask to numpy HxW of ints
        if torch.is_tensor(mask_item):
            mask_np = mask_item.detach().cpu().numpy().astype(np.int32)
        else:
            mask_np = np.array(mask_item, dtype=np.int32)

        # infer num classes and colormap if needed
        if num_cls is None:
            max_cls = int(mask_np.max()) if mask_np.size else 0
            num_cls = max_cls + 1
        if colors is None or colors.shape[0] < num_cls:
            colors = _make_colormap(num_cls, seed=0)  # colors in uint8 BGR

        # colorize mask
        color_mask = _mask_to_color(mask_np, colors)

        # overlay
        if overlay:
            # ensure same size
            if color_mask.shape[:2] != img_bgr.shape[:2]:
                # resize color_mask to image
                color_mask_resized = cv2.resize(color_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                color_mask_resized = color_mask
            overlay_img = cv2.addWeighted(img_bgr, 0.7, color_mask_resized, 0.3, 0)
        else:
            overlay_img = None

        # show windows
        cv2.imshow('Image', img_bgr)
        cv2.imshow('Mask (colored)', color_mask if color_mask.shape == img_bgr.shape else cv2.resize(color_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST))
        if overlay:
            cv2.imshow('Overlay', overlay_img)

        # save if requested
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, f'{idx:04d}_image.jpg'), img_bgr)
            cv2.imwrite(os.path.join(save_dir, f'{idx:04d}_mask.png'), color_mask)
            if overlay:
                cv2.imwrite(os.path.join(save_dir, f'{idx:04d}_overlay.jpg'), overlay_img)
            saved += 1

        if wait_for_key:
            key = cv2.waitKey(0) & 0xFF
            # ESC to exit early
            if key == 27:
                print("[show_dataset] ESC pressed, exiting visualization.")
                break
            # else continue to next sample
        else:
            # small delay to update windows
            cv2.waitKey(100)

    cv2.destroyAllWindows()
    if save_dir:
        print(f"[show_dataset] saved {saved} samples to {save_dir}")


if __name__ == "__main__":
    data_class_names = ['text', 'title', 'picture', 'table', 'list-item', 'page-header', 'page-footer',
                        'section-header', 'footnote', 'caption', 'formula']
    data_class_map = {name: i for i, name in enumerate(data_class_names)}

    dataset = DocumentJsonDataset(
        pkl_dir=['/media/win/PTMP/PDF/DartDataset/dart_발행공시_2401-2509/pkl'],
        pdf_dir=['/media/win/PTMP/PDF/DartDataset/dart_발행공시_2401-2509/PDF'],
        class_map=data_class_map,
        image_size={'type': 'Fix', 'size': [500, 500] },
        output_mask_size={ 'type': 'Original' },
    )
    show_dataset(dataset, max_samples=50, overlay=True, wait_for_key=True, save_dir=None)
