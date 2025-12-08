import os.path
import threading
import time

import blosc
import lmdb
import pickle

import cv2

import torch
from torch_geometric.data import Dataset
import numpy as np

from queue import Queue
from train.infer.onnx.BoxRFDGNN import get_rf_features
from train.infer.common_util import resize_image, to_gray

seed = int(time.time())


class DocumentLMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, pre_transform=None, cache_size=0, readahead=False, keep_raw_data=False,
                 rf_names=None, data_prob=None):
        super(DocumentLMDBDataset, self).__init__(None, transform, pre_transform)

        if type(lmdb_path) is str:
            self.lmdb_path_list = [lmdb_path]
        elif type(lmdb_path) is list:
            self.lmdb_path_list = lmdb_path

        self.total_samples = 0
        self.lmdb_length_list = []
        self.lmdb_key_list = []

        for lmdb_path in self.lmdb_path_list:
            if not os.path.exists(lmdb_path):
                raise Exception('Invalid lmdb_path = %s' % lmdb_path)
            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=(cache_size > 0 or readahead), meminit=False)
            with env.begin(write=False) as txn:
                db_keys = pickle.loads(txn.get(b'__keys__'))
            env.close()
            self.lmdb_key_list.append(db_keys)
            self.lmdb_length_list.append(len(db_keys))
            print(f'{lmdb_path} ({len(db_keys)} samples)')
            self.total_samples += len(db_keys)

        self.cache_size = cache_size
        self.cache = Queue()
        self.keep_raw_data = keep_raw_data
        self.rf_names = rf_names
        self.fill_thread = None
        self.env = None
        self.txn = None
        self.current_db_idx = None  # Track currently opened LMDB

        # Probability distribution for selecting DBs
        if data_prob is not None:
            if len(data_prob) != len(self.lmdb_path_list):
                raise ValueError("Length of data_prob must match number of LMDB paths")
            prob_sum =  sum(data_prob)
            self.data_prob = [prob / prob_sum for prob in data_prob]
        else:
            self.data_prob = [1.0 / len(self.lmdb_path_list)] * len(self.lmdb_path_list)

    def len(self):
        return self.total_samples

    def load_cache(self):
        cache_data = []

        while len(cache_data) < self.cache_size:
            # Choose DB index based on probability distribution
            db_idx = np.random.choice(len(self.lmdb_path_list), p=self.data_prob)
            lmdb_path = self.lmdb_path_list[db_idx]
            lmdb_length = self.lmdb_length_list[db_idx]

            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)
            txn = env.begin(write=False)
            start_idx = np.random.randint(0, lmdb_length)
            refill_size = max(100, int(self.cache_size * 0.25))
            for i in range(refill_size):
                data_idx = (start_idx + i) % lmdb_length
                key = self.lmdb_key_list[db_idx][data_idx]
                data_bytes = txn.get(key)
                if data_bytes is None:
                    # print(f'Index {key} out of range or missing in {lmdb_path}')
                    continue
                try:
                    data = pickle.loads(blosc.decompress(data_bytes))
                    data = self.post_process_data(data)
                except Exception as ex:
                    print(ex)
                    continue
                cache_data.append(data)
            np.random.shuffle(cache_data)
            env.close()
        return cache_data

    def fill_cache_data(self):
        while True:
            if self.cache.qsize() < self.cache_size:
                cache_data = self.load_cache()
                for d in cache_data:
                    self.cache.put(d)
            else:
                time.sleep(1)
        # print(self.cache.qsize())

    def post_process_data(self, data):
        raw_data = data.raw_data
        # data.nn_feature = torch.tensor(raw_data['nn_features'], dtype=torch.float)
        rf_feature = []
        if self.rf_names is not None:
            for idx, custom_feature in enumerate(raw_data['custom_features']):
                f = get_rf_features(custom_feature, self.rf_names)
                rf_feature.append(f)
        data.rf_features = torch.tensor(rf_feature, dtype=torch.float)
        data.text_patterns = torch.tensor(raw_data['text_patterns'], dtype=torch.float)
        img_features = raw_data['image_features']

        if img_features is not None:
            data.img_features = torch.tensor(img_features, dtype=torch.float)
        else:
            data.img_features = torch.tensor([], dtype=torch.float)

        if 'image_data' in raw_data:
            img_gray = cv2.imdecode(raw_data['image_data'], cv2.IMREAD_GRAYSCALE)
            img_gray = img_gray.astype(np.float32)
            # img_resized = resize_image(img, (500, 500))
            # img_gray = to_gray(img_resized)

            img_min, img_max = img_gray.min(), img_gray.max()
            if img_max > img_min:
                img_norm = (img_gray - img_min) / (img_max - img_min)
            else:
                img_norm = np.zeros_like(img_gray)

            img_norm = np.expand_dims(img_norm, axis=-1)
            img_chw = np.transpose(img_norm, (2, 0, 1))
            img_bchw = np.expand_dims(img_chw, axis=0)
            data.image_data = torch.from_numpy(img_bchw).float()
        else:
            data.image_data = torch.tensor([], dtype=torch.float)

        if not self.keep_raw_data:
            del data.raw_data
        return data

    def get(self, idx):
        if self.cache_size == 0:
            # Map global index to local LMDB index
            for db_idx, db_len in enumerate(self.lmdb_length_list):
                if idx < db_len:
                    lmdb_path = self.lmdb_path_list[db_idx]
                    local_idx = idx
                    break
                else:
                    idx -= db_len
            else:
                raise IndexError(f'Global index {idx} out of range (total={self.total_samples}).')

            # Open LMDB environment if not already opened or if switching to a different DB
            if self.env is None or not hasattr(self, "current_db_idx") or self.current_db_idx != db_idx:
                if self.env is not None:
                    self.env.close()
                self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)
                self.txn = self.env.begin(write=False)
                self.current_db_idx = db_idx

            while True:
                try:
                    key = self.lmdb_key_list[db_idx][local_idx]
                    data_bytes = self.txn.get(key)
                    if data_bytes is None:
                        raise IndexError(f'Index {local_idx} out of range or missing in LMDB dataset {lmdb_path}.')
                    data = pickle.loads(blosc.decompress(data_bytes))
                    data = self.post_process_data(data)
                    return data
                except Exception as ex:
                    print(f'{local_idx} in {lmdb_path}: {str(ex)}')
                    # Fallback: sample a random index globally
                    rand_idx = np.random.randint(0, self.total_samples)
                    return self.get(rand_idx)

        else:
            while True:
                if self.cache.empty():
                    if self.fill_thread is None:
                        self.fill_thread = threading.Thread(target=self.fill_cache_data)
                        self.fill_thread.start()
                data = self.cache.get()
                if data.x.shape[0] > 5000:
                    continue
                else:
                    break
        return data

    def close(self):
        """Release all LMDB resources and terminate background threads."""
        if self.env is not None:
            self.env.close()
            self.env = None
        if self.fill_thread is not None:
            self.fill_thread.join()
            self.fill_thread = None

    def __del__(self):
        """Ensure resources are released when the object is garbage-collected."""
        self.close()
