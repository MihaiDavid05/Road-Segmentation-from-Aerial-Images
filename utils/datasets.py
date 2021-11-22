import os
import logging
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.image as mpimg


class BaseDataset(Dataset):
    def __init__(self, images_dir, gt_dir, gt_thresh, resize_test, pad_train=None):
        self.images_dir = images_dir
        self.gt_dir = gt_dir
        self.resize_test = resize_test
        self.gt_thresh = gt_thresh
        self.pad_train = pad_train
        self.ids = [img_name.split('.')[0] for img_name in os.listdir(images_dir)]
        logging.info(f'Dataset created with {len(self.ids)} samples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, resize_test=False, gt_thresh=0.5, pad_size=None, is_mask=False, is_test=False):
        img_ndarray = img
        if is_test:
            if resize_test:
                img = img.resize((400, 400), resample=Image.NEAREST if is_mask else Image.BICUBIC)
            img_ndarray = np.asarray(img)
            img_ndarray = img_ndarray / 255

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_test:
            if not is_mask:
                rimg = img_ndarray - np.min(img_ndarray)
                img_ndarray = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
                img_ndarray = img_ndarray / 255
            else:
                img_ndarray = np.where(img_ndarray > gt_thresh, 1, 0)

        if is_mask and (np.all(img == 0) or np.all(img == 1)):
            print("Useless image!")

        if not is_test and pad_size is not None:
            if is_mask:
                img_ndarray = np.pad(img_ndarray, pad_width=((pad_size, pad_size), (pad_size, pad_size)),
                                     mode='symmetric')
            else:
                img_ndarray = np.pad(img_ndarray, pad_width=((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                                     mode='symmetric')

        return img_ndarray

    def __getitem__(self, idx):
        name = self.ids[idx]

        # Get idx corresponding image and GT
        gt_mask_file = glob.glob(self.gt_dir + name + '.png')
        img_file = glob.glob(self.images_dir + name + '.png')

        # Read the image and mask
        gt_mask = mpimg.imread(gt_mask_file[0])
        raw_mask = gt_mask.copy()
        if self.pad_train is not None:
            raw_mask = np.pad(raw_mask, pad_width=((self.pad_train, self.pad_train), (self.pad_train, self.pad_train)),
                              mode='symmetric')
        img = mpimg.imread(img_file[0])

        # Preprocess both image and mask
        img = self.preprocess(img, self.resize_test, pad_size=self.pad_train, is_mask=False)
        gt_mask = self.preprocess(gt_mask, self.resize_test, pad_size=self.pad_train, gt_thresh=self.gt_thresh,
                                  is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(gt_mask.copy()).long().contiguous(),
            'raw_mask': torch.as_tensor(raw_mask.copy()).float().contiguous(),
        }
