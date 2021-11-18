import os
import logging
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.image as mpimg


class BaseDataset(Dataset):
    def __init__(self, images_dir, gt_dir, gt_thresh, scale):
        self.images_dir = images_dir
        self.gt_dir = gt_dir
        self.scale = scale
        self.gt_thresh = gt_thresh
        self.ids = [img_name.split('.')[0] for img_name in os.listdir(images_dir)]
        logging.info(f'Dataset created with {len(self.ids)} samples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, scale, gt_thresh=None, is_mask=False):

        # TODO: Check this scaling for test images !
        # w, h = img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # img_ndarray = np.asarray(pil_img)
        img_ndarray = img

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            rimg = img_ndarray - np.min(img_ndarray)
            img_ndarray = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
            img_ndarray = img_ndarray / 255
        else:
            img_ndarray = np.where(img_ndarray > gt_thresh, 1, 0)

        return img_ndarray

    def __getitem__(self, idx):
        name = self.ids[idx]

        # Get idx corresponding image and GT
        gt_mask_file = glob.glob(self.gt_dir + name + '.png')
        img_file = glob.glob(self.images_dir + name + '.png')

        # Read the image and mask
        gt_mask = mpimg.imread(gt_mask_file[0])
        img = mpimg.imread(img_file[0])

        # Preprocess both image and mask
        img = self.preprocess(img, self.scale, is_mask=False)
        gt_mask = self.preprocess(gt_mask, self.scale, self.gt_thresh, is_mask=True)

        # TODO: Here .contiguous() is used
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(gt_mask.copy()).long().contiguous()
        }
