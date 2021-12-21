import os
import argparse
import logging
import cv2
import imutils
import datetime
import numpy as np
from utils.config import read_config


class Augmenter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.augmentations = cfg["augmentations"]
        self.index = 0
        self.img_out_dir = cfg["train_data"]
        self.gt_out_dir = cfg["gt_data"]
        if not os.path.exists(self.img_out_dir):
            os.mkdir(self.img_out_dir)
        if not os.path.exists(self.gt_out_dir):
            os.mkdir(self.gt_out_dir)

    def __call__(self, img_name, idx):
        img_path = self.cfg["base_train_data"] + img_name
        gt_path = self.cfg["base_gt_data"] + img_name
        # Read as int BGR 0-255
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        img_name = img_name.split('_')[0]
        self.save(img_name, img, gt)
        for aug_key in self.augmentations:
            new_imgs, new_gts = self.__getattribute__(aug_key)(img, gt)
            for new_img, new_gt in zip(new_imgs, new_gts):
                self.save(img_name, new_img, new_gt)
        print("Image {} done !".format(idx))

    def rotate(self, img, gt):
        new_images = []
        new_gts = []
        for degree in self.augmentations["rotate"]:
            new_images.append(imutils.rotate(img, degree))
            new_gts.append(imutils.rotate(gt, degree))
        return new_images, new_gts

    def flip(self, img, gt):
        new_images = []
        new_gts = []
        for flip_type in self.augmentations["flip"]:
            new_images.append(cv2.flip(img, flip_type))
            new_gts.append(cv2.flip(gt, flip_type))
        return new_images, new_gts

    def crop_resize(self, img, gt):
        crop_config = self.augmentations["crop_resize"]
        new_img = img[crop_config["y_start"]:crop_config["y_end"], crop_config["x_start"]:crop_config["x_end"]]
        new_gt = gt[crop_config["y_start"]:crop_config["y_end"], crop_config["x_start"]:crop_config["x_end"]]
        new_img = cv2.resize(new_img, (400, 400), interpolation=cv2.INTER_LINEAR)
        new_gt = cv2.resize(new_gt, (400, 400), interpolation=cv2.INTER_LINEAR)
        return [[new_img], [new_gt]]

    def pixel_transform(self, img, gt):
        pixel_config = self.augmentations["pixel_transform"]
        min_contrast, max_contrast = pixel_config["contrast"]
        min_bright, max_bright = pixel_config["brightness"]
        new_img = np.clip(img * np.random.uniform(min_contrast, max_contrast) +
                          np.random.uniform(min_bright, max_bright), 0, 255)
        return [[new_img], [gt]]

    def gamma_correction(self, img, gt):
        gamma_corrections = self.augmentations["gamma_correction"]
        new_images = []
        new_gts = []
        for gamma in gamma_corrections:
            inv_gamma = 1.0 / gamma
            table = np.array([((k / 255.0) ** inv_gamma) * 255 for k in np.arange(0, 256)]).astype("uint8")
            # apply gamma correction using the lookup table
            res = cv2.LUT(img, table)
            new_images.append(res)
            new_gts.append(gt)
        return new_images, new_gts

    def save(self, img_name, image, gt):
        new_name = img_name + "_" + str(self.index + 1) + self.cfg["extension"]
        image_save_path = self.img_out_dir + new_name
        cv2.imwrite(image_save_path, image)
        gt_save_path = self.gt_out_dir + new_name
        cv2.imwrite(gt_save_path, gt)
        self.index += 1


def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str,
                        help='Augmentation configuration filename that you want to use during the run.')

    return parser.parse_args()


if __name__ == '__main__':
    # Get command line arguments and configuration dictionary
    args = get_args()
    config_path = 'configs/' + args.config_filename + '.yaml'
    config = read_config(config_path)
    config["name"] = args.config_filename

    # Set file for logging
    log_filename = config["name"] + '.log'
    log_dir = config["log_dir_path"] + config["name"]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_filename = log_dir + '/augmentation_log.log'
    if os.path.exists(log_filename):
        os.remove(log_filename)
        now = datetime.datetime.now()
        log_filename = log_dir + '/augmentation_log_' + str(now.minute) + '_' + str(now.second) + '.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Configuration file used is <{config["name"]}>\n')

    # Get all basic training images
    train_images_names = os.listdir(config["base_train_data"])

    # Define an image augmenter
    augmenter = Augmenter(config)

    # Iterate through types of augmentations specified in configuration file
    for i, img_filename in enumerate(train_images_names):
        # Skip image if it is set for exclusion in config file
        if augmenter.cfg["skip_gt"] and i + 1 in augmenter.cfg["skip_gt"]:
            continue
        augmenter(img_filename, i)
