from utils.models import *
from utils.datasets import BaseDataset


def build_network(config):
    """
    Build netowrk according to type specified in config.
    """
    if config.model == 'unet':
        net = UNet(n_channels=config.channels, n_classes=config.classes, bilinear=True)
    else:
        raise KeyError("Model specified not implemented")
    return net


def build_dataset(config):
    """
    Build dataset according to configuration file.
    """
    train_dir = config.train_data
    gt_dir = config.gt_data
    resize_test = config.resize_test
    gt_thresh = config.gt_thresh
    pad_train = config.pad_size

    if config.dataset == 'basic':
        dataset = BaseDataset(train_dir, gt_dir, gt_thresh, resize_test, pad_train)
    else:
        raise KeyError("Dataset specified not implemented")
    return dataset
