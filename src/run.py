import os
import torch
import argparse
import logging
import datetime
from utils.config import *
from utils.builders import *
from utils.trainval import train
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from utils.helpers import load_pretrain_model
import numpy as np

SEED = 45
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Configuration filename that you want to use during the run.')

    return parser.parse_args()


if __name__ == '__main__':
    # Set random number generator seed for numpy
    rng = np.random.RandomState(SEED)

    # Get command line arguments and configuration dictionary
    args = get_args()
    config_path = 'configs/' + args.config_filename + '.yaml'
    config = read_config(config_path)
    config = DotConfig(config)
    config.name = args.config_filename

    # Set file for logging
    log_filename = config.name + '.log'
    log_dir = config.log_dir_path + config.name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_filename = log_dir + '/log.log'
    if os.path.exists(log_filename):
        now = datetime.datetime.now()
        log_filename = log_dir + '/log_' + str(now.minute) + '_' + str(now.second) + '.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Configuration file used is <{config.name}>\n')

    # Check for cuda availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device is {device}')

    # Build network according to config file and send it to device
    net = build_network(config)
    net.to(device=device)

    # Show network format for a 400x400x3 image input
    summary(net, (3, 400, 400), 1)

    # Build dataset according to config file
    dataset = build_dataset(config)

    # Load pretrained VGG13
    if config.pretrain is not None:
        net = load_pretrain_model(net, config)
        logging.info(f'Loaded pretrained weights!\n')

    # Train network
    writer = SummaryWriter(log_dir=log_dir)
    train(net, dataset, config, writer, rng=rng, device=device)

    # TODO: ideas:
    # DONE: See differences in losses CE or FocalLoss (CE IS better for 100 training images, maybe try for augmented dataset) !!!!!!!

    # Look at 3D MININet architecture and UNet++ architecture and dilated convolutions
    # Check gt_thresh importance
    # Check difference between resizing test image before and after network or padding training image in order to have same dimension with test
    # Augmentations
    # Regularization
    # Post process images with erosion and dilation
    # Check learning rate schedulers (especially CLR)
    # Check optimizers (maybe Adam or AdamW)
