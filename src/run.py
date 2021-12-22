import argparse
import logging
from utils.builders import *
from utils.trainval import train
from torch.utils.tensorboard import SummaryWriter
from utils.helpers import load_pretrain_model, setup
import numpy as np

# Setting seeds
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

    # Get configuration and device and set logging
    config, device, log_dir = setup(args, 'log')

    # Build network according to config file and send it to device
    net = build_network(config)
    net.to(device=device)

    # Build dataset according to config file
    dataset = build_dataset(config)

    # Load pretrained VGG13
    if config.pretrain is not None:
        net = load_pretrain_model(net, config)
        logging.info(f'Loaded pretrained weights!\n')

    # Train network
    writer = SummaryWriter(log_dir=log_dir)
    train(net, dataset, config, writer, rng=rng, device=device)
