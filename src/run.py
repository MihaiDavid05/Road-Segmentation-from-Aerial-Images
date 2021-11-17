import os
import torch
import argparse
import logging
from utils.config import read_config
from utils.builders import *
from utils.training import train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Configuration filename that you want to use during the run.')
    parser.add_argument('--test', action='store_true', help='Also test and create submission file.')

    return parser.parse_args()


if __name__ == '__main__':
    # Get command line arguments and configuration dictionary
    args = get_args()
    config_path = 'configs/' + args.config_filename + '.yaml'
    config = read_config(config_path)

    # Set file for logging
    log_filename = args.config_filename + '.log'
    os.remove(log_filename)
    logging.basicConfig(filename=args.config_filename + '.log', level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Configuration file used is {args.config_filename}\n')

    # Check for cuda availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device is {device}')

    # Build network according to config file and send it to device
    net = build_network(config)
    net.to(device=device)

    # Build dataset according to config file
    dataset = build_dataset(config)

    # Train network
    train(net, dataset, config, device=device)
