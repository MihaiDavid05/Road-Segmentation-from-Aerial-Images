import os
import argparse
import logging
import datetime
from utils.config import *
from utils.builders import *
from utils.trainval import predict

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
    parser.add_argument('model', type=str, help='Model corresponding to an experiment.')
    parser.add_argument('--save', action='store_true', help='Save predicted masks')

    return parser.parse_args()


if __name__ == '__main__':
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
    log_filename = log_dir + '/predict_log.log'
    if os.path.exists(log_filename):
        now = datetime.datetime.now()
        log_filename = log_dir + '/predict_log_' + str(now.minute) + '_' + str(now.second) + '.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Configuration file used is <{config.name}>\n')

    # Check for cuda availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device is {device}')

    # Build network according to config file and send it to device
    net = build_network(config)
    net.to(device=device)

    # Build dataset according to config file
    dataset = build_dataset(config)

    # Load weights
    # TODO: Here
    # checkpoint_path = 'checkpoints/' + args.model + '/checkpoint_55.pth'
    checkpoint_path = 'src/checkpoint_best.pth'

    net.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Generate prediction
    predict(args, config, net, dataset, device)
