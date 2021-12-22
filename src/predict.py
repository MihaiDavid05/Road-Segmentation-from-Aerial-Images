import argparse
from utils.builders import *
from utils.trainval import predict
from utils.helpers import setup

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
    parser.add_argument('model', type=str, help='Folder name, under checkpoints folder, for the model weights.')
    parser.add_argument('--save', action='store_true', help='Save predicted masks')

    return parser.parse_args()


if __name__ == '__main__':
    # Get command line arguments and configuration dictionary
    args = get_args()

    # Get configuration and device and set logging
    config, device, _ = setup(args, 'predict_log')

    # Build network according to config file and send it to device
    net = build_network(config)
    net.to(device=device)

    # Build dataset according to config file
    dataset = build_dataset(config)

    # Load weights
    # TODO: Here BEST in loc de 30
    checkpoint_path = 'checkpoints/' + args.model + '/checkpoint_30.pth'
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Generate prediction
    predict(args, config, net, dataset, device)
