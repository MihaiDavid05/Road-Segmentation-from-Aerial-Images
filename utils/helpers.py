import torch
import numpy as np
from PIL import Image


def crop_image(img, w=400, h=400):
    """
    # Split image into 400x400 patches
    Args:
        img: Initial image
        w: Width of patch
        h: Size of patch

    Returns: 4 patches

    """
    # Get initial image size
    orig_width, orig_height = img.size

    # Setting the points for cropped image
    crop1 = (0, 0, w, h)
    crop2 = (orig_width - w, 0, orig_width, h)
    crop3 = (0, orig_height - h, w, orig_height)
    crop4 = (orig_width - w, orig_height - h, orig_width, orig_height)

    # Cropped image of above dimension
    im1 = img.crop(crop1)
    im2 = img.crop(crop2)
    im3 = img.crop(crop3)
    im4 = img.crop(crop4)

    return [im1, im2, im3, im4]


def overlay_masks(masks, orig_img, mode='avg'):
    """
    Mask patches are in next order: upper left, upper right, lower left, lower right.
    Aggregate patches probabilities masks into final mask.
    Args:
        masks: Patches of images.
        orig_img: Original image.
        mode: Way of combining the patches - average or maximum between overlapping areas.

    Returns: A full sized mask.

    """
    # Swap channels order for each masks
    masks = [x.transpose((1, 2, 0)) for x in masks]
    # Get initial test image size
    orig_w, orig_h = orig_img.size
    w, h = masks[0].shape[0], masks[0].shape[1]

    if len(masks) == 1:
        return masks
    else:
        # Build divider image. Each pixel will represent the number of overlapping patches in that point.
        divider = np.ones((orig_w, orig_h))
        divider[(orig_h - h):h, :] = 2
        divider[:, (orig_w - w):w] = 2
        divider[(orig_h - h):h, (orig_w - w):w] = 4
        # Pad patches to same size
        masks[0] = np.pad(masks[0], ((0, orig_h - h), (0, orig_w - w), (0, 0)), 'constant', constant_values=(0, 0))
        masks[1] = np.pad(masks[1], ((0, orig_h - h), (orig_w - w, 0), (0, 0)), 'constant', constant_values=(0, 0))
        masks[2] = np.pad(masks[2], ((orig_h - h, 0), (0, orig_w - w), (0, 0)), 'constant', constant_values=(0, 0))
        masks[3] = np.pad(masks[3], ((orig_h - h, 0), (orig_w - w, 0), (0, 0)), 'constant', constant_values=(0, 0))

        class_masks = []
        for i in range(2):
            mask = np.stack([m[:, :, i] for m in masks], axis=-1)
            if mode == 'avg':
                # Average the patches
                mask = np.sum(mask, axis=-1)
                mask = mask / divider
            else:
                # Take maximum value in overlapping patches
                mask = np.max(mask, axis=-1)
            # Gather masks
            class_masks.append(mask)
        # Build numpy array
        proba_mask = np.stack(class_masks, axis=-1).transpose((2, 0, 1))

        return proba_mask


def value_to_class(v, foreground_thresh):
    """
    Assign a class label to a patch
    Args:
        v: One-hot encoded patch
        foreground_thresh: Threshold for assign a patch to foreground or background.

    Returns: Class label

    """
    df = np.sum(v)
    if df > foreground_thresh:
        # Road
        return 1
    else:
        # Background
        return 0


def mask_to_image(mask, from_patches=False):
    """
    Transform probability mask to image.
    Args:
        mask: Probability mask
        from_patches: Whether the input came from multiple patches or as an entire image

    Returns: Image

    """
    if mask.ndim == 2:
        if not from_patches:
            mask = 1 - mask
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255).astype(np.uint8), 'L')


def load_pretrain_model(net, config):
    """
    Load VGG13 pretrain model
    Args:
        net: Network
        config: Config dictionary

    Returns: Network with pretrained weights

    """
    # Get parameters of VGG13 and own network
    pretrained_vgg_params = torch.load(config.pretrain)
    net_params = net.state_dict()

    # Load weights according to network configuration
    if config.bilinear:
        if config.cut_last_convblock:
            good_vgg_keys = list(key for key in pretrained_vgg_params.keys() if key.startswith('features') and
                                 int(key.split('.')[1]) not in [21, 22, 24, 25, 28, 29, 31, 32])
            good_net_keys = list(key for key in net_params.keys() if (key.startswith('inc') or key.startswith('down'))
                                 and 'num_batches_tracked' not in key and 'down4' not in key and 'down3' not in key)
        else:
            good_vgg_keys = list(key for key in pretrained_vgg_params.keys() if key.startswith('features'))
            good_net_keys = list(key for key in net_params.keys() if (key.startswith('inc') or key.startswith('down'))
                                 and 'num_batches_tracked' not in key)
    else:
        good_vgg_keys = list(key for key in pretrained_vgg_params.keys() if key.startswith('features') and
                             int(key.split('.')[1]) not in [28, 29, 31, 32])
        good_net_keys = list(key for key in net_params.keys() if (key.startswith('inc') or key.startswith('down'))
                             and 'num_batches_tracked' not in key and 'down4' not in key)

    # Map remaining own network parameters name to VGG13 parameters name
    key_mapping = dict(zip(good_net_keys, good_vgg_keys))
    new_state_dict = {}

    # Load params and freeze layers
    for net_param_key, net_param_value in net.named_parameters():
        if net_param_key in good_net_keys:
            replaced_param = pretrained_vgg_params[key_mapping[net_param_key]]
            replaced_param.requires_grad = False
            new_state_dict[net_param_key] = replaced_param
        else:
            new_state_dict[net_param_key] = net_param_value

    # Load buffers
    for net_buffer_key, net_buffer_value in net.named_buffers():
        new_state_dict[net_buffer_key] = net_buffer_value

    net.load_state_dict(new_state_dict)

    return net
