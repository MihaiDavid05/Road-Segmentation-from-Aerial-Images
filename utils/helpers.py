import torch
import numpy as np
from PIL import Image


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def crop_image(img, w=400, h=400):
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
    Args:
        masks: Patches of images.
        orig_img: Original image.
        mode: Way of combining the patches - average or maximum between overlapping areas.

    Returns: A full sized image (ndarray)

    """
    masks = [x.transpose((1, 2, 0)) for x in masks]
    orig_w, orig_h = orig_img.size
    w, h = masks[0].shape[0], masks[0].shape[1]

    if len(masks) == 1:
        return masks
    else:
        divider = np.ones((orig_w, orig_h))
        divider[(orig_h - h):h, :] = 2
        divider[:, (orig_w - w):w] = 2
        divider[(orig_h - h):h, (orig_w - w):w] = 4
        masks[0] = np.pad(masks[0], ((0, orig_h - h), (0, orig_w - w), (0, 0)), 'constant', constant_values=(0, 0))
        masks[1] = np.pad(masks[1], ((0, orig_h - h), (orig_w - w, 0), (0, 0)), 'constant', constant_values=(0, 0))
        masks[2] = np.pad(masks[2], ((orig_h - h, 0), (0, orig_w - w), (0, 0)), 'constant', constant_values=(0, 0))
        masks[3] = np.pad(masks[3], ((orig_h - h, 0), (orig_w - w, 0), (0, 0)), 'constant', constant_values=(0, 0))

        fgr_mask = np.stack([m[:, :, 1] for m in masks], axis=-1)
        if mode == 'avg':
            fgr_mask = np.sum(fgr_mask, axis=-1)
            fgr_mask = fgr_mask / divider
        else:
            fgr_mask = np.max(fgr_mask, axis=-1)

        bgr_mask = np.stack([m[:, :, 0] for m in masks], axis=-1)
        if mode == 'avg':
            bgr_mask = np.sum(bgr_mask, axis=-1)
            bgr_mask = bgr_mask / divider
        else:
            bgr_mask = np.max(bgr_mask, axis=-1)

        proba_mask = np.stack([bgr_mask, fgr_mask], axis=-1).transpose((2, 0, 1))

        return proba_mask


def value_to_class(v, foreground_thresh):
    df = np.sum(v)
    if df > foreground_thresh:
        # Road
        return 1
    else:
        # Background
        return 0


def get_binary_patch_mask(mask, fore_thresh):
    patches = img_crop(mask, 16, 16)
    patches = np.asarray(patches)
    labels = np.asarray(
        [value_to_class(np.mean(patches[i]), foreground_thresh=fore_thresh) for i in range(patches.shape[0])])
    return labels


def submission_format_metric(pred_mask, gt_mask, fore_thresh):
    gt_mask = gt_mask.squeeze().detach().cpu().numpy()
    labels = get_binary_patch_mask(gt_mask, fore_thresh)

    pred_mask = pred_mask.detach().cpu().numpy()
    preds = get_binary_patch_mask(pred_mask, fore_thresh)
    # print(np.all(preds == 0))
    # print(np.all(preds == 1))
    # print(np.all(labels == 0))
    # print(np.all(labels == 1))
    f1 = get_f1(preds, labels)

    return f1


def get_precision_recall_accuracy(preds, labels):
    """
    Compute precision, recall and accuracy.
    :param preds: Predictions.
    :param labels: Labels or ground truths.
    :return: Precision, recall, accuracy.
    """
    preds = np.ravel(preds)
    labels = np.ravel(labels)
    # Compute true positives, false positives, false negatives and true negatives
    tp = len(np.where(np.logical_and(preds == 1, labels == 1))[0])
    fp = len(np.where(np.logical_and(preds == 1, labels == 0))[0])
    fn = len(np.where(np.logical_and(preds == 0, labels == 1))[0])
    tn = len(np.where(np.logical_and(preds == 0, labels == 0))[0])

    # Compute precision recall and accuracy
    if tp + fp == 0:
        print("Problem precision")
        precision = 0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        print("Problem recall")
        recall = 0
    else:
        recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return precision, recall, accuracy


def get_f1(preds, labels):
    """
    Compute F1 score.
    :param preds: Predictions.
    :param labels: Labels or ground truths.
    :return: F1 score.
    """
    precision, recall, _ = get_precision_recall_accuracy(preds, labels)
    if precision == 0 and recall == 0:
        return np.nan
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score


def mask_to_image(mask, from_patches=False):
    # mask = mask[::-1, :, :]
    if mask.ndim == 2:
        if not from_patches:
            mask = 1 - mask
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        z = np.argmax(mask, axis=0) * 255 #/ mask.shape[0]
        return Image.fromarray(z.astype(np.uint8), 'L')


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def load_pretrain_model(net, config):
    pretrained_vgg_params = torch.load(config.pretrain)
    net_params = net.state_dict()
    # config.cut_last_convblock

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
    if not config.bilinear:
        good_vgg_keys = list(key for key in pretrained_vgg_params.keys() if key.startswith('features') and
                             int(key.split('.')[1]) not in [28, 29, 31, 32])
        good_net_keys = list(key for key in net_params.keys() if (key.startswith('inc') or key.startswith('down'))
                             and 'num_batches_tracked' not in key and 'down4' not in key)

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


def cross_entropy(preds, gt):
    preds = preds.squeeze().detach().cpu().numpy()
    gt = gt.squeeze().detach().cpu().numpy()
    return -np.sum(preds * np.log(gt + 0.000000000000001))
