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


def mask_to_image(mask):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


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
    good_vgg_keys = list(key for key in pretrained_vgg_params.keys() if key.startswith('features'))
    net_params = net.state_dict()
    good_net_keys = list(key for key in net_params.keys() if (key.startswith('inc') or key.startswith('down'))
                         and 'num_batches_tracked' not in key)
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
