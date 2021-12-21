import numpy as np
import re


def patch_to_label(patch, foreground_threshold=0.25):
    """
    # assign a label to a patch
    Args:
        patch: Input patch
        foreground_threshold: Threshold for assign a patch to foreground or background.

    Returns: An integer representing road or background

    """
    df = np.mean(patch)
    if df > foreground_threshold:
        # Road
        return 1
    else:
        # Background
        return 0


def mask_to_submission_strings(image_filename, pred, foreground_threshold):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    patch_size = 16
    for j in range(0, pred.shape[1], patch_size):
        for i in range(0, pred.shape[0], patch_size):
            patch = pred[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, foreground_threshold)
            yield "{:03d}_{}_{},{}".format(img_number, j, i, label)


def masks_to_submission(submission_filename, foreground_threshold, image_filenames):
    """
    Converts images into a submission file
    Args:
        submission_filename: Submission file path
        foreground_threshold: Threshold for assign a patch to foreground or background.
        image_filenames: Test image filenames
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for img_filename, pred in image_filenames:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(img_filename, pred, foreground_threshold))
