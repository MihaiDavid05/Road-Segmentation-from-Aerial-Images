from utils.helpers import *
import glob
import matplotlib.image as mpimg


def compute_pixel_wise(image, fg_thresh):
    fg = 0
    bg = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > fg_thresh:
                fg += 1
            else:
                bg += 1
    return fg, bg


def pixel_label_count(gt, gt_thresholds, fg_thresh):
    fg_pixel = {}
    bg_pixel = {}

    for gt_thresh in gt_thresholds:
        gt_set = np.where(gt > gt_thresh, 1, 0)

        for image in gt_set:
            fg, bg = compute_pixel_wise(image, fg_thresh)
            if gt_thresh in fg_pixel:
                fg_pixel[gt_thresh] += fg
                bg_pixel[gt_thresh] += bg
            else:
                fg_pixel[gt_thresh] = fg
                bg_pixel[gt_thresh] = bg

        n = fg_pixel[gt_thresh] + bg_pixel[gt_thresh]
        fg_pixel[gt_thresh] /= n
        bg_pixel[gt_thresh] /= n
        print("gt_thresh %0.2f:  fg_pixel = %f, bg_pixel = %f" % (gt_thresh, fg_pixel[gt_thresh], bg_pixel[gt_thresh]))


def patch_label_count(gt, gt_thresholds, fg_thresh):
    fg_patch = {}
    bg_patch = {}

    for gt_thresh in gt_thresholds:
        gt_set = np.where(gt > gt_thresh, 1, 0)
        cropped_images = [img_crop(img, 16, 16) for img in gt_set]
        for patches in cropped_images:
            for patch in patches:
                if np.mean(patch) > fg_thresh:
                    if gt_thresh in fg_patch:
                        fg_patch[gt_thresh] += 1
                    else:
                        fg_patch[gt_thresh] = 1
                else:
                    if gt_thresh in bg_patch:
                        bg_patch[gt_thresh] += 1
                    else:
                        bg_patch[gt_thresh] = 1

        n = fg_patch[gt_thresh] + bg_patch[gt_thresh]
        fg_patch[gt_thresh] /= n
        bg_patch[gt_thresh] /= n

        print("gt_thresh %0.2f: fg_patch = %f, bg_patch = %f" % (gt_thresh, fg_patch[gt_thresh], bg_patch[gt_thresh]))


if __name__ == '__main__':
    # indices = np.arange(1, 101, 1)
    # gts = [mpimg.imread(glob.glob("data/training/groundtruth/*" + str(idx) + '.png')[0]) for idx in indices]
    # gt_threshs = np.arange(0.2, 0.55, 0.05)
    # fg_threshold = 0.25
    #
    # pixel_label_count(gts, gt_threshs, fg_threshold)
    # # gt_thresh 0.20: fg_pixel = 0.205596, bg_pixel = 0.794404
    # # gt_thresh 0.25: fg_pixel = 0.204740, bg_pixel = 0.795260
    # # gt_thresh 0.30: fg_pixel = 0.203802, bg_pixel = 0.796198
    # # gt_thresh 0.35: fg_pixel = 0.202915, bg_pixel = 0.797085
    # # gt_thresh 0.40: fg_pixel = 0.202064, bg_pixel = 0.797936
    # # gt_thresh 0.45: fg_pixel = 0.201271, bg_pixel = 0.798729
    # # gt_thresh 0.50: fg_pixel = 0.200468, bg_pixel = 0.799532
    # patch_label_count(gts, gt_threshs, fg_threshold)
    # # gt_thresh 0.20: fg_patch = 0.263248, bg_patch = 0.736752
    # # gt_thresh 0.25: fg_patch = 0.262400, bg_patch = 0.737600
    # # gt_thresh 0.30: fg_patch = 0.261472, bg_patch = 0.738528
    # # gt_thresh 0.35: fg_patch = 0.260720, bg_patch = 0.739280
    # # gt_thresh 0.40: fg_patch = 0.259872, bg_patch = 0.740128
    # # gt_thresh 0.45: fg_patch = 0.259184, bg_patch = 0.740816
    # # gt_thresh 0.50: fg_patch = 0.258512, bg_patch = 0.741488

    x, y = 3
    print(x, y)
