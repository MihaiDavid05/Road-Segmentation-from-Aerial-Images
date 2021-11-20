import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyDiceLoss:
    def __init__(self, net, inp, target, multiclass):
        self.net = net
        self.inp = inp
        self.target = target
        self.multiclass = multiclass

    def compute(self):
        criterion = nn.CrossEntropyLoss()
        return criterion(self.inp, self.target) + self.dice_loss()

    @classmethod
    def dice_coeff(cls, inp, target, reduce_batch_first=False, epsilon=1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert inp.size() == target.size()
        if inp.dim() == 2 and reduce_batch_first:
            raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {inp.shape})')

        if inp.dim() == 2 or reduce_batch_first:
            inter = torch.dot(inp.reshape(-1), target.reshape(-1))
            sets_sum = torch.sum(inp) + torch.sum(target)
            if sets_sum.item() == 0:
                sets_sum = 2 * inter

            return (2 * inter + epsilon) / (sets_sum + epsilon)
        else:
            # compute and average metric for each batch element
            dice = 0
            for i in range(inp.shape[0]):
                dice += dice_coeff(inp[i, ...], target[i, ...])
            return dice / inp.shape[0]

    @classmethod
    def multiclass_dice_coeff(cls, inp, target, reduce_batch_first=False, epsilon=1e-6):
        # Average of Dice coefficient for all classes
        assert inp.size() == target.size()
        dice = 0
        for channel in range(inp.shape[1]):
            dice += dice_coeff(inp[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

        return dice / inp.shape[1]

    def dice_loss(self):
        self.inp = F.softmax(self.inp, dim=1).float()
        self.target = F.one_hot(self.target, self.net.n_classes).permute(0, 3, 1, 2).float()
        # Dice loss (objective to minimize) between 0 and 1
        assert self.inp.size() == self.target.size()
        fn = self.multiclass_dice_coeff if self.multiclass else self.dice_coeff
        return 1 - fn(self.inp, self.target, reduce_batch_first=True)


class MeanSquaredErrorLoss:
    def __init__(self, net, inp, target):
        self.net = net
        self.inp = inp
        self.target = target

    def compute(self):
        criterion = nn.MSELoss()
        return criterion(self.inp, self.target)


def dice_coeff(inp, target, reduce_batch_first=False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert inp.size() == target.size()
    if inp.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {inp.shape})')

    if inp.dim() == 2 or reduce_batch_first:
        inter = torch.dot(inp.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(inp) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(inp.shape[0]):
            dice += dice_coeff(inp[i, ...], target[i, ...])
        return dice / inp.shape[0]


def multiclass_dice_coeff(inp, target, reduce_batch_first=False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert inp.size() == target.size()
    dice = 0
    for channel in range(inp.shape[1]):
        dice += dice_coeff(inp[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / inp.shape[1]


def dice_loss(inp, target, multiclass=False):
    # Dice loss (objective to minimize) between 0 and 1
    assert inp.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(inp, target, reduce_batch_first=True)
