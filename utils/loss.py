import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inp, target):
        if inp.dim() > 2:
            inp = inp.view(inp.size(0), inp.size(1), -1)  # N,C,H,W => N,C,H*W
            inp = inp.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inp = inp.contiguous().view(-1, inp.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(inp)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != inp.data.type():
                self.alpha = self.alpha.type_as(inp.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


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
