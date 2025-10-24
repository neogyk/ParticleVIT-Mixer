import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

def batch_entropy(x):
    """Estimate the differential entropy by assuming a gaussian distribution of
    values for different samples of a mini-batch.
    """
    if x.shape[0] <= 1:raise Exception("The batch entropy can only be calculated for |batch| > 1.")
    x = torch.flatten(x, start_dim=1)
    x_std = torch.std(x, dim=0)
    entropies = 0.5 * torch.log(np.pi * np.e * x_std**2 + 1)
    return torch.mean(entropies)


class LBELoss(nn.Module):
    """Computation of the LBE + CE loss.
    See also https://www.wolframalpha.com/input/?i=%28%28x-0.8%29*0.5%29**2+for+x+from+0+to+2+y+from+0+to+0.5
    """

    def __init__(self, num, lbe_alpha=0.5, lbe_alpha_min=0.2, lbe_beta=0.5):
        super(LBELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        lbe_alpha = torch.ones(num) * lbe_alpha
        self.lbe_alpha_p = torch.nn.Parameter(lbe_alpha, requires_grad=True)
        self.lbe_alpha_min = torch.FloatTensor([lbe_alpha_min]).to("cuda")
        self.lbe_beta = lbe_beta

    def lbe_per_layer(self, a, i):
        lbe_alpha_l = torch.abs(self.lbe_alpha_p[i])
        lbe_l = (batch_entropy(a) - torch.maximum(self.lbe_alpha_min, lbe_alpha_l)) ** 2
        return lbe_l * self.lbe_beta

    def __call__(self, output, target):
        output, A = output
        ce = self.ce(output, target)
        if A is None:
            return ce, ce, torch.zeros(1)
        losses = [self.lbe_per_layer(a, i) for i, a in enumerate(A)]
        lbe = torch.mean(torch.stack(losses)) * ce
        return ce + lbe, ce, lbe


class CELoss(nn.Module):
    """Wrapper around the Cross Entropy loss to be compatible with models
    that output intermediate results.
    """
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lbe_alpha_p = torch.zeros(1)
        self.lbe_beta = torch.zeros(1)

    def __call__(self, output, target):
        output, A = output
        ce = self.ce(output, target)
        return ce, ce, 0.0


def label_smoothing_loss(pred, target, smoothing=0.1):
    """Label smoothing loss"""
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(pred, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    #denominator = 1/torch.sqrt(1e-8 + torch.var(target))
    smooth_loss = -log_probs.mean(dim=-1)
    loss = (confidence * nll_loss + smoothing * smooth_loss)
    return loss.mean()


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction="none"):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        ce_loss =  F.cross_entropy(input.to(torch.float32).view(-1, input.shape[-1]),
                                   target.to(torch.long).view(-1), 
                                   reduction="none",
                                   weight=weight,
                                   label_smoothing=0.2
                                   ).view(target.shape)
        pt = torch.exp(-ce_loss)
        focal_loss = (((1 - pt) ** self.gamma) * ce_loss)
        return focal_loss
