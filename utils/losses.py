# Thanks to rwightman's timm package
# github.com:rwightman/pytorch-image-models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        return self._compute_losses(x, target).mean()

class AbsLoss(object):
    r"""An abstract class for loss functions. 
    """
    def __init__(self):
        self.record = []
        self.bs = []
    
    def compute_loss(self, pred, gt):
        r"""Calculate the loss.
        
        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.

        Return:
            torch.Tensor: The loss.
        """
        pass
    
    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss
    
    def _average_loss(self):
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record*bs).sum()/bs.sum()
    
    def _reinit(self):
        self.record = []
        self.bs = []

class KL_DivLoss(AbsLoss):
    r"""The Kullback-Leibler divergence loss function.
    """
    def __init__(self):
        super(KL_DivLoss, self).__init__()
        
        self.loss_fn = nn.KLDivLoss()
        
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss

class JSloss(nn.Module):
    """  Compute the Jensen-Shannon loss using the torch native kl_div"""
    def __init__(self, reduction='batchmean'):
        super(JSloss, self).__init__()
        self.red = reduction
        
    def forward(self, input, target):
        net = F.softmax(((input + target)/2.),dim=1)
        return 0.5 * (F.kl_div(input.log(), net, reduction=self.red) + 
                    F.kl_div(target.log(), net, reduction=self.red))

class EMDLoss(nn.Module):
    """EMDLoss class
    """
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p_pred: torch.Tensor, p_true: torch.Tensor):
        assert p_true.shape == p_pred.shape, 'Length of the two distribution must be the same'
        cdf_target = torch.cumsum(p_true, dim=1)  # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_pred, dim=1)  # cdf for values [1, 2, ..., 10]
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1))
        return samplewise_emd.mean()