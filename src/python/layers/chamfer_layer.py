import torch
from torch import nn


class ChamferLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ChamferLoss, self).__init__()
        assert(reduction in ['mean', 'sum', 'none'])
        if reduction == 'mean':
            self.reduce = lambda x: torch.mean(x) 
        elif reduction == 'sum':
            self.reduce = lambda x: torch.sum(x)
        else:
            self.reduce = lambda x: x

    def forward(self, trgt, pred):
        """
        Args:
          trgt: [b, n, 3] points for target
          pred: [b, m, 3] points for predictions
        Returns:
          accuracy, complete, chamfer
        """
        trgt = trgt.unsqueeze(2)
        pred = pred.unsqueeze(1)
        diff = trgt - pred  # [b, n, m, 3]
        dist = 0.5 * torch.norm(diff, dim=-1)**2  # [b, n, m]
        complete = torch.mean(dist.min(2)[0], dim=1)  # [b]
        accuracy = torch.mean(dist.min(1)[0], dim=1)  # [b]
        
        complete = self.reduce(complete)
        accuracy = self.reduce(accuracy)
        chamfer = 0.5 * (complete + accuracy)
        return accuracy, complete, chamfer
