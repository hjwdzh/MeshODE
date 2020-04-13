import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
import multiprocessing


class ChamferDist(nn.Module):
    """Compute chamfer distance on GPU using O(n^2) dense distance matrix."""
    def __init__(self, reduction='mean'):
        super(ChamferLoss, self).__init__()
        assert(reduction in ['mean', 'sum', 'none'])
        if reduction == 'mean':
            self.reduce = lambda x: torch.mean(x) 
        elif reduction == 'sum':
            self.reduce = lambda x: torch.sum(x)
        else:
            self.reduce = lambda x: x

    def forward(self, tar, src):
        """
        Args:
          tar: [b, n, 3] points for target
          src: [b, m, 3] points for source
        Returns:
          accuracy, complete, chamfer
        """
        tar = tar.unsqueeze(2)
        src = src.unsqueeze(1)
        diff = tar - src  # [b, n, m, 3]
        dist = torch.norm(diff, dim=-1)  # [b, n, m]
        complete = torch.mean(dist.min(2)[0], dim=1)  # [b]
        accuracy = torch.mean(dist.min(1)[0], dim=1)  # [b]
        
        complete = self.reduce(complete)
        accuracy = self.reduce(accuracy)
        chamfer = 0.5 * (complete + accuracy)
        return accuracy, complete, chamfer


def find_nn_id(args):
    """Eval distance between point sets.
    Args:
      src: [m, 3] np array points for source
      tar: [n, 3] np array points for target
    Returns:
      nn_idx: [m,] np array, index of nearest point in target
    """
    src, tar = args
    tree = cKDTree(tar)
    _, nn_idx = tree.query(src, k=1, n_jobs=1)

    return nn_idx


def find_nn_id_parallel(args):
    """Eval distance between point sets.
    Args:
      src: [m, 3] np array points for source
      tar: [n, 3] np array points for target
      idx: int, batch index
    Returns:
      nn_idx: [m,] np array, index of nearest point in target
      idx
    """
    src, tar, idx = args
    tree = cKDTree(tar)
    _, nn_idx = tree.query(src, k=1, n_jobs=1)

    return idx, nn_idx

    
class ChamferDistKDTree(nn.Module):
    """Compute chamfer distances on CPU using KDTree."""
    
    def __init__(self, reduction='mean', njobs=1):
        """Initialize loss module.
        
        Args:
          reduction: str, reduction method. choice of 'mean', 'sum'.
          njobs: int, number of parallel workers to use during eval.
        """
        super(ChamferDistKDTree, self).__init__()
        self.njobs = njobs
        
        assert(reduction in ['mean', 'sum'])
        if reduction == 'mean':
            self.reduce = lambda x: torch.mean(x, axis=-1) 
        else:  # sum
            self.reduce = lambda x: torch.sum(x, axis=-1)
        if self.njobs != 1:
            self.p = multiprocessing.Pool(njobs)
    
    def find_batch_nn_id(self, src, tar, njobs):
        """Batched eval of distance between point sets.
        Args:
          src: [batch, m, 3] np array points for source
          tar: [batch, n, 3] np array points for target
        Returns:
          batch_nn_idx: [batch, m], np array, index of nearest point in target
        """
        b = src.shape[0]
        if njobs != 1:
#             raise NotImplementedError("Parallel implementation is still under development. "
#                                       "Use njobs = 1 for serial execution instead.")
            src_tar_pairs = tuple(zip(src, tar, range(b)))
#             p = multiprocessing.Pool(njobs)
            result = self.p.map(find_nn_id_parallel, src_tar_pairs)
            seq_arr = np.array([r[0] for r in result])
            batch_nn_idx = np.stack([r[1] for r in result], axis=0)
            batch_nn_idx = batch_nn_idx[np.argsort(seq_arr)]
        else:
            batch_nn_idx = np.stack([find_nn_id((src[i], tar[i])) for i in range(b)], axis=0)
        
        return batch_nn_idx
        
    def forward(self, tar, src):
        """
        Args:
          src: [batch, m, 3] points for source
          tar: [batch, n, 3] points for target
        Returns:
          accuracy: [batch, m], accuracy measure for each point in source
          complete: [batch, n], complete measure for each point in target
          chamfer: [batch,], chamfer distance between source and target
        """
        bs = src.shape[0]
        device = src.device
        src_np = src.data.cpu().numpy()
        tar_np = tar.data.cpu().numpy()
        batch_tar_idx = (torch.from_numpy(self.find_batch_nn_id(src_np, tar_np, njobs=self.njobs))
                         .type(torch.LongTensor)
                         .to(device))  # [b, m]
        batch_src_idx = (torch.from_numpy(self.find_batch_nn_id(tar_np, src_np, njobs=self.njobs))
                         .type(torch.LongTensor)
                         .to(device))  # [b, n]
        batch_tar_idx_b = torch.arange(bs).view(-1, 1).expand(-1, src.shape[1])  # [b, m, 3]
        batch_src_idx_b = torch.arange(bs).view(-1, 1).expand(-1, tar.shape[1])  # [b, n, 3]

        src_to_tar_diff = tar[batch_tar_idx_b, batch_tar_idx] - src  # [b, m, 3]
        tar_to_src_diff = src[batch_src_idx_b, batch_src_idx] - tar  # [b, n, 3]
        accuracy = torch.norm(src_to_tar_diff, dim=-1, keepdim=False)  # [b, m]
        complete = torch.norm(tar_to_src_diff, dim=-1, keepdim=False)  # [b, n]
        chamfer = 0.5 * (self.reduce(accuracy) + self.reduce(complete))
        return accuracy, complete, chamfer