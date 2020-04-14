"""Training script shapenet deformation space experiment.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '../../src/python')
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '../src/python/layers')

from shapenet_dataloader import ShapeNetVertexSampler, ShapeNetMeshLoader
import train_utils as utils
from layers.pointnet_model import PointNetEncoder
from layers.chamfer_layer import ChamferDistKDTree
from layers.deformation_layer import NeuralFlowDeformer

import argparse
import json
import os
import numpy as np
from collections import defaultdict
np.set_printoptions(precision=4)

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter


# Various choices for losses and optimizers
LOSSES = {
    'l1': F.l1_loss,
    'l2': F.l2_loss,
    'huber': F.smooth_l1_loss,
}

OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'rmsprop': optim.RMSprop,
}


def train_or_eval(mode, args, encoder, deformer, chamfer_dist, dataloader, epoch, 
                  global_step, device, logger, writer, optimizer, vis_loader=None):
    """Training / Eval function."""
    modes = ["train", "eval"]
    if not mode in modes:
        raise ValueError(f"mode ({mode}) must be one of {modes}.")
    if mode == 'train':
        encoder.train()
        deformer.train()
    else:
        encoder.eval()
        deformer.eval()
    tot_loss = 0
    count = 0
    criterion = LOSSES(args.loss_type)
    with torch.set_grad_enabled(mode == 'train'):
        for batch_idx, data_tensors in enumerate(dataloader):
            # send tensors to device
            data_tensors = [t.to(device) for t in data_tensors]
            source_pts, target_pts = data_tensors
            bs = len(source_pts)
            optimizer.zero_grad()

            source_latents = encoder(source_pts)
            target_latents = encoder(target_pts)

            src_to_tar = deformer(source_pts[..., :3], source_latents, target_latents)
            tar_to_src = deformer(target_pts[..., :3], target_latents, source_latents)

            # symmetric pair of matching losses
            _, _, src_to_tar_dist = chamfer_dist(src_to_tar, target_pts[..., :3])
            _, _, tar_to_src_dist = chamfer_dist(tar_to_src, source_pts[..., :3])

            src_to_tar_loss = criterion(src_to_tar_dist, torch.zeros_like(src_to_tar_dist))
            tar_to_src_loss = criterion(tar_to_src_dist, torch.zeros_like(tar_to_src_dist))

            loss = src_to_tar_loss + tar_to_src_loss
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_value_(encoder.module.parameters(), args.clip_grad)
            torch.nn.utils.clip_grad_value_(deformer.module.parameters(), args.clip_grad)

            optimizer.step()

            tot_loss += loss.item()
            count += source_pts.size()[0]
            if batch_idx % args.log_interval == 0:
                # logger log
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss Sum: {:.6f}\t"
                    "Loss s2t: {:.6f}\tLoss t2s: {:.6f}".format(
                        epoch, batch_idx * bs, len(dataloader) * bs,
                        100. * batch_idx / len(dataloader), loss.item(),
                        src_to_tar_loss.item(), tar_to_src_loss.item()))
                # tensorboard log
                writer.add_scalar(f'{mode}/loss_sum', loss, global_step=int(global_step))
                writer.add_scalar(f'{mode}/loss_s2t', loss_src_to_tar, global_step=int(global_step))
                writer.add_scalar(f'{mode}/loss_t2s', loss_tar_to_src, global_step=int(global_step))

            global_step += 1
    tot_loss /= count
    
    # visualize a few deformations in tensorboard
    if args.vis_mesh and (vis_loader is not None):
        with torch.set_grad_enabled(False):
            n_meshes = 4
            idx_choices = np.random.permutation(len(vis_loader))[:n_meshes]
            results = []
            for idx in idx_choices:
                data_tensors = vis_loader[idx] 
                data_tensors = [t.unsqueeze(0).to(device) for t in data_tensors]
                vi, fi, vj, fj = data_tensors
                lat_i = encoder(vi)
                lat_j = encoder(vj)
                vi_j = deformer(vi[..., :3], lat_i, lat_j)
                vj_i = deformer(vj[..., :3], lat_j, lat_i)
                accu_i, _, _ = chamfer_dist(vi_j, vj)  # [1, m]
                accu_j, _, _ = chamfer_dist(vj_i, vi)  # [1, n]
                # find the max dist between pairs of original shapes for normalizing colors
                chamfer_dist.set_reduction_method('max')
                _, _, max_dist = chamfer_dist(vi, vj)  # [1,]
                chamfer_dist.set_reduction_method('mean')
                
                # normalize the accuracies wrt. the distance between src and tgt meshes
                ci = utils.colorize_scalar_tensors(accu_i / max_dist, 
                                                   vmin=0, vmax=1, cmap='coolwarm')
                cj = utils.colorize_scalar_tensors(accu_j / max_dist, 
                                                   vmin=0, vmax=1, cmap='coolwarm')
                
                # add colorized mesh to tensorboard
                writer.add_mesh(f'samp{idx}/src', vertices=vi, faces=fi)
                writer.add_mesh(f'samp{idx}/tar', vertices=vj, faces=fj)
                writer.add_mesh(f'samp{idx}/src_to_tar', vertices=vi_j, faces=fi, colors=ci)
                writer.add_mesh(f'samp{idx}/tar_to_src', vertices=vj_i, faces=fj, colors=cj)
    
    return tot_loss


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ShapeNet Deformation Space")
    
    parser.add_argument("--batch_size_per_gpu", type=int, default=16, metavar="N",
                        help="input batch size for training (default: 10)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--pseudo_train_epoch_size", type=int, default=2048, metavar="N",
                        help="number of samples in an pseudo-epoch. (default: 2048)")
    parser.add_argument("--pseudo_eval_epoch_size", type=int, default=128, metavar="N",
                        help="number of samples in an pseudo-epoch. (default: 2048)")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="R",
                        help="learning rate (default: 0.001)")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--data_root", type=str, default="../../data/shapenet",
                        help="path to data folder root (default: ../../data/shapenet)")
    parser.add_argument("--deformer_arch", type=str, choices=["imnet", "vanilla"], default="imnet",
                        help="deformer architecture. (default: imnet)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--log_dir", type=str, required=True, help="log directory for run")
    parser.add_argument("--optim", type=str, default="adam", choices=list(OPTIMIZERS.keys()))
    parser.add_argument("--loss_type", type=str, default="l2", choices=list(LOSSES.keys()))
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint if resume is needed")
    parser.add_argument("-n", "--nsamples", default=2048, type=int,
                        help="number of sample points to draw per shape.")
    parser.add_argument("--lat_dims", default=64, type=int, help="number of latent dimensions.")
    parser.add_argument("--encoder_nf", default=16, type=int,
                        help="number of base number of feature layers in encoder (pointnet).")
    parser.add_argument("--deformer_nf", default=256, type=int,
                        help="number of base number of feature layers in deformer (imnet).")
    parser.add_argument("--pseudo_batch_size", default=1024, type=int,
                        help="size of pseudo batch during eval.")
    parser.add_argument("--lr_scheduler", dest='lr_scheduler', action='store_true')
    parser.add_argument("--no_lr_scheduler", dest='lr_scheduler', action='store_false')
    parser.set_defaults(lr_scheduler=True)
    parser.add_argument("--normals", dest='normals', action='store_true')
    parser.add_argument("--no_normals", dest='normals', action='store_false')
    parser.set_defaults(normals=True)
    parser.add_argument("--visualize_mesh", dest='vis_mesh', action='store_true')
    parser.add_argument("--no_visualize_mesh", dest='vis_mesh', action='store_false')
    parser.set_defaults(vis_mesh=True)
    parser.add_argument("--clip_grad", default=1., type=float,
                        help="clip gradient to this value. large value basically deactivates it.")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    # adjust batch size based on the number of gpus available
    args.batch_size = int(torch.cuda.device_count()) * args.batch_size_per_gpu

    # log and create snapshots
    os.makedirs(args.log_dir, exist_ok=True)
    filenames_to_snapshot = glob("*.py") + glob("*.sh") + glob("layers/*.py")
    utils.snapshot_files(filenames_to_snapshot, args.log_dir)
    logger = utils.get_logger(log_dir=args.log_dir)
    with open(os.path.join(args.log_dir, "params.json"), 'w') as fh:
        json.dump(args.__dict__, fh, indent=2)
    logger.info("%s", repr(args))

    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))

    # random seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create dataloaders
    trainset = ShapeNetVertexSampler(data_root=args.data_root, split="train", category="chair", 
                                     nsamples=5000, normals=args.normals)
    evalset = ShapeNetVertexSampler(data_root=args.data_root, split="val", category="chair",
                                    nsamples=5000, normals=args.normals)

    train_sampler = RandomSampler(trainset, replacement=True, num_samples=args.pseudo_epoch_size)
    eval_sampler = RandomSampler(evalset, replacement=True, num_samples=args.num_log_images)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              sampler=train_sampler, **kwargs)
    eval_loader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             sampler=eval_sampler, **kwargs)
    if args.vis_mesh:
        # for loading full meshes for visualization
        vis_loader = ShapeNetMeshLoader(data_root=args.data_root, split="val", category="chair", 
                                        normals=args.normals)
    else:
        vis_loader = None

    # setup model
    in_feat = 6 if args.normals else 3
    encoder = PointNetEncoder(nf=16, in_features=in_feat, out_features=latent_size).to(device)
    deformer = NeuralFlowDeformer(latent_size=args.lat_dims, f_width=args.deformer_nf, s_nlayers=3, 
                                  s_width=16, method='rk4', nonlinearity='leakyrelu', arch='imnet')
    all_model_params = list(deformer.parameters())+list(encoder.parameters())

    optimizer = OPTIMIZERS[args.optim](all_model_params, lr=args.lr)
    start_ep = 0
    global_step = np.zeros(1, dtype=np.uint32)
    tracked_stats = np.inf

    if args.resume:
        resume_dict = torch.load(args.resume)
        start_ep = resume_dict["epoch"]
        global_step = resume_dict["global_step"]
        tracked_stats = resume_dict["tracked_stats"]
        encoder.load_state_dict(resume_dict["encoder_state_dict"])
        deformer.load_state_dict(resume_dict["deformer_state_dict"])
        optimizer.load_state_dict(resume_dict["optim_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # more threads don't seem to help
    chamfer_dist = ChamferDistKDTree(reduction='mean', njobs=1)
    chamfer_dist = nn.DataParallel(chamfer_dist)
    chamfer_dist.to(device)
    encoder = nn.DataParallel(encoder)
    encoder.to(device)
    deformer = nn.DataParallel(deformer)
    deformer.to(device)

    model_param_count = lambda model: sum(x.numel() for x in model.parameters())
    logger.info(("{}(encoder) + {}(deformer) paramerters in total"
                 .format(model_param_count(encoder), model_param_count(deformer))))

    checkpoint_path = os.path.join(args.log_dir, "checkpoint_latest.pth.tar")

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # training loop
    for epoch in range(start_ep + 1, args.epochs + 1):
        loss_train = train_or_eval("train", args, encoder, deformer, chamfer_dist, train_loader, 
                                   epoch, global_step, device, logger, writer, optimizer, None)
        loss_eval = train_or_eval("eval", args, encoder, deformer, chamfer_dist, eval_loader, 
                                  epoch, global_step, device, logger, writer, optimizer, vis_loader)
        if args.lr_scheduler:
            scheduler.step(loss_eval)
        if loss_eval < tracked_stats:
            tracked_stats = loss_eval
            is_best = True
        else:
            is_best = False

        utils.save_checkpoint({
            "epoch": epoch,
            "encoder_state_dict": encoder.module.state_dict(),
            "deformer_state_dict": deformer.module.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "tracked_stats": tracked_stats,
            "global_step": global_step,
        }, is_best, epoch, checkpoint_path, "_meshflow", logger)

if __name__ == "__main__":
    main()