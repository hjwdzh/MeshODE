import torch
from torch import nn
# from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

import numpy as np

class DeformationFlowNetwork(nn.Module):
    def __init__(self, dim=3, latent_size=1, nlayers=4, width=50):
        """Intialize deformation flow network.
        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          nlayers: int, number of neural network layers. >= 2.
          width: int, number of neurons per hidden layer. >= 1.
        """
        super(DeformationFlowNetwork, self).__init__()
        self.dim = dim
        self.latent_size = latent_size
        self.nlayers = nlayers
        self.width = width
        
        nlin = nn.ReLU()
        modules = [nn.Linear(dim + latent_size, width), nlin]
        for i in range(nlayers-2):
            modules += [nn.Linear(width, width), nlin]
        modules += [nn.Linear(width, dim)]
        self.net = nn.Sequential(*modules)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, latent_vector, points):
        """
        Args:
          latent_vector: tensor of shape [batch, latent_size], latent code for each shape
          points: tensor of shape [batch, num_points, dim], points representing each shape
        Returns:
          velocities: tensor of shape [batch, num_points, dim], velocity at each point
        """
        latent_vector = latent_vector.unsqueeze(1).expand(-1, points.shape[1], -1)  # [batch, num_points, latent_size]
#         import pdb; pdb.set_trace()
        points_latents = torch.cat((points, latent_vector), axis=-1)  # [batch, num_points, dim + latent_size]
        b, n, d = points_latents.shape
        res = self.net(points_latents.reshape([-1, d]))
        res = res.reshape([b, n, self.dim])
        return res

    
class DeformationSignNetwork(nn.Module):
    def __init__(self, latent_size=1, nlayers=3, width=20):
        """Initialize deformation sign network.
        Args:
          latent_size: int, size of latent space. >= 1.
          nlayers: int, number of neural network layers. >= 2.
          width: int, number of neurons per hidden layer. >= 1.
        """
        super(DeformationSignNetwork, self).__init__()
        self.latent_size = latent_size
        self.nlayers = nlayers
        self.width = width
        
        nlin = nn.Tanh()
        modules = [nn.Linear(latent_size, width, bias=False), nlin]
        for i in range(nlayers-2):
            modules += [nn.Linear(width, width, bias=False), nlin]
        modules += [nn.Linear(width, 1, bias=False), nlin]
        self.net = nn.Sequential(*modules)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)

    def forward(self, dir_vector):
        """
        Args:
          dir_vector: tensor of shape [batch, latent_size], latent direction.
        Returns:
          signs: tensor of shape [batch, 1]
        """
        dir_vector = dir_vector / (torch.norm(dir_vector, dim=-1, keepdim=True) + 1e-6)  # normalize
        signs = self.net(dir_vector)
        return signs
    

class NeuralFlowModel(nn.Module):
    def __init__(self, dim=3, latent_size=1, f_nlayers=4, f_width=50, 
                 s_nlayers=3, s_width=20):
        super(NeuralFlowModel, self).__init__()
        self.flow_net = DeformationFlowNetwork(dim=dim, latent_size=latent_size, 
                                               nlayers=f_nlayers, width=f_width)
        self.sign_net = DeformationSignNetwork(latent_size=latent_size, 
                                               nlayers=s_nlayers, width=s_width)
        self.latent_source = None
        self.latent_target = None
        self.latent_updated = False
    
    def update_latents(self, latent_source, latent_target):
        self.latent_source = latent_source
        self.latent_target = latent_target
        self.latent_updated = True
    
    def forward(self, t, points):
        """
        Args:
          t: float, deformation parameter between 0 and 1.
          points: [batch, num_points, dim]
        """
        # reparametrize eval along latent path as a function of a single scalar t
        if not self.latent_updated:
            raise RuntimeError('Latent not updated. '
                               'Use .update_latents() to update the source and target latents.')
        flow = self.flow_net(self.latent_source + t * (self.latent_target - self.latent_source), points)
        sign = self.sign_net(self.latent_target - self.latent_source)
        return flow * sign
        
    
    
class NeuralFlowDeformer():
    def __init__(self, dim=3, latent_size=1, f_nlayers=4, f_width=50, 
                 s_nlayers=3, s_width=20, method='dopri5', device='cuda'):
        """Initialize. The parameters are the parameters for the Deformation Flow network.
        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          f_nlayers: int, number of neural network layers for flow network. >= 2.
          f_width: int, number of neurons per hidden layer for flow network. >= 1.
          s_nlayers: int, number of neural network layers for sign network. >= 2.
          s_width: int, number of neurons per hidden layer for sign network. >= 1.
          device: str, device to evaluate on.
        """
        super(NeuralFlowDeformer, self).__init__()
        self.device = device
        self.method = method
        self.timing = torch.from_numpy(np.array([0, 1]).astype('float32'))
        self.timing = self.timing.to(device)

        self.net = NeuralFlowModel(dim=dim, latent_size=latent_size, 
                                   f_nlayers=f_nlayers, f_width=f_width,
                                   s_nlayers=s_nlayers, s_width=s_width).to(device)

    @property
    def parameters(self):
        return self.net.parameters()

    def forward(self, latent_source, latent_target, points):
        """Forward transformation (source -> target).
        
        Args:
          latent_source: tensor of shape [batch, latent_size]
          latent_target: tensor of shape [batch, latent_size]
          points: [batch, num_points, dim]
        Returns:
          points_transformed: [batch, num_points, dim]
        """
        self.net.update_latents(latent_source, latent_target)
        points_transformed = odeint(self.net, points, self.timing, method=self.method)[1]
        return points_transformed

    def inverse(self, latent_source, latent_target, points):
        """Inverse transformation (target -> source).
        
        Args:
          latent_source: tensor of shape [batch, latent_size]
          latent_target: tensor of shape [batch, latent_size]
          points: [batch, num_points, dim] 
        Returns:
          points_transformed: [batch, num_points, dim]
        """
        return self.forward(latent_target, latent_source, points)
