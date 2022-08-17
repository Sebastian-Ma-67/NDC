from pyexpat import features
import torch
import torch.nn as nn
from torch import distributions as dist

class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, out_bool, out_float, decoder, encoder=None, device=None):
        super().__init__()
        self.out_bool = out_bool
        self.out_float = out_float
        self.ef_dim = 128
        
        self.decoder = decoder.to(device)

        self.encoder = encoder.to(device)
        
        if self.out_bool:
            self.linear_bool = nn.Linear(self.ef_dim, 3)
        if self.out_float:
            self.linear_float = nn.Linear(self.ef_dim, 3)

        self._device = device

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############

        features = self.encode_inputs(inputs)
        out = self.decode(p, features, **kwargs)
        
        if self.out_bool:
            out_bool = self.linear_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.linear_float(out)
            return out_float

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        features = self.encoder(inputs)

        return features

    def decode(self, p, features, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            features (tensor): latent conditioned code
        '''

        logits = self.decoder(p, features, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
