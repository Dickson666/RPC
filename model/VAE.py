import torch
import torch.nn as nn
from model.common import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, input_dim, hidden, latent) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden, latent)
        self.decoder = Decoder(input_dim, hidden, latent)

    def sample_z(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sample_z(mean, logvar)
        res = self.decoder(z)
        return res, mean, logvar