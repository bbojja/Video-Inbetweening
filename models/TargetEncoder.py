import torch.nn as nn

from models.vqvae import VQVAE

class TargetEncoder(nn.Module):
    def __init__(self, vqvae=None, in_channel=3, embed_dim=64):
        super(TargetEncoder, self).__init__()

        self.vqvae = vqvae if vqvae is not None else VQVAE(in_channel=in_channel, embed_dim=embed_dim)
        # for params in self.vqvae.parameters():
            # params.requires_grad = False

    def forward(self, x):
        quant_t, quant_b, diff, _, _ = self.vqvae.encode(x)
        return quant_t, quant_b