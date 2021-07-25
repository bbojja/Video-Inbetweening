import torch
import torch.nn as nn

from models.vqvae import VQVAE

class OffsetNetwork(VQVAE):
    def __init__(self, vqvae=None, in_channel=6, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99):
        super(OffsetNetwork, self).__init__(
            in_channel=in_channel,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            embed_dim=embed_dim,
            n_embed=n_embed,
            decay=decay
        )

        # Fix pre-trained VQVAE
        self.vqvae = vqvae if vqvae is not None else VQVAE(in_channel=1, embed_dim=1)
        for params in self.vqvae.parameters():
            params.requires_grad = False

    def offset_encode(self, frames, next_frames):
        inputs = torch.cat((frames, next_frames), dim=1)
        offset_quant_t, offset_quant_b, diff, _, _ = self.encode(inputs)

        return offset_quant_t, offset_quant_b

    def forward(self, frames, next_frames, offset_only=False, offset_weight=1):
        inputs = torch.cat((frames, next_frames), dim=1)
        offset_quant_t, offset_quant_b, diff, _, _ = self.encode(inputs)

        quant_t = offset_weight * offset_quant_t
        quant_b = offset_weight * offset_quant_b
        if not offset_only:
            frame1_quant_t, frame1_quant_b, _, _, _ = self.vqvae.encode(frames)
            quant_t += frame1_quant_t
            quant_b += frame1_quant_b

        dec = self.vqvae.decode(quant_t, quant_b)
        return dec, diff