import torch
import torch.nn as nn

from models.ConvLSTMCell import ConvLSTMCell
from models.vqvae import VQVAE

from models.TargetEncoder import TargetEncoder
from models.StateEncoder import StateEncoder
from models.OffsetNetwork import OffsetNetwork

class VideoInbetweeningNetwork(nn.Module):
    def __init__(self, nf, in_channel, embed_dim, offset_vqvae=None, vqvae=None, offset_net=None):
        super(VideoInbetweeningNetwork, self).__init__()

        self.embed_dim = embed_dim

        self.vqvae = vqvae if vqvae is not None else VQVAE(in_channel=in_channel, embed_dim=embed_dim * 3)
        for params in self.vqvae.parameters():
            params.requires_grad = False

        self.state_encoder = StateEncoder(nf, in_channel, embed_dim)
        self.target_encoder = TargetEncoder(in_channel=in_channel, embed_dim=embed_dim)
        self.offset_encoder = OffsetNetwork(vqvae=offset_vqvae, in_channel=in_channel * 3, embed_dim=embed_dim) if offset_net is None else offset_net

        for params in self.offset_encoder.parameters():
            params.requires_grad = False

    # reshapes latent matrices to be fed into vqvaa
    def reshapeLatentMatrices(self, quant_t, quant_b):
        quant_t = quant_t.reshape(
            shape=(
                quant_t.shape[0] * quant_t.shape[1],
                quant_t.shape[2],
                quant_t.shape[3],
                quant_t.shape[4]
            )
        )

        quant_b = quant_b.reshape(
            shape=(
                quant_b.shape[0] * quant_b.shape[1],
                quant_b.shape[2],
                quant_b.shape[3],
                quant_b.shape[4]
            )
        )

        return quant_t, quant_b

    def forward(self, conditioning_frames, target_frame, num_frames_offset):
        # conditioning_frames: [b, seq_len, c, h, w]
        # target_frame: [b, c, h, w]

        state_frame = conditioning_frames[:, -1, :, :, :]

        state_encoding_quant_t, state_encoding_quant_b = self.state_encoder(conditioning_frames, future_seq=num_frames_offset - 1)
        state_encoding_quant_t, state_encoding_quant_b = self.reshapeLatentMatrices(state_encoding_quant_t, state_encoding_quant_b)

        target_encoding_quant_t, target_encoding_quant_b = self.target_encoder(target_frame)
        target_encoding_quant_t = target_encoding_quant_t.unsqueeze(1).expand(-1, num_frames_offset - 1, -1, -1, -1)
        target_encoding_quant_b = target_encoding_quant_b.unsqueeze(1).expand(-1, num_frames_offset - 1, -1, -1, -1)
        target_encoding_quant_t, target_encoding_quant_b = self.reshapeLatentMatrices(target_encoding_quant_t, target_encoding_quant_b)

        offset_encoding_quant_t, offset_encoding_quant_b = self.offset_encoder.offset_encode(state_frame, target_frame)
        offset_encoding_quant_t = offset_encoding_quant_t.unsqueeze(1).expand(-1, num_frames_offset - 1, -1, -1, -1)
        offset_encoding_quant_b = offset_encoding_quant_b.unsqueeze(1).expand(-1, num_frames_offset - 1, -1, -1, -1)
        offset_encoding_quant_t, offset_encoding_quant_b = self.reshapeLatentMatrices(offset_encoding_quant_t, offset_encoding_quant_b)

        quant_t = torch.cat((state_encoding_quant_t, target_encoding_quant_t, offset_encoding_quant_t), dim=1)
        quant_b = torch.cat((state_encoding_quant_b, target_encoding_quant_b, offset_encoding_quant_b), dim=1)

        output = self.vqvae.decode(quant_t, quant_b)
        output = output.reshape(
            shape=(
                conditioning_frames.shape[0],
                num_frames_offset - 1,
                output.shape[1],
                output.shape[2],
                output.shape[3]
            )
        )

        return output