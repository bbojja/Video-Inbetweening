import torch
import torch.nn as nn

from models.ConvLSTMCell import ConvLSTMCell
from models.vqvae import VQVAE

class StateEncoder(nn.Module):
    def __init__(self, nf, in_channel, embed_dim, vqvae=None):
        super(StateEncoder, self).__init__()

        self.embed_dim = embed_dim

        self.vqvae = vqvae if vqvae is not None else VQVAE(in_channel=in_channel, embed_dim=embed_dim)

        self.encoder_1_convlstm_t = ConvLSTMCell(input_dim=embed_dim,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm_t = ConvLSTMCell(input_dim=nf,
                                                hidden_dim=nf,
                                                kernel_size=(3, 3),
                                                bias=True)

        self.decoder_1_convlstm_t = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm_t = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN_t = nn.Conv3d(in_channels=nf,
                                     out_channels=embed_dim,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

        self.encoder_1_convlstm_b = ConvLSTMCell(input_dim=embed_dim,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm_b = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm_b = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm_b = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN_b = nn.Conv3d(in_channels=nf,
                                     out_channels=embed_dim,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, quant_t, quant_b, seq_len, future_step,
                    h_tt, c_tt, h_tt2, c_tt2, h_tt3, c_tt3, h_tt4, c_tt4,
                    h_tb, c_tb, h_tb2, c_tb2, h_tb3, c_tb3, h_tb4, c_tb4):

        outputs_t = []
        outputs_b = []

        # encoder
        for t in range(seq_len):
            h_tt, c_tt = self.encoder_1_convlstm_t(input_tensor=quant_t[:, t, :, :],
                                               cur_state=[h_tt, c_tt])  # we could concat to provide skip conn here
            h_tt2, c_tt2 = self.encoder_2_convlstm_t(input_tensor=h_tt,
                                                 cur_state=[h_tt2, c_tt2])  # we could concat to provide skip conn here
            h_tb, c_tb = self.encoder_1_convlstm_b(input_tensor=quant_b[:, t, :, :],
                                               cur_state=[h_tb, c_tb])  # we could concat to provide skip conn here
            h_tb2, c_tb2 = self.encoder_2_convlstm_b(input_tensor=h_tb,
                                                 cur_state=[h_tb2, c_tb2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector_t = h_tt2
        encoder_vector_b = h_tb2

        # decoder
        for t in range(future_step):
            h_tt3, c_tt3 = self.decoder_1_convlstm_t(input_tensor=encoder_vector_t,
                                                 cur_state=[h_tt3, c_tt3])  # we could concat to provide skip conn here
            h_tt4, c_tt4 = self.decoder_2_convlstm_t(input_tensor=h_tt3,
                                                 cur_state=[h_tt4, c_tt4])  # we could concat to provide skip conn here
            encoder_vector_t = h_tt4
            outputs_t += [h_tt4]  # predictions

            h_tb3, c_tb3 = self.decoder_1_convlstm_b(input_tensor=encoder_vector_b,
                                                 cur_state=[h_tb3, c_tb3])  # we could concat to provide skip conn here
            h_tb4, c_tb4 = self.decoder_2_convlstm_b(input_tensor=h_tb3,
                                                 cur_state=[h_tb4, c_tb4])  # we could concat to provide skip conn here
            encoder_vector_b = h_tb4
            outputs_b += [h_tb4]  # predictions

        outputs_t = torch.stack(outputs_t, 1)
        outputs_t = outputs_t.permute(0, 2, 1, 3, 4)
        outputs_t = self.decoder_CNN_t(outputs_t)
        outputs_t = torch.nn.Sigmoid()(outputs_t)
        outputs_b = torch.stack(outputs_b, 1)
        outputs_b = outputs_b.permute(0, 2, 1, 3, 4)
        outputs_b = self.decoder_CNN_b(outputs_b)
        outputs_b = torch.nn.Sigmoid()(outputs_b)

        return outputs_t, outputs_b

    def forward(self, x, future_seq=0, hidden_state=None):
        frames = x.reshape(
            shape=(
                x.shape[0] * x.shape[1],
                x.shape[2],
                x.shape[3],
                x.shape[4]
            )
        )            

        quant_t, quant_b, diff, _, _ = self.vqvae.encode(frames)

        quant_t = quant_t.reshape(
            x.shape[0], 
            x.shape[1],
            quant_t.shape[1],
            quant_t.shape[2],
            quant_t.shape[3]
        )

        quant_b = quant_b.reshape(
            x.shape[0], 
            x.shape[1],
            quant_b.shape[1],
            quant_b.shape[2],
            quant_b.shape[3]
        )

        # find size of different input dimensions
        b, seq_len, _, h_t, w_t = quant_t.size()
        b, seq_len, _, h_b, w_b = quant_b.size()

        # initialize hidden states
        h_tt, c_tt = self.encoder_1_convlstm_t.init_hidden(batch_size=b, image_size=(h_t, w_t))
        h_tt2, c_tt2 = self.encoder_2_convlstm_t.init_hidden(batch_size=b, image_size=(h_t, w_t))
        h_tt3, c_tt3 = self.decoder_1_convlstm_t.init_hidden(batch_size=b, image_size=(h_t, w_t))
        h_tt4, c_tt4 = self.decoder_2_convlstm_t.init_hidden(batch_size=b, image_size=(h_t, w_t))
        h_tb, c_tb = self.encoder_1_convlstm_b.init_hidden(batch_size=b, image_size=(h_b, w_b))
        h_tb2, c_tb2 = self.encoder_2_convlstm_b.init_hidden(batch_size=b, image_size=(h_b, w_b))
        h_tb3, c_tb3 = self.decoder_1_convlstm_b.init_hidden(batch_size=b, image_size=(h_b, w_b))
        h_tb4, c_tb4 = self.decoder_2_convlstm_b.init_hidden(batch_size=b, image_size=(h_b, w_b))

        # autoencoder forward
        outputs_t, outputs_b = self.autoencoder(quant_t, quant_b, seq_len, future_seq,
                                     h_tt, c_tt, h_tt2, c_tt2, h_tt3, c_tt3, h_tt4, c_tt4,
                                     h_tb, c_tb, h_tb2, c_tb2, h_tb3, c_tb3, h_tb4, c_tb4)

        outputs_t = outputs_t.permute(0, 2, 1, 3, 4)
        outputs_b = outputs_b.permute(0, 2, 1, 3, 4)

        return outputs_t, outputs_b