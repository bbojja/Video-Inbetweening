import os
import torch

from models.VideoInbetweeningNetwork import VideoInbetweeningNetwork, OffsetNetwork
from data.MovingMNIST import MovingMNIST

from tqdm import tqdm

import argparse

import os
import os.path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=24, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--use_amp', default=False, type=bool, help='mixed-precision training')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--n_hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')

opt = parser.parse_args()

def train():
    train_data = MovingMNIST(
        train=True,
        data_root=os.getcwd() + '/data',
        seq_len=20,
        image_size=64,
        deterministic=True,
        num_digits=2
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading pre-trained vq-vae weights into offset network
    offset_weights = torch.load("checkpoints/offset_encoder.pt")
    vqvae_weights = torch.load("checkpoints/vqvae.pt")
    for name, weights in vqvae_weights.items():
        offset_weights[name] = weights
    
    offset_net = OffsetNetwork(in_channel=2, embed_dim=1)
    offset_net.load_state_dict(offset_weights)
    model = VideoInbetweeningNetwork(nf=opt.n_hidden_dim, in_channel=1, embed_dim=1, offset_net=offset_net).to(device)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(opt.epochs):
        for batch in tqdm(train_loader):
            offset_frames = 6
            optimizer.zero_grad()

            # batch : [b, n, h, w, c]
            conditioning_frames = batch[:, :10, :, :, :]
            conditioning_frames = conditioning_frames.permute(0, 1, 4, 2, 3)
            conditioning_frames = conditioning_frames.to(device)

            target_frame = batch[:, 10+offset_frames-1, :, :, :]
            target_frame = target_frame.permute(0, 3, 1, 2)
            target_frame = target_frame.to(device)

            inbetween_frames = batch[:, 10:10+offset_frames-1, :, :, :]
            inbetween_frames = inbetween_frames.permute(0, 1, 4, 2, 3)
            inbetween_frames = inbetween_frames.to(device)

            predicted_inbetween_frames = model(conditioning_frames, target_frame, offset_frames)

            loss = criterion(predicted_inbetween_frames, inbetween_frames)
            loss.backward()
            optimizer.step()
        
        torch.save(model.state_dict(), "checkpoints/model.pth")


if __name__ == '__main__':
    train()


