import torch

import torch.optim as optim
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as transforms

import torchvision.utils

import numpy as np

from convlstm import ConvLSTM
from utils import *

b_size = 128

transform = transforms.Compose([transforms.ToTensor()])

raw_data = dset.MNIST("../autoencoderMNIST/MNIST", download=False, transform=transform)
dloader = torch.utils.data.DataLoader(raw_data, batch_size=b_size,
                                      shuffle=True, drop_last=True)


encoder = ConvLSTM(input_size=(28,28),
                   input_dim=1,
                   hidden_dim=[1],
                   kernel_size=(3,3),
                   num_layers=1,
                  )

decoder = ConvLSTM(input_size=(28,28),
                   input_dim=1,
                   hidden_dim=[1],
                   kernel_size=(3,3),
                   num_layers=1,
                  )

encoder.cuda()
decoder.cuda()

crit = nn.BCELoss()
crit.cuda()

threshold = nn.Threshold(0., 0.0)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=0.001)

s = 1

for e in range(100):
    for i, v in enumerate(dloader):
        optimizer.zero_grad()

        images =  Variable(v[0].cuda()).view(4, 32, 1, 28, 28)

        ########
        #Encoder
        ########

        hidden = encoder.get_init_states(32)
        _, encoder_state = encoder(images.clone(), hidden)
        ########
        #Decoder
        ########
        #prepare inputs for decoder
        rev_images = flip_var(images, 0)
        dec_input = torch.cat((Variable(torch.zeros(1,32,1,28,28).cuda()), rev_images[:-1,:,:,:,:]), 0)

        decoder_out, _ = decoder(dec_input, encoder_state)

        #######
        #loss##
        #######

        cut = threshold(decoder_out)
        loss = crit(cut, rev_images)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Epoch: {0} | Iter: {1} |".format(e, i))
            print("Loss: {0}".format(loss.data.cpu().numpy()[0]))
            print("===========================")


        if i % 500 == 0:
            samples = decoder_out.clone().view(128,1,28,28).data.cpu()[:64,:,:,:]
            samples = torch.cat((samples, rev_images.view(128,1,28,28).data.cpu()[:64,:,:,:]))

            torchvision.utils.save_image(samples,
                                "./out/{0},epoch{1},iter{2}.png".format(s,
                                                                         e,i),
                                         nrow=16)
            s += 1

