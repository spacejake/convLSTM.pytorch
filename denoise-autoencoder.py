
import itertools

import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms

import torchvision.utils

from skimage import data, img_as_float, color
from skimage.util import random_noise

from convlstm import ConvLSTM
from utils import *

from transforms import RandomNoiseWithGT

batch_size = 32#4
seq_size=16

transform = transforms.Compose([
    RandomNoiseWithGT(),
    transforms.ToTensor()
])

raw_data = dset.MNIST("../autoencoderMNIST/MNIST", download=True, transform=transform)
dloader = torch.utils.data.DataLoader(raw_data, batch_size=batch_size,
                                      shuffle=True, drop_last=True)


encoder = ConvLSTM(input_size=(7,7),
                   input_dim=1,
                   hidden_dim=[1,2,4,8,16],
                   kernel_size=(3,3),
                   num_layers=5,
                   #batch_first=True,
                  )

decoder = ConvLSTM(input_size=(7,7),
                   input_dim=16,
                   hidden_dim=[16,8,4,2,1],
                   kernel_size=(3,3),
                   num_layers=5,
                   #batch_first=True,
                  )

encoder.cuda()
decoder.cuda()

crit = nn.MSELoss()
crit.cuda()

threshold = nn.Threshold(0., 0.0)
#params = list(encoder.parameters()) + list(decoder.parameters())
params = itertools.chain(encoder.parameters(), decoder.parameters())
optimizer = optim.Adam(params)#, weight_decay=1e-4)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

s = 1


#def corrupt(images):
#    for i in images:
#        print("image dim: ", i.size())

for e in range(100):
    exp_lr_scheduler.step()
    for i, v in enumerate(dloader):
        optimizer.zero_grad()
        # Split batch of 4 28x28x1 images into
        # a batch of 4 images each divided into 16 sequences (Blocks) of with 7x7x1 images
        #print("Size of v: ", v[0].size())


        corrupt_imgs = (v[0])[:,:,:,:28]
        gt_imgs = (v[0])[:,:,:,28:]

        #print(corrupt_imgs.size())
        #print(gt_imgs.size())

        images =  Variable(corrupt_imgs.cuda()).view(seq_size, batch_size, 1, 7, 7)
        gt =  Variable(gt_imgs.cuda()).view(seq_size, batch_size, 1, 7, 7)

        #corrupt(images)
        #exit()
        #print("Size of input: ", images.data.cpu().size())
        #torchvision.utils.save_image(images.clone().data.cpu()[:15, :, :, :], "./out/test.png", nrow=4)

        #exit()

        ########
        #Encoder
        ########
        encoder_out, encoder_state = encoder(images.clone())

        ########
        #Decoder
        ########
        #prepare inputs for decoder
        #rev_images = flip_var(images, 0)
        #dec_input = torch.cat((Variable(torch.zeros(1,32,1,28,28).cuda()), rev_images[:-1,:,:,:,:]), 0)

        decoder_out, _ = decoder(encoder_out)#, encoder_state)

        #######
        #loss##
        #######

        #loss = crit(decoder_out, gt)

        cut = threshold(decoder_out)
        loss = crit(cut, gt)


        #print("decoder_out: ", decoder_out.size())
        #output = decoder_out.contiguous().view(batch_size,  1, 28, 28)

        #print("Output: ", output.size())
        #print("GT: ", gt.size())

        #loss = crit(output, gt)

        loss.backward()
        optimizer.step()

        # Merge each image back into 1 28x28 image
        #final_out = decoder_out.clone().view(batch_size, 1, 1, 28, 28)

        if i % 100 == 0:
            print("Epoch: {0} | Iter: {1} | LR:{2}".format(e, i, exp_lr_scheduler.get_lr()[0]))
            print("Loss: {0}".format(loss.data.cpu().numpy()))#[0]))
            print("===========================")


        if i % 500 == 0:
            samples = images.clone().view(batch_size,  1, 28, 28).data.cpu()[:1,:,:,:]
            samples = torch.cat((samples, decoder_out.clone().view(batch_size,  1, 28, 28).data.cpu()[:1,:,:,:]))
            #samples = torch.cat((samples, output.data.cpu()[:1,:,:,:]))
            samples = torch.cat((samples, gt.clone().view(batch_size,  1, 28, 28).data.cpu()[:1,:,:,:]))

            torchvision.utils.save_image(samples,
                                "./out/{0},epoch{1},iter{2}.png".format(s,
                                                                         e,i),
                                         nrow=3)
            s += 1

