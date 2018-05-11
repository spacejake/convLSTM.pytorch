
import torch
import torch.nn as nn


class DenoiesCNN(nn.Module):
    def __init__(self, in_channel, hidden_channels):
        super(DenoiesCNN, self).__init__()
        #layers = []
        #self.layers = nn.ModuleList()
        self.layers = nn.Sequential()
        self.kernel_size =  3
        '''
        in_channel = 1 # Network has same dim for input and output
        hidden_channels = [16, 16, 16,
                           32, 32, 32,
                           64, 64,
                           128,
                           256,
                           128,
                           62,
                           32,
                           16,
                           1]
        '''

        for idx, hid_chnl in enumerate(hidden_channels):
            conv_lyr = nn.Sequential(              # input shape (in_channel, ?, ?)
                nn.Conv2d(
                    in_channels=in_channel,        # input height
                    out_channels=hid_chnl,         # n_filters
                    kernel_size=self.kernel_size,  # filter size
                    stride=1,                      # filter movement/step
                    padding=self.kernel_size//2    # for same width/height of image after con2d, padding=(kernel_size-1)/2 if stride=1
                ),                                 # output shape (hid_chnl, ?, ?)
                nn.ReLU(),                         # activation
            )
            self.layers.add_module("layer_{}".format(idx), conv_lyr)

            # setup input size for next layer
            in_channel = hid_chnl

        #self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, input):
        output = self.layers(input)

        #for layer in self.layers:
        #    x = layer(x)

            #x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        #output = self.out(x)
        return output, input    # return x for visualization



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization