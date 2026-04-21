import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):

    def __init__(self, inChannels=1, outChannels=1):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """
        super(MLP, self).__init__()

        # Initialize convolutional layers.
        self.lnr1 = nn.Linear(inChannels, 256)
        self.lnr2 = nn.Linear(256, 256)
        self.lnr3 = nn.Linear(256, 256)
        self.lnr4 = nn.Linear(256, 256)
        self.lnr5 = nn.Linear(256, outChannels)

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        x = F.leaky_relu(self.lnr1(x), negative_slope = 0.2)
        x = F.leaky_relu(self.lnr2(x), negative_slope = 0.2)
        x = F.leaky_relu(self.lnr3(x), negative_slope = 0.2)
        x = F.leaky_relu(self.lnr4(x), negative_slope = 0.2)
        x =F.tanh(self.lnr5(x))*10
        return x

class up(nn.Module):

    def __init__(self, inChannels, outChannels, kernel_size, padding, padding_mode, up_x, up_y):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """
        super(up, self).__init__()
        self.up_x = up_x
        self.up_y = up_y

        self.padding = padding
        self.padding_mode = padding_mode

        # Initialize convolutional layers.
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size, stride=1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.02)
        self.conv.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=(self.up_y,self.up_x), mode='bilinear', align_corners=True)
        # Padding
        x = F.pad(x, self.padding, mode=self.padding_mode)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv(x), negative_slope = 0.2)
        return x


class XfieldsFlow(nn.Module):
    """
    A class for creating Xfields-Flow like architecture as specified by the
    Xfields paper.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, h_res, w_res, min_=-1, max_=1, ngf=4, inChannels=1, outChannels=1,
                    device='cuda'):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels
            outChannels : int
                number of output channels.
        """


        super(XfieldsFlow, self).__init__()
        self.coordconv = torch.tensor([[[[min_, min_],
                                  [max_, min_]],
                                 [[min_, max_],
                                  [max_, max_]]]],dtype=torch.float32, device=device)

        padx,pady,up_x,up_y = self.upsampling_factor_padding(h_res,w_res)

        num_l = len(padx)
        layer_specs = [ngf*16, ngf*16 , ngf*16 , ngf*8 , ngf*8 , ngf*8 , ngf*4 ]
        layer_specs.extend([ngf*4]*(num_l-len(layer_specs)))

        # Initial upsampling + conv
        self.initial_padding = (0, padx[0],0,pady[0])
        # self.initial_padding = 'same'
        self.upconv1 = up(inChannels, layer_specs[0], 1, self.initial_padding, 'reflect', up_x[0], up_y[0])

        # Remaining upsampling + conv
        model = []
        input_size = layer_specs[0] + 2 # Adding 2 due to coordconv concat
        for num, num_filter in enumerate(layer_specs[1:]):
            padding = (1, 1 + padx[num+1], 1, 1 + pady[num+1])
            model += [up(input_size, num_filter, 3, padding, 'reflect', up_x[num+1], up_y[num+1])]
            input_size = num_filter
        self.model = nn.Sequential(*model)

        self.last_layer = nn.Conv2d(input_size, outChannels, 3, stride=1)
        nn.init.normal_(self.last_layer.weight, mean=0, std=0.02)
        self.last_layer.bias.data.fill_(0.0)

        # self.last_act = lambda x: x#nn.Tanh()
        self.last_act = nn.Tanh()
        self.last_padding = (1,1,1,1)
        self.last_padding_mode = 'reflect'


    def upsampling_factor_padding(self, h_res,w_res):
        res_temp = h_res
        py =[res_temp%2]
        while res_temp!=1:
            res_temp = res_temp//2
            py.append(res_temp%2)

        del py[-1]
        py = np.flip(py)

        res_temp = w_res
        px =[res_temp%2]
        while res_temp!=1:
            res_temp = res_temp//2
            px.append(res_temp%2)

        del px[-1]
        px = np.flip(px)

        lx = len(px)
        ly = len(py)
        up_x = 2*np.ones((lx))
        up_y = 2*np.ones((ly))

        if lx > ly:
            py = np.append(py,[0]*(lx-ly))
            up_y = np.append(up_y,[1]*(lx-ly))


        if ly > lx:
            px = np.append(px,[0]*(ly-lx))
            up_x = np.append(up_x,[1]*(ly-lx))

        return px,py,up_x,up_y

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                coordinates input.

        Returns
        -------
            tensor
                flow output.
        """

        # Initial up+conv
        x = self.upconv1(x)

        # Concatenate coordconv
        coordconv_tl = self.coordconv.tile((x.shape[0],1,1,1))
        coordconv_tl = F.pad(coordconv_tl, self.initial_padding, mode='reflect')
        x = torch.cat((x, coordconv_tl), dim=1)

        # Remaining up+conv layers
        x = self.model(x)

        # Padding + last layer
        x = F.pad(x, self.last_padding, mode=self.last_padding_mode)
        # Convolution + Leaky ReLU
        x = self.last_act(self.last_layer(x))

        return x
