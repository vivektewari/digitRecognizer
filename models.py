import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(6, 6),
                 stride=(1, 1), padding=(5, 5), pool_size=(2, 2)):
        super().__init__()
        self.pool_size = pool_size
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(out_channels / 3),
            kernel_size=tuple(np.array(kernel_size) + np.array([1, -1])),
            stride=stride,
            padding=tuple(np.array(padding) + np.array([1, 0])),
            bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(out_channels / 3),
            kernel_size=tuple(np.array(kernel_size) + np.array([-1, 1])),
            stride=stride,
            padding=tuple(np.array(padding) + np.array([0, 1])),
            bias=False)
        self.conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(out_channels / 3) + 1,
            kernel_size=tuple(np.array(kernel_size) + np.array([0, 0])),
            stride=stride,
            padding=tuple(np.array(padding) + np.array([0, 0])),
            bias=False)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=tuple(np.array(kernel_size) + np.array([0, 0])),
            stride=stride,
            padding=tuple(np.array(padding) + np.array([0, 0])),
            bias=False)

        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input1, pool_size=None, pool_type='max'):
        if pool_size is None: pool_size = self.pool_size
        x = input1
        self.in_channels = 13
        if self.in_channels == 1:
            x1 = F.relu_(self.conv1(input1))  # self.bn1(
            x2 = F.relu_(self.conv2(input1))
            x3 = F.relu_(self.conv3(input1))
        else:
            x = F.relu_(self.conv(input1))
        # x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            if self.in_channels == 1:
                # pool_size = tuple(np.array(self.pool_size) + np.array([1, -1]))
                x1 = F.max_pool2d(x1, kernel_size=pool_size)
                # pool_size = tuple(np.array(self.pool_size) + np.array([-1, 1]))

                x2 = F.max_pool2d(x2, kernel_size=pool_size)
                x3 = F.max_pool2d(x3, kernel_size=pool_size)
            else:
                x = F.max_pool2d(x, kernel_size=pool_size)
                return x

        elif pool_type == 'avg':
            if self.in_channels == 1:
                x1 = F.avg_pool2d(x1, kernel_size=pool_size)
                # pool_size = tuple(np.array(self.pool_size) + np.array([-1, 1]))

                x2 = F.avg_pool2d(x2, kernel_size=pool_size)
                x3 = F.avg_pool2d(x3, kernel_size=pool_size)
            else:
                x = F.avg_pool2d(x, kernel_size=pool_size)
            return x
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        elif pool_type == 'none':
            d = 0
        else:
            raise Exception('Incorrect argument!')
        return torch.cat([x1, x2, x3], dim=1)


class FeatureExtractor(nn.Module):
    def __init__(self, start_channel=4, input_image_dim=(28, 28), channels=[2],
                 convs=[4], strides=[1], pools=[2], pads=[1], fc1_p=[10, 10]):
        super().__init__()
        self.num_blocks = len(channels)
        self.start_channel = channels[0]
        self.conv_blocks = nn.ModuleList()
        self.input_image_dim = input_image_dim
        self.fc1_p = fc1_p
        self.mode_train = 0

        last_channel = start_channel
        last_layer_size = self.input_image_dim
        for i in range(self.num_blocks):
            self.conv_blocks.append(ConvBlock(in_channels=last_channel, out_channels=channels[i],
                                              kernel_size=(convs[i], convs[i]), stride=(strides[i], strides[i]),
                                              pool_size=(pools[i], pools[i]), padding=pads[i]))
            last_channel = channels[i]

        # getting dim of output of conv blo
        conv_dim = self.get_conv_output_dim()
        if self.fc1_p is not None:
            self.fc1 = nn.Linear(conv_dim, fc1_p[0], bias=True)
            self.fc2 = nn.Linear(fc1_p[0], fc1_p[1], bias=True)
        self.activation_l = torch.nn.ReLU()
        self.activation = torch.nn.Softmax(dim=1)
        self.init_weight()
        self.dropout = nn.Dropout(0.3)

    def get_conv_output_dim(self):
        input_ = torch.Tensor(np.zeros((1,1)+self.input_image_dim))
        x = self.cnn_feature_extractor(input_)
        return len(x.flatten())

    @staticmethod
    def init_layer(layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_weight(self):
        for i in range(self.num_blocks):
            self.init_layer(self.conv_blocks[i].conv)
            self.init_layer(self.conv_blocks[i].conv1)
            self.init_layer(self.conv_blocks[i].conv2)
            self.init_layer(self.conv_blocks[i].conv3)
        if self.fc1_p is not None:
            self.init_layer(self.fc1)
            self.init_layer(self.fc2)
        # init_layer(self.conv2)
        # init_bn(self.bn1)
        # init_bn(self.bn2)

    def cnn_feature_extractor(self, x):
        # input 501*64
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
            # x_70=torch.quantile(x, 0.7)
            # x_50 = torch.quantile(x, 1)
            # x=self.activation_l((x-x_50-1)/max(x_50,0.01))
            # x = self.activation_l(x/torch.max(x))

        return x

    def forward(self, input_):
        x = self.cnn_feature_extractor(input_ / 255)
        if self.fc1_p is None:
            x = x.flatten(start_dim=1, end_dim=-1)
            return self.activation(x)

        x = self.activation_l(x)

        x = self.activation_l(self.fc1(x.flatten(start_dim=1, end_dim=-1)))
        if self.mode_train == 1:
            x = self.dropout(x)
        x = x / torch.max(x)
        x = self.activation(self.fc2(x))

        return x

    def forward1(self, input_):
        x = self.cnn_feature_extractor(input_)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.activation(x / torch.max(x))

        return x


class FCLayered(FeatureExtractor):

    def __init__(self, num_blocks=1, start_channel=4, input_image_dim=(28, 28)):
        super().__init__()
        self.fc1 = nn.Linear(784, 30, bias=False)
        self.fc2 = nn.Linear(30, 2, bias=False)
        self.activation = torch.nn.Softmax()

    def forward(self, input_):
        x = input_

        x = self.fc1(x.flatten(start_dim=1, end_dim=-1))
        if self.mode == 'train': self.dropout = nn.Dropout(0.2)
        x = self.fc2(x)
        if self.mode == 'train': self.dropout = nn.Dropout(0.2)
        x = self.activation(x)
        return x
