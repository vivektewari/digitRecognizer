import torch.nn.functional as F
import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,kernel_size=(6, 6),
                 stride=(1, 1),padding=(5, 5),pool_size=(2,2)):
        super().__init__()
        self.pool_size=pool_size
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # self.conv2 = nn.Conv2d(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     kernel_size=(3, 3),
        #     stride=(1, 1),
        #     padding=(1, 1),
        #     bias=False)

        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input1, pool_size=None, pool_type='max'):
        if pool_size is None: pool_size=self.pool_size
        x = input1
        x = F.relu_(self.conv1(x))  # self.bn1(
        # x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)

        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        elif pool_type == 'none':
            d = 0
        else:
            raise Exception('Incorrect argument!')
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, start_channel=4, input_image_dim=(28, 28), channels=[2],
                 convs=[4], strides=[1], pools=[2],pads=[1]):
        super().__init__()
        self.num_blocks = len(channels)
        self.start_channel = channels[0]
        self.conv_blocks = nn.ModuleList()
        self.input_image_dim = input_image_dim

        last_channel = start_channel
        last_layer_size = self.input_image_dim
        for i in range(self.num_blocks):
            self.conv_blocks.append(ConvBlock(in_channels=last_channel, out_channels=channels[i],
                                              kernel_size=(convs[i],convs[i]), stride=(strides[i], strides[i]),
                                              pool_size=(pools[i], pools[i]), padding=pads[i]))
            last_channel = channels[i]

            # last_layer_size = (last_layer_size[0]-convs[i][0]+strides[i][0]+pads[i][0])/strides[i][0]
        # self.fc1 = nn.Linear(last_layer_size[0] * last_layer_size[1] * last_channel, 10, bias=True)
        self.activation_l=torch.nn.Sigmoid()
        self.activation = torch.nn.Softmax()
        self.init_weight()

    @staticmethod
    def init_layer(layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_weight(self):
        for i in range(self.num_blocks): self.init_layer(self.conv_blocks[i].conv1)
        # self.init_layer(self.fc1)
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
            #x = self.activation_l(x/torch.max(x))

        return x



    def forward(self, input_):
        x = self.cnn_feature_extractor(input_)
        x = self.fc1(x.flatten(start_dim=1, end_dim=-1))
        x = self.activation(x)
        return x

    def forward(self, input_):
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
        x = self.fc2(x)
        x = self.activation(x)
        return x
