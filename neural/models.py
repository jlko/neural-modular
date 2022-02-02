import torch.nn as nn
import torchvision.models as models

from neural.custom_resnet.models import resnet18 as custom_resnet18


class MLP(nn.Module):
    def __init__(self, cfg, data_info):
        super().__init__()

        p = cfg.dropout_p

        nonlin = nn.LeakyReLU
        bn = nn.BatchNorm1d
        self.block = nn.Sequential(*[
            nn.Linear(data_info['D_in'], 200),
            bn(200),
            nonlin(),
            nn.Linear(200, 200),
            bn(200),
            nonlin(),
            nn.Linear(200, 100),
            bn(100),
            nonlin(),
            nn.Linear(100, 50),
            bn(50),
            nonlin(),
            nn.Dropout(p),
            nn.Linear(50, data_info['D_out']),
        ])

         # todo add bias
    def forward(self, x):
        return self.block(x)


class ResNetMLP(nn.Module):
    def __init__(self, cfg, data_info):
        super().__init__()

        p = cfg.dropout_p

        nonlin = nn.LeakyReLU
        bn = nn.BatchNorm1d

        self.block_1 = nn.Sequential(*[
            nn.Linear(data_info['D_in'], 200),
            # bn(200),
            nonlin(),
            nn.Linear(200, 200),
            nonlin(),
            nn.Linear(200, 200),
            nn.Dropout(p),
            ])

        self.block_2 = nn.Sequential(*[
            nn.Linear(200, 200),
            bn(200),
            nonlin(),
            nn.Linear(200, 50),
            nn.Dropout(p),
            ])

        self.block_3 = nn.Sequential(*[
            nn.Linear(50, 50),
            bn(50),
            nonlin(),
            nn.Linear(50, data_info['D_out']),
            ])

        self.blocks = [self.block_1, self.block_2, self.block_3]

        self.linear_1 = nn.Linear(data_info['D_in'], 200)
        self.linear_2 = nn.Linear(data_info['D_in'], 50)
        self.linears = [
            self.linear_1, self.linear_2, None]

        # todo add bias
    def forward(self, x):
        out = x
        for block, linear in zip(self.blocks, self.linears):
            if linear is not None:
                out = block(out) + linear(x)
            else:
                out = block(out)
        return out


class LSTM(nn.Module):
    def __init__(self, cfg, data_info):
        super().__init__()

        self.model = nn.LSTM(
            input_size=data_info['D_in'],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout_p,
            batch_first=True)

        self.linear = nn.Linear(
            cfg.num_layers * cfg.hidden_size, data_info['D_out'])

    def forward(self, x):
        output, (h_n, c_n) = self.model(x)

        # batch dim first
        h_n = h_n.transpose(0, 1)
        # flatten hidden states
        h_n = h_n.reshape(x.shape[0], -1)
        # project to out dim
        out = self.linear(h_n)

        return out


class ResNet18(nn.Module):
    """ResNet 18"""
    def __init__(self, cfg, data_info):
        super().__init__()
        self.model = models.regnet_y_400mf(num_classes=data_info['D_out'])

        # greyscale images
        if len(data_info['D_in']) == 2:
            fix_conv1(self.model)

    def forward(self, x):
        return self.model(x)


class Mobilenetv3(nn.Module):
    """ResNet 18"""
    def __init__(self, cfg, data_info):
        super().__init__()
        self.model = models.mobilenet_v3_small(num_classes=data_info['D_out'])

        # greyscale images
        if len(data_info['D_in']) == 2:
            raise

    def forward(self, x):
        return self.model(x)


class ResNet18Tweaked(nn.Module):
    """Tweaked ResNet18.

    See source for further info, has some CIFAR specific tweaks.

    Could probably be sped up? Is half the speed of built-in ResNet18 for some
    reason.
    """
    def __init__(self, cfg, data_info):
        super().__init__()
        self.model = custom_resnet18(num_classes=data_info['D_out'])

        # greyscale images
        if len(data_info['D_in']) == 2:
            fix_conv1(self.model)

    def forward(self, x):
        return self.model(x)


def fix_conv1(model):
    c1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=c1.out_channels,
        kernel_size=c1.kernel_size,
        stride=c1.stride,
        padding=c1.padding,
        bias=c1.bias)
