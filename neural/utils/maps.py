"""Map strings to classes."""
from neural.datasets import CIFAR10, FashionMNIST
from neural.models import (
    ResNet18, ResNet18Tweaked, Mobilenetv3, MLP, ResNetMLP, LSTM)

dataset = dict(
    CIFAR10=CIFAR10,
    FashionMNIST=FashionMNIST)

model = dict(
    ResNet18=ResNet18,
    ResNet18Tweaked=ResNet18Tweaked,
    Mobilenetv3=Mobilenetv3,
    MLP=MLP,
    ResNetMLP=ResNetMLP,
    LSTM=LSTM,
)

