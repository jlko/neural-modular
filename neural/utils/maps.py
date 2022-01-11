"""Map strings to classes."""
from neural.datasets import RandomLGSSM
from neural.models import MLP, ResNetMLP, LSTM

dataset = dict(
    RandomLGSSM=RandomLGSSM,
    )

model = dict(
    MLP=MLP,
    ResNetMLP=ResNetMLP,
    LSTM=LSTM,
)
