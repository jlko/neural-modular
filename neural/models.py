import torch.nn as nn


# maybe look up in that neurips paper how they
class MLP(nn.Module):
    def __init__(self, cfg, data_in, data_out):
        super().__init__()

        p = cfg.dropout_p

        nonlin = nn.LeakyReLU
        bn = nn.BatchNorm1d
        self.block = nn.Sequential(*[
            nn.Linear(data_in, 200),
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
            nn.Linear(50, data_out),
        ])

         # todo add bias
    def forward(self, x):
        return self.block(x)


class ResNetMLP(nn.Module):
    def __init__(self, cfg, data_in, data_out):
        super().__init__()

        p = cfg.dropout_p

        nonlin = nn.LeakyReLU
        bn = nn.BatchNorm1d

        self.block_1 = nn.Sequential(*[
            nn.Linear(data_in, 200),
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
            nn.Linear(50, data_out),
            ])

        self.blocks = [self.block_1, self.block_2, self.block_3]

        self.linear_1 = nn.Linear(data_in, 200)
        self.linear_2 = nn.Linear(data_in, 50)
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
    def __init__(self, cfg, data_in, data_out):
        super().__init__()

        self.model = nn.LSTM(
            input_size=data_in,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout_p,
            batch_first=True)

        self.linear = nn.Linear(cfg.num_layers * cfg.hidden_size, data_out)

    def forward(self, x):
        output, (h_n, c_n) = self.model(x)

        # batch dim first
        h_n = h_n.transpose(0, 1)
        # flatten hidden states
        h_n = h_n.reshape(x.shape[0], -1)
        # project to out dim
        out = self.linear(h_n)

        return out
