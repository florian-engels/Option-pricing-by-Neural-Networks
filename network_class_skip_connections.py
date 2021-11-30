import torch.nn as nn
import torch

class Deepest_Net(nn.Module):
    """
    Deepest lvl architecture of NN with depth of "lvl"
    """
    def __init__(self, dim_in, q, lvl):
        super().__init__()
        self.lvl = lvl
        self.input_layer = nn.Linear(dim_in, dim_in*q, bias=False)
        self.hidden_layer = nn.ModuleList([nn.Linear(dim_in*q, dim_in*q, bias=False) for _ in range(1, 2**lvl)])
        self.norm_layer = nn.ModuleList([nn.BatchNorm1d(dim_in*q, eps=1e-08) for _ in range(1, 2**lvl + 1)])
        self.act_func = nn.ReLU()
        self.output_layer = nn.Linear(dim_in*q, 1)

        nn.init.uniform_(self.input_layer.weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))
        for i in range(2 ** lvl - 1):
            nn.init.uniform_(self.hidden_layer[i].weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))
        nn.init.uniform_(self.output_layer.weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))

    def forward(self, x_input):
        stack = self.input_layer(x_input)
        skip_connection = torch.empty([int(2 ** self.lvl / 2 - 1), stack.size()[0], stack.size()[1]])
        for i in range(1, 2**self.lvl):
            stack = self.norm_layer[i-1](stack)
            stack = self.act_func(stack)
            stack = self.hidden_layer[i-1](stack)
            if i % 2 == 0:
                skip_connection[int(i/2 - 1)] = stack # Zuordnung der skip connections zu indizes (2->0, 4->1, 6->2, 8->3)
        stack = self.norm_layer[-1](stack)
        stack = self.act_func(stack)
        stack = self.output_layer(stack)
        return stack, skip_connection


class Middle_Net(nn.Module):
    """
    middle layer architecture with skip connections AND base shallow layer
    """
    def __init__(self, dim_in, q, lvl):
        super().__init__()
        self.input_layer = nn.Linear(dim_in, dim_in * q, bias=False)
        self.hidden_layer = nn.ModuleList(
            [nn.Linear(dim_in * q, dim_in * q, bias=False) for _ in range(1, 2 ** lvl)])
        self.norm_layer = nn.ModuleList([nn.BatchNorm1d(dim_in * q, eps=1e-08) for _ in range(1, 2 ** lvl + 1)])
        self.act_func = nn.ReLU()
        self.lvl = lvl
        self.output_layer = nn.Linear(dim_in*q, 1)

        nn.init.uniform_(self.input_layer.weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))
        for i in range(2 ** lvl - 1):
            nn.init.uniform_(self.hidden_layer[i].weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))
        nn.init.uniform_(self.output_layer.weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))

    def forward(self, x_input, skip_connection):
        stack = self.input_layer(x_input)
        skip_connection_new = torch.empty([int(2 ** self.lvl / 2 - 1), stack.size()[0], stack.size()[1]])
        for i in range(1, 2**self.lvl):
            stack = self.norm_layer[i-1](stack)
            stack = self.act_func(stack)
            stack = self.hidden_layer[i-1](stack)
            if (i % 2) == 0:
                skip_connection_new[int(i/2 - 1)] = stack
            stack = stack + skip_connection[i-1]
        stack = self.norm_layer[-1](stack)
        stack = self.act_func(stack)
        stack = self.output_layer(stack)
        return stack, skip_connection_new


class Multilevel_Net(nn.Module):
    """
    calling the different deep networks with additive skip connection
    """
    def __init__(self, dim_in, q, Level):
        super().__init__()
        self.Level = Level
        self.deepnet = Deepest_Net(dim_in=dim_in, q=q, lvl=Level-1)
        self.middle_nets = nn.ModuleList([Middle_Net(dim_in=dim_in, q=q, lvl=lvl) for lvl in range(Level-2, -1, -1)])

    def forward(self, x_input):
        output = torch.zeros([x_input.size()[0], 1])
        output_term, skip_connection = self.deepnet(x_input)
        output += output_term
        for i in range(self.Level-1):
            output_term, skip_connection = self.middle_nets[i](x_input, skip_connection)
            output += output_term
        return output

    def predict(self, x_input):

        return self.forward(x_input)
