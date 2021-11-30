import torch.nn as nn
import torch

class Base_Net(nn.Module):
    """
    Base Class of Neural Network in Multilevel Architecture
    l = 0
    """
    def __init__(self, dim_in, q):
        super().__init__()
        self.input_layer = nn.Linear(dim_in, dim_in*q, bias=False)
        self.norm_layer = nn.BatchNorm1d(dim_in*q)
        self.act_func = nn.ReLU()
        self.output_layer = nn.Linear(dim_in*q, 1)

        nn.init.uniform_(self.input_layer.weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))
        nn.init.uniform_(self.output_layer.weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))

    def forward(self, x_input):
        stack = self.input_layer(x_input)
        stack = self.norm_layer(stack)
        stack = self.act_func(stack)
        stack = self.output_layer(stack)
        return stack


class Deep_Net(nn.Module):
    """
    "middle" Class of Neural Network in Multilevel Architecture
    lvl = {1,...,Level-1}
    """
    def __init__(self, dim_in, q, lvl):
        super().__init__()
        self.hidden_layer_list = nn.ModuleList([nn.Linear(dim_in*q, dim_in*q, bias=False) for i in range((2**lvl)-1)])
        self.input_layer = nn.Linear(dim_in, dim_in*q, bias=False)
        self.output_layer = nn.Linear(dim_in*q, 1)
        self.act_func_list = nn.ModuleList([nn.ReLU() for _ in range(2**lvl)])
        self.norm_layer_list = nn.ModuleList([nn.BatchNorm1d(dim_in*q) for i in range(2**lvl)])

        nn.init.uniform_(self.input_layer.weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))
        for i in range(2 ** lvl - 1):
            nn.init.uniform_(self.hidden_layer_list[i].weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))
        nn.init.uniform_(self.output_layer.weight, a=-(dim_in ** (-1 / 2)), b=dim_in ** (-1 / 2))

    def forward(self, x_input, lvl):
        # input layer
        stack = self.input_layer(x_input)
        stack = self.norm_layer_list[0](stack)
        stack = self.act_func_list[0](stack)
        stack = self.hidden_layer_list[0](stack)

        # deep hidden layers
        for i in range(1, (2**lvl)-1):
            stack = self.norm_layer_list[i](stack)
            stack = self.act_func_list[i](stack)
            stack = self.hidden_layer_list[i](stack)

        # output layer
        stack = self.norm_layer_list[-1](stack)
        stack = self.act_func_list[-1](stack)
        stack = self.output_layer(stack)
        return stack


class Multilevel_Net(nn.Module):
    """
    Combined Multilevel Architecture
    """
    def __init__(self, dim_in, q, Level):
        super().__init__()
        self.base_level_net = Base_Net(dim_in, q)
        self.deep_level_nets = nn.ModuleList([Deep_Net(dim_in, q, lvl) for lvl in range(1, Level)])
        self.Level = Level
        self.input_normalization = nn.BatchNorm1d(dim_in)

    def forward(self, x_input):
        # first normalize the inputs -> then start with the network structure
        output = self.input_normalization(x_input)
        output = self.base_level_net(output)
        for lvl in range(1, self.Level):
            output = output + self.deep_level_nets[lvl-1](x_input, lvl)
        return output

    def predict(self, x_input):

        return self.forward(x_input)
