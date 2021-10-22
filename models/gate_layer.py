import torch
import torch.nn as nn

class GateLayer(nn.Module):
    def __init__(self, ipFeats, opFeats, sizeMask):
        super(GateLayer, self).__init__()
        self.ipFeats = ipFeats
        self.opFeats = opFeats
        self.sizeMask = sizeMask
        self.weight = nn.Parameter(torch.ones(opFeats))

    def forward(self, input):
        return input * self.weight.view(*self.sizeMask)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.ipFeats, self.opFeats is not None)
