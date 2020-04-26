import torch
import torch.nn as nn
import torch.nn.functional as F

class Project(nn.Module):
    """
    A fixed project layer for distribution 
    """
    def __init__(self, reg_max=16):
        super(Project, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project).reshape(-1, 4)
        return x
