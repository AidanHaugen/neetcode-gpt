import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        std = (2 / (fan_in + fan_out)) ** (1/2)
        torch.manual_seed(0)
        return torch.round(torch.randn(fan_out, fan_in) * std, decimals = 4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        std = (2 / fan_in) ** (1/2)
        torch.manual_seed(0)
        return torch.round(torch.randn(fan_out, fan_in) * std, decimals = 4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        stds = []

        for i in range(num_layers):
            if (init_type == 'xavier'): 
                std = (2.0 / (dims[i] + dims[i + 1])) ** (1.0/2.0)
            elif (init_type == 'kaiming'):
                std = (2.0 / dims[i]) ** (1.0/2.0)
            else:
                std = 1.0

            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)
        

        relu = nn.ReLU()
        
        x = torch.randn(1, input_dim)
        for w in weights:
            linear = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
            with torch.no_grad():
                linear.weight.copy_(w)

            x = linear(x)
            x = relu(x)
            stds.append(round(x.std().item(), 2))

        return stds

