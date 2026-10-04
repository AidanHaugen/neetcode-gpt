import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        z, a = None, None
        
        for i in range(len(weights)):
            w_i = weights[i]
            b_i = biases[i]

            if type(a) != type(None):
                z = np.dot(a, w_i)
            else:
                z = np.dot(x, w_i)

            z = z + b_i

            a = np.maximum(0, z)

        return np.round(z, 5)