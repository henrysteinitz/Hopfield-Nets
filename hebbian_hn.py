import numpy as np
from utils import tanh, scale, unscale, scale_factors, sign
from time import sleep

class HebbianHopfieldNet:

    def __init__(self, patterns):
        pattern_size = len(patterns[0])
        patterns = [np.array(p) for p in patterns]
        self.scale_factors = scale_factors(patterns)
        self.build_weights([scale(p, *self.scale_factors) for p in patterns])


    def build_weights(self, patterns):
        pattern_size = len(patterns[0])
        num_patterns = len(patterns)
        weights = np.zeros((pattern_size, pattern_size))
        for p in patterns:
            for i in range(len(p)):
                for j in range(i+1, len(p)):
                    weights[i][j] += (1 / num_patterns) * p[i] * p[j]
                    weights[j][i] = weights[i][j]
        self.weights = weights

    def play(self, input, steps=10000):
        memory = scale(np.array(input), *self.scale_factors)
        for i in range(steps):
            memory = sign(tanh((np.matmul(self.weights, memory))))
        return unscale(memory, *self.scale_factors)
