import numpy as np

def tanh(tensor):
    return 2 / (1 + np.exp(-1 * tensor)) - 1

def scale(pattern, m, b):
    return m*pattern + b

def unscale(pattern, m, b):
    return (pattern - b) / m

def scale_factors(patterns):
    max = np.max([np.max(p) for p in patterns])
    min = np.min([np.min(p) for p in patterns])

    m = 2 / (max - min)
    b = 1 - (m*max)
    return (m, b)
