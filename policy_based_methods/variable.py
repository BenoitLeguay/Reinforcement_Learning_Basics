import torch.nn as nn

layer_name_to_obj = {
    'linear': nn.Linear
}

activation_name_to_obj = {
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'softmax': nn.Softmax
}