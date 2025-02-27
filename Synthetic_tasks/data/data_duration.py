import torch
import numpy as np
from torch.autograd import Variable


def data_generator(N, seq_length,max_duration):
    """
    Generate data for the duration discrimination task
    :param seq_length: The total sequence length
    :param max_duration: The maximal duration of each burst
    :param N: The batch size
    :return: Input and target data tensor
    """

    X = torch.zeros([N, 1, seq_length])
    Y = torch.zeros([N, 1])
    for i in range(N):
        duration1 = np.random.randint(5, max_duration)
        duration2 = np.random.randint(5, max_duration)
        position1 = np.random.randint(0, seq_length - duration1 - duration2 - 2)
        position2 = np.random.randint(position1 + duration1 + 1, seq_length - duration2 -1)

        X[i, 0, position1:position1+duration1] = 1
        X[i, 0, position2:position2+duration2] = 1

        Y[i, 0] = (duration1 - duration2)
    return Variable(X), Variable(Y)