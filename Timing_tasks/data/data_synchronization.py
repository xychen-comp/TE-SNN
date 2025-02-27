import torch
import numpy as np
from torch.autograd import Variable


def data_generator(N, seq_length, ISI = [3,9]):
    """
    Generate data for the synchronization task
    :param seq_length: The total sequence length
    :param ISI: The range of inter-stimulus intervals between stimulus
    :param N: The batch size
    :return: Input and target data tensor
    """

    X = torch.zeros([N, 1, seq_length])
    Y = torch.zeros([N, 1, seq_length])
    for i in range(N):
        cycle = (np.random.randint(ISI[0], ISI[1]))
        positions = np.random.randint(0,seq_length//4)
        indexes1 = np.arange(positions, positions+cycle*3+1, cycle)
        X[i, 0, indexes1] = 1
        indexes2 = np.arange(positions, seq_length, cycle)
        Y[i, 0, indexes2] = 1

    return Variable(X), Variable(Y)