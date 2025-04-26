import torch
import numpy as np
from torch.autograd import Variable


def data_generator(N, seq_length):
    """
    Generate data for the interval discrimination task
    :param seq_length: The total sequence length
    :param N: The batch size
    :return: Input and target data tensor
    """
    X= torch.zeros([N, 1, seq_length])
    Y = torch.zeros([N, 1])

    for i in range(N):

        interval1 = np.random.randint(7, 10)
        interval2 = np.random.randint(3, 6)
        order = np.random.choice([0, 1], size=7, p=[0.5, 0.5])
        np.random.shuffle(order)

        choices = np.where(order == 1, interval1, interval2)  # Randomly sample interval1 or interval2 based on order
        index = np.cumsum(choices)  # Cumulative sum to generate the sequence
        index = np.concatenate(([0], index))
        start = np.random.randint(0, 20)
        index = index+start
        X[i, 0, index] = 1
        sum_ones = np.sum(order[order == 1])  # Sum of interval1 counts
        Y[i, 0] = sum_ones
    return Variable(X), Variable(Y)


