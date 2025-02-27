import torch
import numpy as np
from torch.autograd import Variable


def data_generator(T, mem_length, b_size, encode=False):
    """
    Generate data for the delayed recall task
    :param T: The delay length
    :param mem_length: The length of the input sequences to be recalled
    :param b_size: The batch size
    :return: Input and target data tensor
    """
    seq = torch.from_numpy(np.random.randint(1, 9, size=(b_size, mem_length))).float()
    zeros = torch.zeros((b_size, T))
    marker = 9 * torch.ones((b_size, mem_length + 1))
    placeholders = torch.zeros((b_size, mem_length))

    x = torch.cat((seq, zeros[:, :-1], marker), 1)
    y = torch.cat((placeholders, zeros, seq), 1).long()
    if not encode:
        x_out, y = Variable(x), Variable(y)
    else:
        one_hot=torch.eye(10)[x.long(),]
        x_out,y =Variable(one_hot[:,:,1:].transpose(1,2)), Variable(y)
    return x_out, y