import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .Hyperparameters import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = args.thresh
lens = 0.2


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1. - self.p
            X = X * mask * (1.0 / (1 - self.p))
            return X
        return X

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,thresh_t):  # input = membrane potential- threshold
        ctx.save_for_backward(input,thresh_t)
        return input.gt(thresh_t).float()  # is firing ???
    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input,thresh_t = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input-thresh_t) < lens
        return grad_input * temp.float(), -grad_input * temp.float()

act_fun_adp = ActFun_adp.apply




class TESNN(nn.Module):
    def __init__(self, in_size=1, out_size=10, time_window=784, permute=False, permute_matrix=None, te_type=None,
                 hidden_size=[64, 64], beta=0.02,decay=0.5, output_spike=False, update_fun=None, dt_lim=[1e-1, 1e2]):
        super(TESNN, self).__init__()
        self.input_size = in_size
        self.output_size = out_size
        self.T = time_window
        self.hidden_size = hidden_size
        self.output_spike = output_spike
        self.permute = permute
        self.dt_lim = dt_lim
        self.i2h_1 = nn.Linear(self.input_size, self.hidden_size[0])
        self.i2h_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.i2h_3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])
        self.h2o_3 = nn.Linear(self.hidden_size[2], self.output_size)
        self.stride = 1
        self.permute_matrix = permute_matrix

        self.co2 = beta
        self.decay = decay

        self.te = te_type

        if args.recurrent:
            print('---Recurrent---')
            self.h2h_1 = nn.Linear(self.hidden_size[0], self.hidden_size[0])

        if self.te == 'TE-R':
            dt = self.dt_lim[0] / (self.dt_lim[1] ** (torch.arange(self.hidden_size[-1]).float().to(device) / self.hidden_size[-1]))
            self.dt = dt.to(device)
            self.fre = nn.Parameter((torch.rand(self.hidden_size[-1], device=device)+0.99) * math.pi * 2)
        elif self.te == 'TE-N':
            self.ce = nn.Parameter(torch.zeros(self.hidden_size[-1], self.T, device=device))
            nn.init.normal_(self.ce, 0.01, 0.01)
        else:
            self.ce = torch.zeros(self.hidden_size[-1], self.T, device=device)
        self.time_idx = torch.arange(self.T).unsqueeze(0).to(device)
        self.dropout0 = DropoutNd(args.dropout)
        self.dropout1 = DropoutNd(args.dropout)
        self.dropout2 = DropoutNd(args.dropout)

    def mem_update_hidden_wv(self, x, mem, spike, thresh_t, co1, a=None, b=None):
        if args.te != 'LIF':
            thresh_t = thresh_t + mem * co1 - (thresh_t - thresh) * self.co2
        mem = mem * self.decay * (1. - spike) + x
        spike = act_fun_adp(mem, thresh_t)
        return mem, spike, thresh_t

    def forward(self, input):
        if args.te == 'TE-R':
            fre_complex = (1j * (self.fre * self.dt)).unsqueeze(1)
            self.ce = torch.exp(fre_complex * self.time_idx).real * 0.1

        N = input.size(0)
        h2h1_mem = h2h1_spike = torch.zeros(N, self.hidden_size[0], device=device)

        h2h2_mem = h2h2_spike = torch.zeros(N, self.hidden_size[1], device=device)

        h2h3_mem = h2h3_spike = torch.zeros(N, self.hidden_size[2], device=device)
        output_sum = torch.zeros(N, self.output_size, device=device)

        h2h4_mem = h2h4_spike = torch.zeros(N, self.output_size, device=device)

        thresh1 = torch.ones(N, self.hidden_size[0], device=device) * thresh
        thresh2 = torch.ones(N, self.hidden_size[1], device=device) * thresh
        thresh3 = torch.ones(N, self.hidden_size[2], device=device) * thresh
        thresh4 = torch.ones(N, self.output_size,device=device) * thresh

        input = input.squeeze()
        input = input.view(N, -1)  # [N, 784]
        if self.permute:
            input = input[:, self.permute_matrix]

        for step in range(self.T):
            start_idx = step * self.stride
            if start_idx < (self.T - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            if args.recurrent:
                h1_input = self.i2h_1(self.dropout0(input_x.float())) + self.h2h_1(self.dropout1(h2h1_spike))
            else:
                h1_input = self.i2h_1(self.dropout0(input_x.float()))

            h2h1_mem, h2h1_spike, thresh1 = self.mem_update_hidden_wv(h1_input, h2h1_mem, h2h1_spike,
                                                                      thresh1, self.ce[:self.hidden_size[0], step])

            h2_input = self.i2h_2(self.dropout1(h2h1_spike))

            h2h2_mem, h2h2_spike, thresh2 = self.mem_update_hidden_wv(h2_input, h2h2_mem, h2h2_spike,
                                                                      thresh2, self.ce[:self.hidden_size[1], step])

            h3_input = self.i2h_3(h2h2_spike)
            h2h3_mem, h2h3_spike, thresh3 = self.mem_update_hidden_wv(h3_input, h2h3_mem, h2h3_spike,
                                                                      thresh3, self.ce[:self.hidden_size[-1], step])
            h2o3_mem = self.h2o_3(h2h3_spike)
            if self.output_spike:
                h2h4_mem, h2h4_spike, thresh4 = self.mem_update_hidden_wv(h2o3_mem, h2h4_mem, h2h4_spike,
                                                                          thresh4, self.ce[:10, step])
                output_sum = output_sum + h2h4_spike
            else:
                output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.



        outputs = output_sum / self.T

        return outputs

