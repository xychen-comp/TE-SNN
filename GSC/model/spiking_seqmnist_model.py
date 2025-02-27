import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .Hyperparameters import args

device = torch.device('cuda:'+str(args.cuda)  if torch.cuda.is_available() else "cpu")

algo = args.algo
thresh = args.thresh
lens = args.lens
decay = args.decay
cfg_fc = args.fc

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
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
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
    def __init__(self, in_size=1,output_size=10, permute=False, permute_matrix=None,output_spike=False,dropout=0.):
        super().__init__()
        self.co2 = args.beta
        self.decay = args.decay
        self.permute = permute
        self.permute_matrix = permute_matrix
        self.input_size = in_size
        self.output_size = output_size
        self.T = 101
        self.output_spike=output_spike
        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        if args.recurrent:
            self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])

        self.h2o_3 = nn.Linear(cfg_fc[1], self.output_size)

        if args.te == 'TE-R':
            dt = 0.1 / ((1e2) ** (torch.arange(cfg_fc[1]).float().to(device) / cfg_fc[1]))
            self.dt = dt.to(device)
            self.fre = nn.Parameter((torch.rand(cfg_fc[1], device=device)+0.99) * math.pi * 2)
            nn.init.normal_(self.fre, math.pi, math.pi)
        elif args.te == 'TE-N':
            self.ce = nn.Parameter(torch.zeros(cfg_fc[1], self.T, device=device))
            nn.init.normal_(self.ce, 0.01, 0.01)
        else:
            self.ce = torch.zeros(cfg_fc[1], self.T, device=device)
            
        self.time_idx = torch.arange(self.T).unsqueeze(0).to(device)
        self.drop0 = DropoutNd(dropout)
        self.drop1 = DropoutNd(dropout)
        self.drop2 = DropoutNd(dropout)

    def mem_update_hidden_wv(self, x, mem, spike, thresh_t, co1):

        thresh_t = thresh_t + mem * co1 - (thresh_t - thresh) * self.co2

        mem = mem * self.decay * (1. - spike) + x
        spike = act_fun_adp(mem, thresh_t)
        return mem, spike, thresh_t

    def forward(self, input):
        if args.te == 'TE-R':
            fre_complex = (1j * (self.fre * self.dt)).unsqueeze(1)
            self.ce = torch.exp(fre_complex * self.time_idx).real * 0.1
        
        N = input.size(1)
        h2h1_mem = h2h1_spike = torch.zeros(N, cfg_fc[0], device=device)

        h2h2_mem = h2h2_spike = torch.zeros(N, cfg_fc[1], device=device)

        output_sum = torch.zeros(N, self.output_size, device=device)


        thresh1 = torch.ones(N, cfg_fc[0], device=device) * thresh
        thresh2 = torch.ones(N, cfg_fc[1], device=device) * thresh

        input = input.squeeze()

        for step in range(self.T):
            input_x = input[step,:,:]
            
            if args.recurrent:
                h1_input = self.i2h_1(self.drop0(input_x.float())) + self.h2h_1(self.drop1(h2h1_spike))
            else:
                h1_input = self.i2h_1(self.drop0(input_x.float()))
            h2h1_mem, h2h1_spike, thresh1 = self.mem_update_hidden_wv(h1_input, h2h1_mem, h2h1_spike,
                                                                        thresh1, self.ce[:cfg_fc[0], step])

            h2_input = self.i2h_2(self.drop1(h2h1_spike))

            h2h2_mem, h2h2_spike, thresh2 = self.mem_update_hidden_wv(h2_input, h2h2_mem, h2h2_spike,
                                                                        thresh2, self.ce[:cfg_fc[1], step])

            h2o3_mem = self.h2o_3(self.drop2(h2h2_spike))

            output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.

        outputs = output_sum / self.T

        return outputs

