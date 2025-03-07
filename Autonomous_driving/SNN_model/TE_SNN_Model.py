from typing import Dict, List, Tuple, Type, Union
import torch
import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn
import math

from stable_baselines3.common.utils import get_device


lens=0.2
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

class TE_SFNN(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_size,
        neuron_type,
        recurrent,
        device: Union[th.device, str] = "auto",

    ) -> None:
        super().__init__()
        device = get_device(device)
        self.device=device

        self.T = 1000
        self.num_layers = 1
        self.input_size = feature_dim
        self.hidden_size = hidden_size
        self.recurrent = recurrent
        self.i2h = nn.Linear(feature_dim, hidden_size)
        self.total_spike = 0
        self.total_len = 0

        if self.recurrent:
            self.h2h = nn.Linear(hidden_size, hidden_size)
            nn.init.orthogonal_(self.h2h.weight, gain=1)
            nn.init.constant_(self.h2h.bias, 0)
            print('---recurrent---')
        self.neuron_type = neuron_type
        if self.neuron_type == 'TE-N':
            self.ce = nn.Parameter(torch.zeros(hidden_size, self.T, device=device))
            nn.init.normal_(self.ce, 0.01, 0.01)
        elif self.neuron_type == 'TE-R':
            dt = 0.1 / (1e2 ** (torch.arange(hidden_size).to(device) / hidden_size))
            self.dt = dt.to(device)
            self.fre = nn.Parameter((torch.rand(hidden_size, device=device) + 0.99) * math.pi * 2)
            self.time_idx = torch.arange(self.T).unsqueeze(0).to(device)
        elif self.neuron_type == 'LIF':
            self.ce = torch.zeros(hidden_size, self.T, device=device)
        self.step_num = 0
        self.thresh = 0.3
        if self.neuron_type == 'TE-R':
            self.beta = 0.1
        else:
            self.beta = 0.02
        self.decay = 0.5

        print('The neuron type to be used:',self.neuron_type)


    def forward(self, features: th.Tensor, init_states=None):
        if self.neuron_type == 'TE-R':
            fre_complex = (1j * (self.fre * self.dt)).unsqueeze(1)
            self.ce = torch.exp(fre_complex * self.time_idx).real * 0.1
        elif self.neuron_type == 'ALIF':
            self.ce = self.ro.unsqueeze(1).repeat(1, self.T)

        seq_len, bs, _ = features.size()
        out_seq = []

        if init_states is None:
            h1_spike, h1_mem = (torch.zeros(bs, self.hidden_size).to(self.device),
                                torch.zeros(bs, self.hidden_size).to(self.device))
            h1_thresh = torch.ones(bs, self.hidden_size).to(self.device) * self.thresh
            step_num = torch.tensor(0).to(self.device)
        else:
            h1_spike, h1_mem, h1_thresh, step_num = init_states
        for t in range(seq_len):
            # noise = torch.randn(features.shape).to(features.device) * 2 * features.mean()
            # features = features + noise
            x = features[t,:,:].unsqueeze(0)
            idx = (step_num).view(-1).long()
            if self.recurrent:
                h1_input = self.i2h(x) + self.h2h(h1_spike)
            else:
                h1_input = self.i2h(x)
            h1_spike, h1_mem, h1_thresh = self.mem_update_hidden(h1_input, h1_spike, h1_mem, h1_thresh, self.ce[:,idx].transpose(0, 1).unsqueeze(0))

            out_seq.append(h1_spike)
            step_num = step_num+1

        out_seq = torch.cat(out_seq, dim=0) #[T,N,H]

        self.total_len = self.total_len + seq_len
        self.total_spike = self.total_spike + h1_spike.sum()
        #print('frequency:', self.total_spike / ((self.total_len+1) * self.hidden_size))

        return out_seq, (h1_spike, h1_mem, h1_thresh, step_num)

    def mem_update_hidden(self, x, spike, mem, thresh_t, ce):
        if self.neuron_type == "LIF":
            thresh_t = thresh_t
        else:
            thresh_t = thresh_t + mem * ce - (thresh_t - self.thresh) * self.beta
        mem = mem * self.decay * (1. - spike) + x
        spike = act_fun_adp(mem, thresh_t)
        return spike, mem, thresh_t

