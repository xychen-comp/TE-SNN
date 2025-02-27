import torch
import torch.nn as nn
import math

"""
    Altered from https://github.com/byin-cwi/Efficient-spiking-networks
"""


b_j0 = 0.01  # neural threshold baseline
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
lens = 0.5

grads = []
def save_grad(name,step):
    def hook(grad):
        grads.append(grad)
    return hook

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        # temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        # temp =  gaussian(input, mu=0., sigma=lens)
        return grad_input * temp.float() * gamma


act_fun_adp = ActFun_adp.apply


def mem_update_adp(inputs, mem, spike, tau_adp, tau_m, b, dt=1, isAdapt=1):
    #     tau_adp = torch.FloatTensor([tau_adp])
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.
    b = ro * b + (1 - ro) * spike
    #B = b_j0 + beta * b
    B = 0.3 + beta * b

    #mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    mem = mem * alpha + R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b


def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem


class RNN_custom(nn.Module):
    def __init__(self, input_size=2, hidden_dims=[64, 64], output_size=1, time_window=1):
        super(RNN_custom, self).__init__()

        self.time_window = time_window
        self.input_size = input_size
        self.output_size = output_size

        self.relu = nn.ReLU()

        self.r1_dim = hidden_dims[0]
        self.r2_dim = hidden_dims[1]
        self.i2h = nn.Linear(input_size, self.r1_dim)
        self.h2h = nn.Linear(self.r1_dim, self.r1_dim)
        self.h2d = nn.Linear(self.r1_dim, self.r2_dim)
        self.d2o = nn.Linear(self.r2_dim, self.output_size)

        self.tau_adp_r1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_r2 = nn.Parameter(torch.Tensor(self.r2_dim))

        self.tau_m_r1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_m_r2 = nn.Parameter(torch.Tensor(self.r2_dim))

        nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2d.weight)
        nn.init.xavier_uniform_(self.d2o.weight)

        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.h2d.bias, 0)
        nn.init.constant_(self.d2o.bias, 0)

        nn.init.normal_(self.tau_adp_r1, 700, 25)
        nn.init.normal_(self.tau_adp_r2, 700, 25)

        nn.init.normal_(self.tau_m_r1, 20, 5)
        nn.init.normal_(self.tau_m_r2, 20, 5)

        self.b_r1 = self.b_r2 = self.b_o = self.b_d1 = 0

    def compute_input_steps(self, seq_num):
        return int(seq_num / self.stride)

    def forward(self, input, task='duration'):
        batch_size, seq_num, input_dim = input.shape
        self.b_r1 = self.b_r2 = self.b_o = self.b_d1 = b_j0

        r1_mem = r1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        r2_mem = r2_spike = torch.rand(batch_size, self.r2_dim).cuda()

        input = input
        input_steps = self.time_window

        out = []
        for i in range(input_steps):
            input_x = input[:, :, i]
            h_input = self.i2h(input_x.float()) + self.h2h(r1_spike)
            r1_mem, r1_spike, theta_r1, self.b_r1 = mem_update_adp(h_input, r1_mem, r1_spike, self.tau_adp_r1,
                                                                   self.tau_m_r1, self.b_r1)

            d_input = self.h2d(r1_spike)
            r2_mem, r2_spike, theta_r2, self.b_r2 = mem_update_adp(d_input, r2_mem, r2_spike, self.tau_adp_r2,
                                                                   self.tau_m_r2, self.b_r2)

            d2o_mem = self.d2o(r2_spike)
            out.append((d2o_mem))
        output = torch.stack(out, dim=2)
        if task == 'duration':
            return output[:,:,-1] # duration
        elif task == 'syn':
            return output # Synchronization
        elif task == 'interval':
            return output[:, :, -1] # add
        elif task == 'order':
            return output # Copy

    def get_grads(self, input):
        batch_size, seq_num, input_dim = input.shape
        self.b_r1 = self.b_r2 = self.b_o = self.b_d1 = b_j0

        r1_mem = r1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        r2_mem = r2_spike = torch.rand(batch_size, self.r2_dim).cuda()

        input = input
        input_steps = self.time_window
        out_ = []
        for i in range(input_steps):
            input_x = input[:, :, i]
            h_input = self.i2h(input_x.float()) + self.h2h(r1_spike)
            r1_mem, r1_spike, theta_r1, self.b_r1 = mem_update_adp(h_input, r1_mem, r1_spike, self.tau_adp_r1,
                                                                   self.tau_m_r1, self.b_r1)

            d_input = self.h2d(r1_spike)
            r2_mem, r2_spike, theta_r2, self.b_r2 = mem_update_adp(d_input, r2_mem, r2_spike, self.tau_adp_r2,
                                                                   self.tau_m_r2, self.b_r2)


            d2o_mem = self.d2o(r2_spike)
            r2_spike.register_hook(save_grad('grads', i))
            out_.append(d2o_mem)
        out = torch.stack(out_, dim=2)
        return out, grads