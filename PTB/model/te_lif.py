import torch
import torch.nn as nn
from torch.autograd import Function

import inspect
from torch.nn import Module

from .surrogate import SurrogateGradient as SG
class BaseNeuron(Module):
    def __init__(self, exec_mode: str="serial"):
        super(BaseNeuron, self).__init__()
        self.exec_mode    = exec_mode
        self._exec_config = {
            "default": self._serial_process,
            "serial" : self._serial_process,
            "fused"  : self._temporal_fused_process,
        }

    def forward(self, tx, v=None):
        execution_proc = self._exec_config.get(self.exec_mode)
        if execution_proc is not None:
            return execution_proc(tx, v)
        else:
            raise ValueError("Invalid `execution_mode`.")

    def _serial_process(self, _):
        raise NotImplementedError(f"The `{inspect.currentframe().f_code.co_name}` method of the subclass `{type(self).__name__}` needs to be implemented.")

    def _temporal_fused_process(self, _):
        raise NotImplementedError(f"The `{inspect.currentframe().f_code.co_name}` method of the subclass `{type(self).__name__}` needs to be implemented.")

class LIFAct_thresh(Function):
    @staticmethod
    def forward(ctx, v, rest, decay, threshold, time_step, surro_grad):
        ctx.save_for_backward(v, threshold)
        ctx.rest = rest
        ctx.decay = decay
        ctx.time_step = time_step
        ctx.surro_grad = surro_grad
        return v.gt(threshold).float()

    @staticmethod
    def backward(ctx, grad_y):
        (v,threshold) = ctx.saved_tensors
        grad_v = grad_y * ctx.surro_grad(
            v,
            rest=ctx.rest,
            decay=ctx.decay,
            threshold=threshold,
            time_step=ctx.time_step,
        )
        return grad_v, None, None, -grad_v, None, None



class TELIF(BaseNeuron):

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  # TODO: Set a default value
            exec_mode: str = "serial",
            recurrent: bool = False,
            beta: float = 0.02,
            te_type: str = 'N',
    ):
        super(TELIF, self).__init__(exec_mode=exec_mode)
        self.te_type = te_type
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        self.TE = None
        self.beta = beta
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)
        self.return_mem = False
        if self.te_type == 'N':
            print('Non-rhythmic TE is implemented')
        else:
            print('Rhythmic TE is implemented')
            self.dt = None
            self.fre = None
            self.time_idx = torch.arange(self.time_step).unsqueeze(0)


    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            thresh = state[2]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            thresh = torch.ones_like(tx[0]) * self.threshold
            return_state = False
        step = 0
        if self.te_type == 'R':
            fre_complex = (1j * (self.fre * self.dt.to(self.fre.device))).unsqueeze(1)
            self.TE = torch.exp(fre_complex * self.time_idx.to(self.fre.device)).real * 0.1
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            thresh = thresh + v * self.TE[:self.neuron_num,step] - (thresh - self.threshold) * self.beta
            v = v * self.decay * (1. - y) + x
            y = LIFAct_thresh.apply(v, self.rest, self.decay, thresh, self.time_step, self.surro_grad)
            ty.append(y)
            step = step + 1
        if return_state:
            return torch.stack(ty), (v, y, thresh)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)
