

from typing import Optional, Callable, List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class SGDSEvo(Optimizer):

    def __init__(
        self, params,
        lr=1.e-1, momentum=0, dampening=0,
        weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None
    ):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDSEvo, self).__init__(params, defaults)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            smoother = group.get('smoother', None)

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            if smoother:
                smooth_update(
                    params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=group['weight_decay'],
                    momentum=group['momentum'],
                    lr=group['lr'],
                    dampening=group['dampening'],
                    nesterov=group['nesterov'],
                    maximize=group['maximize'],
                    smoother=smoother
                )
            else:
                regular_update(
                    params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=group['weight_decay'],
                    momentum=group['momentum'],
                    lr=group['lr'],
                    dampening=group['dampening'],
                    nesterov=group['nesterov'],
                    maximize=group['maximize'],
                )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def regular_update(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool
):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)

def smooth_update(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    smoother: Callable
):
    deltas = []
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        deltas.append(d_p)

    counts = [delta.size(0) for delta in deltas]
    deltas = smoother(torch.cat(deltas, dim=0))
    deltas = torch.split(deltas, counts)

    for i, param in enumerate(params):
        d_p = deltas[i]
        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)