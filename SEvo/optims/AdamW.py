

from typing import List, Callable

import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class AdamWSEvo(Optimizer):

    def __init__(
        self, params, 
        lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=False,
                        maximize=False, foreach=None, capturable=False,
                        )
        super(AdamWSEvo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            smoother = group.get('smoother', None)

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state_steps.append(state['step'])

            if smoother is not None:
                smooth_update(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    state_steps,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group['maximize'],
                    smoother=smoother
                )
            else:
                regular_update(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    state_steps,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group['maximize'],
                )
        return loss


def regular_update(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        param.addcdiv_(exp_avg, denom, value=-step_size)

def smooth_update(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    smoother: Callable
):
    deltas = []
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # modifications introduced in Theorem 3
        step = max(1, step_t.item() - 1)
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        zeros = grad.eq(0).all(dim=-1, keepdim=True).repeat((1, grad.size(1)))
        mgrad = torch.where(zeros, exp_avg.div(bias_correction1), grad)
        vgrad = torch.where(zeros, exp_avg_sq.div(bias_correction2), grad.pow(2))

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(mgrad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).add_(vgrad, alpha=1 - beta2)

        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        bias_correction2_sqrt = math.sqrt(bias_correction2)

        numer = (exp_avg.div(bias_correction1))
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        deltas.append(numer.div(denom))

    counts = [delta.size(0) for delta in deltas]
    deltas = smoother(torch.cat(deltas, dim=0))
    deltas = torch.split(deltas, counts)

    for i, param in enumerate(params):
        delta = deltas[i] * lr
        param.add_(delta.neg())