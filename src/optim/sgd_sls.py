# Copyright (c) Microsoft. All rights reserved.
import math
from scipy import stats
import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from .qhm import QHM


class SGD_SLS(QHM):
    r"""
    SGD with Smoothed Line Search (SLS) for tuning learning rates

    Stochastic gradient with Quasi-Hyperbolic Momentum (QHM):
        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k) 
        x(k+1) = x(k) - \alpha(k) * d(k)   

    where \alpha(k) is smoothed version of \eta(k) obtained by line search 
    (line search performed loss defined by on current mini-batch)

        \alpha(k) = (1 - \gamma) * \alpha(k-1) + \gamma * \eta(k)
    
    Suggestion: set smoothing parameter by batch size:  \gamma = a * b / n
    The cumulative increase or decrease efficiency per epoch is (1-exp(-a)) 

    How to use it:
    >>> optimizer = SGD_SLS(model.parameters(), lr=1, momentum=0.9, qhm_nu=1,
    >>>                     weight_decay=1e-4, gamma=0.01)
    >>> for input, target in dataset:
    >>>     def eval_loss():
    >>>         output = model(input)
    >>>         loss = loss_fn(output, target)
    >>>         return loss
    >>>     optimizer.zero_grad()
    >>>     loss = eval_loss()
    >>>     loss.backward()
    >>>     optimizer.step(loss, eval_loss)
    """

    def __init__(self, params, lr=1e-3, momentum=0, nu=1, weight_decay=0, gamma=0.01, 
                 ls_evl=0, ls_sdc=0.1, ls_inc=2.0, ls_dec=0.5, ls_max=10, ls_ign=False):

        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>=0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1]: {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))
        if ls_sdc <= 0 or ls_sdc >= 0.5:
            raise ValueError("Invalid value for ls_sdc (0,0.5): {}".format(ls_sdc))
        if ls_inc < 1 :
            raise ValueError("Invalid value for ls_inc (>=1): {}".format(ls_inc))
        if ls_dec <= 0 or ls_dec >= 1:
            raise ValueError("Invalid value for ls_dec (0,1): {}".format(ls_dec))
        if ls_max < 1:
            raise ValueError("Invalid value for ls_max (>=1): {}".format(ls_dec))
        if gamma < 0 or gamma > 1:
            raise ValueError("Invalid value for gamma [0,1]: {}".format(gamma))

        super(SGD_SLS, self).__init__(params, lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay)
        #super().__init__(params, lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay)

        self.state['lr'] = float(lr)
        self.state['gamma'] = gamma
        self.state['eta'] = float(lr)
        self.state['ls_evl'] = ls_evl
        self.state['ls_sdc'] = ls_sdc
        self.state['ls_inc'] = ls_inc
        self.state['ls_dec'] = ls_dec
        self.state['ls_max'] = int(ls_max)
        self.state['ls_ign'] = ls_ign

    def step(self, loss, closure):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, required): A closure that reevaluates the model
                                          and returns the loss.
        """
        # loss already evaluated before calling step(), can pass as argument 
        # CALL IT AGAIN TO ALIGN WITH LINE SEARCH in eval() mode
        if self.state['ls_evl']:
            loss = closure()

        # Weight decay added here, so do not add again in QHM update through super().step()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                if weight_decay > 0:       
                    p.grad.data.add_(weight_decay, p.data)
        # Make copy of parameters into a buffer before doing line search
                state = self.state[p]
                if 'ls_buffer' not in state:
                    state['ls_buffer'] = torch.zeros_like(p.data)
                state['ls_buffer'].copy_(p.data)

        # line search on current mini-batch (not changing input to model)
        f0 = loss.item() + self.L2_regu_loss()
        g2 = self.grad_norm_sqrd()
        # try a large instantaneous step size at beginning of line search
        self.state['eta'] = self.state['ls_inc'] * self.state['lr']
        ls_count = 0
        while ls_count < self.state['ls_max']:
            # update parameters x := x - eta * g
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.data.copy_(self.state[p]['ls_buffer'])
                        p.data.add_(-self.state['eta'], p.grad.data)

            # evaluate loss of new parameters
            f1 = closure().item() + self.L2_regu_loss()
            # back-tracking line search
            if f1 > f0 - self.state['ls_sdc'] * self.state['eta'] * g2:
                self.state['eta'] *= self.state['ls_dec']
            # Goldstein line search: not effective in increasing learning rate
            # elif f1 < f0 - (1 - self.state['ls_sdc']) * self.state['eta'] * g2:
            #     self.state['eta'] *= self.state['ls_inc']
            else:
                break
            ls_count += 1
        else:
            if self.state['ls_ign']:
                self.state['eta'] = self.state['lr']

        # After line search over instantaneous step size, update learning rate by smoothing
        self.state['lr'] = (1 - self.state['gamma']) * self.state['lr'] + self.state['gamma'] * self.state['eta']
        # update lr in parameter groups AND reset weights to original value before line search
        for group in self.param_groups:
            group['lr'] = self.state['lr']
            for p in group['params']:
                if p.grad is not None:
                    p.data.copy_(self.state[p]['ls_buffer'])

        # Apply general QHM update (with self.state['weight_decay_added'] = True)
        self.state['weight_decay_added'] = True
        # super().step(None)
        QHM.step(self, closure=None)
        
        return None

    def L2_regu_loss(self):
        L2_loss = 0.0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                x = p.data.view(-1)
                L2_loss += 0.5 * weight_decay * (x.dot(x)).item()
        return L2_loss

    def grad_norm_sqrd(self):
        normsqrd = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data.view(-1)
                normsqrd += (g.dot(g)).item()
        return normsqrd
