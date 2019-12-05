# Copyright (c) Microsoft. All rights reserved.
import torch
from torch.optim import Optimizer
import math

# keep the most recent half of the added items
class HalfQueue(object):
    def __init__(self, maxN, like_tens):
        self.q = torch.zeros(maxN, dtype=like_tens.dtype,
                             device=like_tens.device)
        self.n = 0
        self.remove = False
        self.maxN = maxN


    def add(self, val):
        if self.remove is True:
            self.q[:-1] = self.q[1:] # probably slow but ok for now
        else:
            self.n += 1
        self.q[self.n - 1] = val
        self.remove = (not self.remove or self.n == self.maxN)

    def mean_std(self):
        b_n = int(math.floor(math.sqrt(self.n)))
        N = int(math.pow(b_n,2))
        q_view = self.q[:N].reshape(b_n, b_n)
        # compute the batch means Y_k
        Yks = torch.mean(q_view, dim=1)
        gbar = torch.mean(self.q[:self.n])
        diffs = Yks - gbar
        std = torch.sqrt(b_n / (b_n - 1) * diffs.dot(diffs))
        # also return sample variance just for comparison.
        return torch.mean(self.q[:self.n]), std, torch.std(self.q[:self.n])

    def reset(self):
        self.n = 0
        self.remove = False
        self.q.zero_()

class Yaida(Optimizer):
    r"""Implements SASA for estimating stationarity using a fixed-window test.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        warmup (int, optional): how long to wait (in step() calls) before changing lr (default: 0)
        N (int, optional): number of observations. choose for CLT. (default: 200)
        C (float, optional): factor by which to decrease the learning rate once stationarity is reached. (default: 0.1)

    Example:
        >>> optimizer = optim.SASAPflugBatch(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def step(self, closure=None):
        loss = None
        assert len(self.param_groups) == 1 # same as lbfgs

        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        minN = group['minN']
        maxN = group['maxN']
        zeta = group['zeta']
        momentum = group['momentum']
        delta = group['delta']
        sigma = group['sigma']

        # like in LBFGS, set global state to be state of first param.
        state = self.state[self._params[0]]

        # State initialization
        # accumulation queues belong to global state.
        if len(state) == 0:
            state['step'] = 0
            state['K'] = 0 # how many samples we have
            state['u'] = HalfQueue(maxN, self._params[0])
            state['v'] = HalfQueue(maxN, self._params[0])

        # before gathering the gradient, add weight decay term
        if weight_decay != 0:
            for p in self._params:
                if p.grad is None:
                    continue
                p.grad.data.add_(weight_decay, p.data)

        g_k = self._gather_flat_grad()
        x_k = self._gather_flat_param()
        d_kminus1 = self._gather_flat_buf('momentum_buffer')

        uk = g_k.dot(x_k)
        vk = d_kminus1.dot(d_kminus1).mul(0.5*group['lr'] * (1.0+momentum)/(1.0-momentum))

        state['u'].add(uk)
        state['v'].add(vk)

        if closure is not None:
            closure([uk.item()], [vk.item()])

        if state['K'] >= minN and state['K'] % group['testfreq'] == 0:
            # just use ratio test
            umean, _, _ = state['u'].mean_std()
            vmean, _, _ = state['v'].mean_std()
            print("umean: ", umean)
            print("vmean: ", vmean)

            drop = torch.abs(umean / vmean - 1) < delta

            if state['step'] > self.warmup and drop:
                print("smaller lr: {}".format(group['lr']*zeta))
                print("nsamp: {}".format(state['K']))

                group['lr'] = group['lr']*zeta
                state['K'] = 0 # need to collect at least minN more samples.
                # should reset the queues here; bad if samples from before corrupt what you have now.
                state['u'].reset()
                state['v'].reset()
        state['K'] += 1

        for p in self._params:
            if p.grad is None:
                continue
            param_state = self.state[p]
            g_k = p.grad.data
            # get momentum buffer.
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf.mul_(momentum).add_(1.0-momentum, g_k)
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(1.0-momentum, g_k)

            # now do update.
            p.data.add_(-group['lr'], buf)

        state['step'] += 1

    def __init__(self, params, lr=-1.0, weight_decay=0, momentum=0, warmup=0, minN=100, maxN=1000, zeta=0.1, sigma=0.05, delta=0.1, testfreq=1000):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, minN=minN, maxN=maxN,
                        zeta=zeta, momentum=momentum, sigma=sigma, delta=delta, testfreq=testfreq)

        super(Yaida, self).__init__(params, defaults)
        self._params = self.param_groups[0]['params']
        self.warmup = warmup # todo: warmup in state?

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)


    # method for gathering/initializing flat buffers that are the same
    # shape as the parameters
    def _gather_flat_buf(self, buf_name):
        views = []
        for p in self._params:
            param_state = self.state[p]
            if buf_name not in param_state:  # init buffer
                view = p.data.new(p.data.numel()).zero_()
            else:
                view = param_state[buf_name].data.view(-1)
            views.append(view)
        return torch.cat(views, 0)


    def _gather_flat_param(self):
        views = []
        for p in self._params:
            view = p.data.view(-1)
            views.append(view)
        return torch.cat(views,0)


    def _set_buf(self, state, buf_name, tensor):
        if buf_name not in state:
            state[buf_name] = tensor.data.clone()
        else:
            state[buf_name][:] = tensor.data
