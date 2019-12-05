# Copyright (c) Microsoft. All rights reserved.
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import math
from scipy import stats


# keep the most recent half of the added items
class HalfQueue(object):
    def __init__(self, maxN, like_tens):
        self.q = torch.zeros(maxN, dtype=like_tens.dtype,
                             device=like_tens.device)
        self.n = 0
        self.remove = False
        self.maxN = maxN

    def double(self):
        newqueue = torch.zeros(self.maxN * 2, dtype=self.q.dtype,
                               device=self.q.device)
        newqueue[0:self.maxN][:] = self.q
        self.q = newqueue
        self.maxN *= 2

    def add(self, val):
        if self.remove is True:
            self.q[:-1] = self.q[1:]  # probably slow but ok for now
        else:
            self.n += 1
        self.q[self.n - 1] = val
        if self.n == self.maxN:
            self.double()
        self.remove = not self.remove  # or self.n == self.maxN)

    def mean_std(self, mode='bm'):
        gbar = torch.mean(self.q[:self.n])
        std_dict = {}
        df_dict = {}

        # sample variance for iid samples.
        std = torch.std(self.q[:self.n])
        std_dict['iid'] = std
        df_dict['iid'] = self.n - 1

        # batch mean variance
        b_n = int(math.floor(math.sqrt(self.n)))
        Yks = F.avg_pool1d(self.q[:self.n].unsqueeze(0).unsqueeze(0),
                           kernel_size=b_n, stride=b_n).view(-1)
        diffs = Yks - gbar
        std = math.sqrt(b_n / (len(Yks) - 1)) * torch.norm(diffs)
        std_dict['bm'] = std
        df_dict['bm'] = b_n - 1

        # overlapping batch mean
        Yks = F.avg_pool1d(self.q[:self.n].unsqueeze(0).unsqueeze(0),
                           kernel_size=b_n, stride=1).view(-1)
        diffs = Yks - gbar
        std = math.sqrt(
            b_n * self.n / (len(Yks) * (len(Yks) - 1))) * torch.norm(diffs)
        std_dict['olbm'] = std
        df_dict['olbm'] = self.n - b_n

        half_point = int(math.floor(self.n / 2))
        mean1 = torch.mean(self.q[:half_point])
        mean2 = torch.mean(self.q[half_point:self.n])
        halfmeans = [mean1.item(), mean2.item()]

        return gbar, std_dict[mode], df_dict[mode], std_dict, df_dict, halfmeans

    def reset(self):
        self.n = 0
        self.remove = False
        self.q.zero_()


# returns True if |u-v| < delta*u with signif level sigma.
def test_onesamp(d, v, sigma, delta, mode='bm', composite=False, verbose=True):
    dmean, dstd, ddf, stds, dfs, halfmeans = d.mean_std(mode=mode)
    v_mean, _, _, _, _, _ = v.mean_std()

    # tolerate delta percentage of error in v
    rhs = delta * v_mean

    K = d.n  # number of samples

    # confidence interval
    t_sigma_df = stats.t.ppf(1 - sigma / 2., ddf)
    half_width = dstd.mul(t_sigma_df / math.sqrt(K))
    dupper = dmean + half_width
    dlower = dmean - half_width
    # A more stable test is to also check if two half-means are of the same sign
    stationarity = (dupper < rhs and dlower > -rhs) and (halfmeans[0] * halfmeans[1] > 0)

    if composite:
        # Use two half tests to avoid false positive caused by crossing 0 in transient phase
        dlower0 = halfmeans[0] - half_width
        dupper0 = halfmeans[0] + half_width
        dlower1 = halfmeans[1] - half_width
        dupper1 = halfmeans[1] + half_width
        stationarity = (dupper0 < rhs and dlower0 > -rhs) and (dupper1 < rhs and dlower1 > -rhs) and (halfmeans[0] * halfmeans[1] > 0)

    if verbose:
        print("dmean: ", dmean)
        print("dstd: ", dstd)
        print("d upper bound: ", dupper)
        print("d lower bound: ", dlower)
        print("delta v_mean: ", rhs)

    return stationarity, dmean, dupper, dlower, rhs, stds, dfs


class SASAYaida(Optimizer):
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

    def step_onesamp(self, closure=None):
        loss = None
        # assert len(self.param_groups) == 1 # same as lbfgs
        # before gathering the gradient, add weight decay term
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            if weight_decay != 0:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.grad.data.add_(weight_decay, p.data)

        group = self.param_groups[0]
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
            state['K'] = 0  # how many samples we have
            state['d'] = HalfQueue(maxN, self._params[0])
            state['v'] = HalfQueue(maxN, self._params[0])
            state['composite'] = True

        # # before gathering the gradient, add weight decay term
        # if weight_decay != 0:
        #     for p in self._params:
        #         if p.grad is None:
        #             continue
        #         p.grad.data.add_(weight_decay, p.data)

        g_k = self._gather_flat_grad()
        x_k = self._gather_flat_param()
        d_kminus1 = self._gather_flat_buf('momentum_buffer')

        uk = g_k.dot(x_k)
        vk = d_kminus1.dot(d_kminus1).mul(
            0.5 * group['lr'] * (1.0 + momentum) / (1.0 - momentum))

        state['d'].add(uk - vk)
        state['v'].add(vk)

        if closure is not None:
            u = uk.item()
            v = vk.item()
            d = u - v
            closure([u], [v], [d], [], [], [], [], [])

        if state['K'] >= minN and state['K'] % group['testfreq'] == 0:
            u_equals_v, dmean, dupper, dlower, rhs, stds, dfs = test_onesamp(
                state['d'], state['v'], sigma, delta, mode=self.mode, composite=state['composite'])
            closure([], [], [], [dmean.item()], [dupper.item()], [dlower.item()], [rhs.item()],
                    stds, dfs)
            if state['step'] > self.warmup and u_equals_v:
                print("smaller lr: {}".format(group['lr'] * zeta))
                print("nsamp: {}".format(state['K']))

                group['lr'] = group['lr'] * zeta
                state['K'] = 0  # need to collect at least minN more samples.
                # should reset the queues here; bad if samples from before corrupt what you have now.
                state['d'].reset()
                state['v'].reset()
                state['composite'] = False
        elif self.logstats:
            if state['d'].n >= 4 and state['K'] % self.logstats == 0:
                u_equals_v, dmean, dupper, dlower, rhs, stds, dfs = test_onesamp(
                    state['d'], state['v'], sigma, delta, mode=self.mode, composite=state['composite'],
                    verbose=False)
                closure([], [], [], [dmean.item()], [dupper.item()], [dlower.item()],
                        [rhs.item()], stds, dfs)

        state['K'] += 1

        for p in self._params:
            if p.grad is None:
                continue
            param_state = self.state[p]
            g_k = p.grad.data
            # get momentum buffer.
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf.mul_(momentum).add_(1.0 - momentum, g_k)
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(1.0 - momentum, g_k)

            # now do update.
            p.data.add_(-group['lr'], buf)

        state['step'] += 1

    def __init__(self, params, lr=-1.0, weight_decay=0, momentum=0,
                 warmup=0, minN=100, maxN=1000, zeta=0.1, sigma=0.05, delta=0.1,
                 testfreq=1000, mode='bm', logstats=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, minN=minN, maxN=maxN,
                        zeta=zeta, momentum=momentum, sigma=sigma, delta=delta,
                        testfreq=testfreq)

        super(SASAYaida, self).__init__(params, defaults)
        self.step_fn = self.step_onesamp
        # self._params = self.param_groups[0]['params']
        self._params = []
        for param_group in self.param_groups:
            self._params += param_group['params']
        self.warmup = warmup  # todo: warmup in state?
        self.mode = mode  # using which variance estimator
        print("using variance estimator: ", mode)
        self.logstats = logstats
        print("logging stats every {} steps".format(logstats))

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
        return torch.cat(views, 0)

    def step(self, closure=None):
        self.step_fn(closure=closure)
