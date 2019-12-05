# Copyright (c) Microsoft. All rights reserved.
import math
from scipy import stats
import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from .qhm import QHM


# Use a Leaky Bucket to store a fraction of most recent statistics 
# Wikepedia article: https://en.wikipedia.org/wiki/Leaky_bucket
class LeakyBucket(object):
    def __init__(self, size, ratio, dtype, device):
        '''
        size:  size of allocated memory buffer to keep the leaky bucket queue,
               which will be doubled whenever the memory is full
        ratio: integer ratio of total number of samples to numbers to be kept:
               1 - keep all, 2 - keep most recent 1/2, 3 - keep most recent 1/3 
        '''
        self.size = size
        self.ratio = int(ratio)
        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self.count = 0          # number of elements kept in queue (excluding leaked)
        self.start = 0          # count = end - start
        self.end = 0
        self.total_count = 0    # total number of elements added (including leaked)
 
    def reset(self):
        self.buffer.zero_()    
        self.count = 0          
        self.start = 0
        self.end = 0
        self.total_count = 0

    def double_size(self):
        newbuffer = torch.zeros(self.size * 2, dtype=self.buffer.dtype, device=self.buffer.device)
        newbuffer[0:self.size][:] = self.buffer
        self.buffer = newbuffer
        self.size *= 2

    def add(self, val):
        if self.end == self.size:               # when the end index reach size
            if self.start < self.ratio:             # if the start index is small
                self.double_size()                      # double the size of buffer
            else:                                   # otherwise shift the queue
                self.buffer[0:self.count] = self.buffer[self.start:self.end] 
                self.start = 0                          # reset start index to 0
                self.end = self.count                   # reset end index to count

        self.buffer[self.end] = val             # always put new value at the end
        self.end += 1                           # and increase end index by one
        if self.total_count % self.ratio == 0:  # if leaky_count is multiple of ratio
            self.count += 1                         # increase count in queue by one
        else:                                   # otherwise leak and keep same count
            self.start += 1                         # increase start index by one
        self.total_count += 1                   # always increase total_count by one

    def mean_std(self, mode='bm'):
        mean = torch.mean(self.buffer[self.start:self.end]).item()
        std_dict = {}
        dof_dict = {}

        # sample variance for iid samples.
        std_iid = torch.std(self.buffer[self.start:self.end])
        std_dict['iid'] = std_iid.item()
        dof_dict['iid'] = self.count - 1

        # batch mean variance
        b_n = int(math.floor(math.sqrt(self.count)))
        Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=b_n).view(-1)
        diffs = Yks - mean
        std_bm = math.sqrt(b_n /(len(Yks)-1))*torch.norm(diffs)
        std_dict['bm'] = std_bm.item()
        dof_dict['bm'] = b_n - 1

        # overlapping batch mean
        Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=1).view(-1)
        diffs = Yks - mean
        std_olbm = math.sqrt(b_n*self.count/(len(Yks)*(len(Yks)-1)))*torch.norm(diffs)
        std_dict['olbm'] = std_olbm.item()
        dof_dict['olbm'] = self.count - b_n

        half_point = self.start + int(math.floor(self.count / 2))
        mean1 = torch.mean(self.buffer[self.start : half_point])
        mean2 = torch.mean(self.buffer[half_point : self.end])
        halfmeans = [mean1.item(), mean2.item()]

        return mean, std_dict[mode], dof_dict[mode], std_dict, dof_dict, halfmeans


def stats_test(bucket, sigma, mode='bm', composite=False, verbose=False):
    mean, std, df, stds, dfs, halfmeans = bucket.mean_std(mode=mode)
    K = bucket.count    # number of samples kept in the leaky bucket

    # confidence interval
    t_sigma_df = stats.t.ppf(1-sigma/2., df)
    half_width = std * t_sigma_df / math.sqrt(K)
    lower = mean - half_width
    upper = mean + half_width
    # The simple confidence interval test    
    # stationarity = lower < 0 and upper > 0
    # A more stable test is to also check if two half-means are of the same sign
    stationarity = (lower < 0 and upper > 0) and (halfmeans[0] * halfmeans[1] > 0)

    if composite:
        # Use two half tests to avoid false positive caused by crossing 0 in transient phase
        lb0 = halfmeans[0] - half_width
        ub0 = halfmeans[0] + half_width
        lb1 = halfmeans[1] - half_width
        ub1 = halfmeans[1] + half_width
        stationarity = (lb0 < 0 and ub0 > 0) and (lb1 < 0 and ub1 > 0) and (halfmeans[0] * halfmeans[1] > 0)

    if verbose:
        print("mean: ", mean)
        print("std: ", std)
        print("upper bound: ", upper)
        print("lower bound: ", lower)
        print("conf. width: ", upper - lower)

    return stationarity, mean, upper, lower, stds, dfs    


class SASA_xd(QHM):
    r"""
    Statistical Adaptive Stochastic Approximation (SASA) with master condition.

    optimizer = SASA(params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0, 
                     drop_factor=2, sigma=0.02, var_mode='bm', leaky_ratio=4, 
                     minN=400, warm_up=0, test_freq=100, logstats=0)

    Stochastic gradient with Quasi-Hyperbolic Momentum (QHM):

        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k) 
        x(k+1) = x(k) - \alpha * d(k)   

    Stationary criterion: 
        E[ <x(k),   d(k)>] - (\alpha / 2) * ||d(k)||^2 ] = 0
    or equivalently,
        E[ <x(k+1), d(k)>] + (\alpha / 2) * ||d(k)||^2 ] = 0

    Args:
        params (iterable): iterable params to optimize or dict of param groups
        lr (float): learning rate, \alpha in QHM update (default:-1 need input)
        momentum (float, optional): \beta in QHM update, range(0,1) (default:0)
        qhm_nu (float, optional): \nu in QHM update, range(0,1) (default: 1)
            \nu = 0: SGD without momentum (\beta is ignored)
            \nu = 1: SGD with momentum and dampened gradient
            \nu = \beta: SGD with "Nesterov momentum"
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dropfactor (float, optional): factor of drop learning rate (default: 2)
        sigma (float, optional): (1 - sigma) confidence interval (default:0.02)  
        var_mode (string, optional): variance computing mode (default: 'mb')
        leaky_ratio (int, optional): leaky bucket ratio to kept (default: 4)
        minN (int, optional): min number of samples for testing (default: 400)
        warm_up (int, optional): number of steps before testing (default: 0)
        testfreq (int, optional): number of steps between testing (default:100)
        logstats (int, optional): number of steps between logs (0 means no log)

    Example:
        >>> optimizer = torch.optim.SASA(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=-1, momentum=0, nu=1, weight_decay=0, 
                 drop_factor=2, sigma=0.02, var_mode='mb', leaky_ratio=4, 
                 minN=1000, warmup=0, testfreq=100, logstats=0):

        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))
        if sigma <= 0 or sigma >= 1:
            raise ValueError("Invalid value for sigma (0,1): {}".format(sigma))
        if leaky_ratio < 1:
            raise ValueError("Invalid value for leaky_ratio (int, >1): {}".format(leaky_ratio))
        if warmup < 0:
            raise ValueError("Invalid value for warmup (int, >1): {}".format(warmup))
        if drop_factor < 1:
            raise ValueError("Invalid value for drop_factor (>=1): {}".format(leaky_ratio))

        super(SASA_xd, self).__init__(params, lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay)
        # New Python3 way to call super()
        # super().__init__(params, lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay)

        # State initialization: leaky bucket belongs to global state.
        p = self.param_groups[0]['params'][0]
        if 'bucket' not in self.state:
            self.state['bucket'] = LeakyBucket(1000, leaky_ratio, p.dtype, p.device)

        self.state['lr'] = float(lr)
        self.state['drop_factor'] = drop_factor
        self.state['sigma'] = sigma
        self.state['var_mode'] = var_mode
        self.state['min_stats'] = int(minN)
        self.state['warmup'] = int(warmup)
        self.state['testfreq'] = int(testfreq)
        self.state['logstats'] = int(logstats)
        self.state['composite'] = True           # first drop use composite statistical test
        # initializw number of steps
        self.state['nSteps'] = 0

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        """
        # Apply general QHM update
        self.state['weight_decay_added'] = False
        # super().step(closure=None)
        QHM.step(self, closure=None)
        self.state['nSteps'] += 1

        # compute <x(k+1), d(k)> and ||d(k)||^2 for statistical test
        u = 0.0
        v = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                xk1 = p.data.view(-1)
                dk = self.state[p]['step_buffer'].data.view(-1)     # OK after super().step()
                u += xk1.dot(dk).item()
                v += dk.dot(dk).item()
        v *= 0.5 * self.state['lr']

        # compute <x(k+1), d(k)> and ||d(k)||^2 for statistical test
        # dk = self._gather_flat_buffer('step_buffer')
        # xk1 = self._gather_flat_param() 
        # u = xk1.dot(dk).item()
        # v = (0.5 * self.state['lr']) * (dk.dot(dk).item())

        # add statistic to leaky bucket
        bucket = self.state['bucket']
        bucket.add(u + v)

        if closure is not None:
            closure([u], [v], [u + v], [], [], [], [], [])

        # check statistics and adjust learning rate
        if bucket.count > self.state['min_stats'] and self.state['nSteps'] % self.state['testfreq'] == 0:
            stationary, mean, ub, lb, stds, dfs = stats_test(bucket, self.state['sigma'], self.state['var_mode'], 
                                                             composite=self.state['composite'], verbose=True)
            if closure is not None:
                closure([], [], [], [mean], [ub], [lb], [], stds, dfs)
            # perform statistical test for stationarity
            if self.state['nSteps'] > self.state['warmup'] and stationary:
                self.state['lr'] /= self.state['drop_factor']
                for group in self.param_groups:
                    for p in group['params']:
                        group['lr'] = self.state['lr']
                self._zero_buffers('momentum_buffer')
                self.state['composite'] = False
                bucket.reset()
        elif self.state['logstats'] and (closure is not None):
            if bucket.count > bucket.ratio and self.state['nSteps'] % self.state['logstats'] == 0:
                _, mean, ub, lb, stds, dfs = stats_test(bucket, self.state['sigma'], self.state['var_mode'],
                                                        composite=self.state['composite'], verbose=False)
                closure([], [], [], [mean], [ub], [lb], [], stds, dfs)
    
        return None     # return None for now

    # methods for gather flat parameters
    def _gather_flat_param(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                view = p.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    # method for gathering/initializing flat buffers that are the same shape as the parameters
    def _gather_flat_buffer(self, buf_name):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if buf_name not in state:  # init buffer
                    view = p.data.new(p.data.numel()).zero_()
                else:
                    view = state[buf_name].data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def _zero_buffers(self, buf_name):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if buf_name in state:
                    state[buf_name].zero_()
        return None