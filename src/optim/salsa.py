# Copyright (c) Microsoft. All rights reserved.
from .sgd_sls import SGD_SLS
from .sasa_xd import SASA_xd


class SALSA(SASA_xd, SGD_SLS):
    r"""
    SALSA: SASA + SSLS 
    SASA: Statistical Adaptive Stochastic Approximation
    SSLS: Smoothed Stochastic Line Search

    Stochastic gradient with Quasi-Hyperbolic Momentum (QHM):
        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k) 
        x(k+1) = x(k) - \alpha(k) * d(k)   

    where \alpha(k) is smoothed version of \eta(k) obtained by line search 
    (line search performed loss defined by on current mini-batch)

        \alpha(k) = (1 - \gamma) * \alpha(k-1) + \gamma * \eta(k)
    
    Suggestion: set smoothing parameter by batch size:  \gamma = a * b / n
    The cumulative increase or decrease efficiency per epoch is (1-exp(-a)) 

    How to use it: (same as SGD_SLS, except for a warmup parameter)
    >>> optimizer = SALSA(model.parameters(), lr=1, momentum=0.9, qhm_nu=1,
    >>>                   weight_decay=1e-4, gamma=0.01, warmup=1000)
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
                 ls_evl=0, ls_sdc=0.1, ls_inc=2.0, ls_dec=0.5, ls_max=10, ls_ign=False,
                 warmup=1000, drop_factor=2, sigma=0.02, var_mode='mb', 
                 leaky_ratio=4, minN=1000, testfreq=100, logstats=0):

        SASA_xd.__init__(self, params, lr, momentum, nu, weight_decay, 
                         drop_factor, sigma, var_mode, leaky_ratio, minN, 
                         warmup, testfreq, logstats)

        # trouble using both as parent class, can only call one __init__
        # SGD_SLS.__init__(self, params, lr, momentum, nu, weight_decay, 
        #                  gamma, ls_sdc, ls_inc, ls_dec, ls_max, ls_ign)
        # Initialize states of SGD_SLS here
        self.state['lr'] = float(lr)
        self.state['gamma'] = gamma
        self.state['eta'] = float(lr)
        self.state['ls_sdc'] = ls_sdc
        self.state['ls_inc'] = ls_inc
        self.state['ls_dec'] = ls_dec
        self.state['ls_max'] = int(ls_max)
        self.state['ls_ign'] = ls_ign

    def step(self, loss, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, required): A closure that reevaluates the model
                                          and returns the loss.
        """
        if self.state['nSteps'] < self.state['warmup']:
            SGD_SLS.step(self, loss, closure)
            self.state['nSteps'] += 1
        else: 
            SASA_xd.step(self, closure=None)

        return None
