# Copyright (c) Microsoft. All rights reserved.
import torch
from torch.optim import Optimizer


class QHM(Optimizer):
    r"""
    Stochastic gradient method with Quasi-Hyperbolic Momentum (QHM):

        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k) 
        x(k+1) = x(k) - \alpha * d(k)   

    "Quasi-hyperbolic momentum and Adam for deep learning" 
        by Jerry Ma and Denis Yarats, ICLR 2019

    optimizer = QHM(params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0) 

    Args:
        params (iterable): iterable params to optimize or dict of param groups
        lr (float): learning rate, \alpha in QHM update (default:-1 need input)
        momentum (float, optional): \beta in QHM update, range[0,1) (default:0)
        qhm_nu (float, optional): \nu in QHM update, range[0,1] (default: 1)
            \nu = 0: SGD without momentum (\beta is ignored)
            \nu = 1: SGD with momentum \beta and dampened gradient (1-\beta)
            \nu = \beta: SGD with "Nesterov momentum" \beta
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:
        >>> optimizer = torch.optim.QHM(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=-1, momentum=0, nu=1, weight_decay=0):
        # nu can take values outside of the interval [0,1], but no guarantees!
        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay)
        super(QHM, self).__init__(params, defaults)

        # extra_buffer == True only in SSLS with momentum > 0 and nu != 1
        self.state['allocate_step_buffer'] = False


    def compute_qhm_direction(self):

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nu = group['nu']

            for p in group['params']:
                if p.grad is None:
                    continue
                x = p.data                      # Optimization parameters
                g = p.grad.data                 # Stochastic gradient
                
                # weight_decay is the same as adding L2 regularization
                if weight_decay > 0:
                    g.add_(weight_decay, x)
                
                # Compute the (negative) step directoin d and necessary momentum  
                state = self.state[p]
                if abs(momentum) < 1e-12 or abs(nu) < 1e-12:    # simply SGD if beta=0 or nu=0
                    d = state['step_buffer'] = g
                else: 
                    if 'momentum_buffer' not in state:
                        h = state['momentum_buffer'] = torch.zeros_like(x)
                    else:
                        h = state['momentum_buffer']
                    # Update momentum buffer: h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
                    h.mul_(momentum).add_(1 - momentum, g) 

                    if abs(nu - 1) < 1e-12:         # if nu=1, then same as SGD with momentum 
                        d = state['step_buffer'] = h
                    else:                           
                        if self.state['allocate_step_buffer']:  # copy from gradient
                            if 'step_buffer' not in state:
                                state['step_buffer'] = torch.zeros_like(g)
                            d = state['step_buffer'].copy_(g)
                        else:                                   # otherwise use gradient buffer
                            d = state['step_buffer'] = g
                        # Compute QHM momentum: d(k) = (1 - \nu) * g(k) + \nu * h(k)
                        d.mul_(1 - nu).add_(nu, h)

    def qhm_update(self):
        """ 
        Perform QHM update, need to call compute_qhm_direction() before calling this.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(-group['lr'], self.state[p]['step_buffer'])

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates model and returns loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.compute_qhm_direction()
        self.qhm_update()

        return loss
