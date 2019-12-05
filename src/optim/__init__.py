# Copyright (c) Microsoft. All rights reserved.
"""
:mod:`torch.optim` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from .qhm import QHM
from .salsa import SALSA
from .sasa_xd import SASA_xd
from .sasa_yaida import SASAYaida
from .sgd_sls import SGD_SLS
from .yaida_baseline import Yaida

