# Copyright (c) Microsoft. All rights reserved.
# Written by Pengchuan Zhang, penzhan@microsoft.com
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image size during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DATA = CN()
# choices=['toy_ill', 'toy_well', 'mnist', 'cifar', 'cifar100', 'imagenet', 'wikitext-2']
_C.DATA.DATASET = 'cifar'
# path to datasets, default=os.getenv('PT_DATA_DIR', './datasets')
_C.DATA.PATH = "./datasets"
# path to other necessary data like checkpoints other than datasets.
# default=os.getenv('PT_DATA_DIR', 'data')
_C.DATA.DATA_DIR = "./data"

# choices=['mse', 'xentropy', 'bce'], msr for least regression or xentropy for classification
_C.LOSS = 'xentropy'


# dataloader
_C.DATALOADER = CN()
# batch size
_C.DATALOADER.BSZ = 128
# samples are drawn with replacement if yes
_C.DATALOADER.RE = 'no'
# number of data loading workers
_C.DATALOADER.WORKERS = 0

# optimizer
_C.OPTIM = CN()
# optimizer, choices=['zero_bat', 'zero_seq', 'yaida_diag', 'yaida_seq',
#  'yaida_ratio', 'lxz', 'baydin',
#  'pflug_bat', 'pflug_seq', 'sgd', 'qhm', 'adam',
#  'statsnon', 'pflug_wilcox',
#  'sasa_xd_seq', 'sasa_xd', 'sgd_sls', 'salsa', 'ssls', 'salsa_new'], default='sgd')
_C.OPTIM.OPT = 'qhm'
# effective learning rate
_C.OPTIM.LR = 1.0
# effective momentum value
_C.OPTIM.MOM = 0.9
# nu value for qhm
_C.OPTIM.NU = 1.0
# weight decay lambda
_C.OPTIM.WD = 5e-4
# Number of Epochs
_C.OPTIM.EPOCHS = 150
# Warm up: epochs of qhm before switching to sasa/salsa
_C.OPTIM.WARMUP = 0
# Drop frequency and factor for all methods
_C.OPTIM.DROP_FREQ = 50
_C.OPTIM.DROP_FACTOR = 10.0
# use validation dataset to adapt learning rate
_C.OPTIM.VAL = 0

# ADAM's default parameters
_C.OPTIM.ADAM = CN()
_C.OPTIM.ADAM.BETA1 = 0.9
_C.OPTIM.ADAM.BETA2 = 0.999

# SASA's default parameters
_C.OPTIM.SASA = CN()
# leaky bucket ratio
_C.OPTIM.SASA.LEAKY_RATIO = 4
# significance level
_C.OPTIM.SASA.SIGMA = 0.01
# Minimal sample size for statistical test
_C.OPTIM.SASA.N = 1000
# delta in equivalence test
_C.OPTIM.SASA.DELTA = 0.02
# method to calculate variance, choices=['iid', 'bm', 'olbm']
_C.OPTIM.SASA.MODE = 'olbm'
# log frequency (iterations) for statistics, 0 means logging at statistical tests
_C.OPTIM.SASA.LOGSTATS = 0
# number of statistical tests in one epoch
_C.OPTIM.SASA.TESTS_PER_EPOCH = 1
_C.OPTIM.SASA.TESTFREQ = 5005

# Line search
_C.OPTIM.LS = CN()
# smoothing factor
_C.OPTIM.LS.GAMMA = 0.01
# Sufficient decreasing constant
_C.OPTIM.LS.SDC = 0.1
# Increase factor
_C.OPTIM.LS.INC = 2.0
# Decrease factor
_C.OPTIM.LS.DEC = 0.5
# Maximal backtracking steps
_C.OPTIM.LS.MAX = 10
# Ignore the backtracking that reaches _C.OPTIM.LS.MAX
_C.OPTIM.LS.IGN = 0
# function call in evaluation mode for line search
_C.OPTIM.LS.EVAL = 1



# models
_C.MODEL = CN()
# choices=model_names + my_model_names + seq_model_names,
#     help='model architecture: ' +
#          ' | '.join(model_names + my_model_names + seq_model_names) +
#          ' (default: resnet18)')
_C.MODEL.ARCH = 'resnet18'
# nonlinearity, choices=['celu', 'softplus']
_C.MODEL.NONLINEARITY = 'celu'
# relative path of checkpoint relative to DATA_DIR
_C.MODEL.MODEL_PATH = ""
# use pre-trained model from torchvision
_C.MODEL.PRETRAINED = False

_C.MODEL.RNN = CN()
# size of word embeddings
_C.MODEL.RNN.EMSIZE = 1500
# number of hidden units per layer
_C.MODEL.RNN.NHID = 1500
# number of layers
_C.MODEL.RNN.NLAYERS = 2
# sequence length when back-propogation through time
_C.MODEL.RNN.BPTT = 35
# dropout applied to layers (0 = no dropout)
_C.MODEL.RNN.DROPOUT = 0.65
# tie the word embedding and softmax weights
_C.MODEL.RNN.TIED = True
# gradient clipping
_C.MODEL.RNN.CLIP = 0.25
# whether we randomly shuttle the sequential data or not
_C.MODEL.RNN.SHUFFLE = 0
# set 1 to make the initial hidden state 0!
_C.MODEL.RNN.INIT0 = 0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# default=os.getenv('PT_OUTPUT_DIR', '/tmp')
_C.OUTPUT_DIR = "/tmp"
# default=os.getenv('PHILLY_LOG_DIRECTORY', None)
_C.BACKUP_LOG_DIR = ""
_C.LOG_FREQ = 10
# evaluate model on validation set
_C.EVALUATE = False
# Only save the last checkpoint in the checkpointer
_C.ONLY_SAVE_LAST = 0
