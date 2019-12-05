# Copyright (c) Microsoft. All rights reserved.
"""Unified script for sasa."""
import argparse
import logging
import os
import os.path as op
import sys

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data.distributed
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from mymodels import *
from rnndata import Corpus, batchify
from rnnmodel import RNNModel
from train_val import seq_train, train, seq_evaluate, validate
from utils import mkdir
from utils import get_opt, SasaWriter, Checkpointer, adjust_learning_rate
from zipdata import ZipData

from config import cfg

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
print("torchvision models: \n", model_names)
# ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 
# 'googlenet', 'inception_v3', 'mobilenet_v2', 'resnet101', 'resnet152', 
# 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 
# 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 
# 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

mymodeldict = {
    'mylinear': LinearNet,
    'myvgg': VGG,
    'myresnet18': ResNet18,
    'myresnet50': ResNet50,
    'mypreactresnet18': PreActResNet18,
    'mylenet': GoogLeNet,
    'mydensenet121': DenseNet121,
    'mymobilenet': MobileNet,
    'mymobilenet2': MobileNetV2,
    'mydpn92': DPN92,
    'myshufflenet': ShuffleNetG2,
    'mysenet18': SENet18,
    'mysresnet18': SResNet18,
    'mysresnet50': SResNet50,
}
my_model_names = list(mymodeldict.keys())
print("customized models: \n", my_model_names)
# ['mylinear', 'myvgg', 'myresnet18', 'myresnet50', 'mypreactresnet18',
# 'mylenet', 'mydensenet121', 'mymobilenet', 'mymobilenet2', 'mydpn92',
# 'myshufflenet', 'mysenet18', 'mysresnet18', 'mysresnet50']

seq_model_names = ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']
print("Sequential models: \n", seq_model_names)

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

################### parse experiment settings #####################
parser = argparse.ArgumentParser(description='PyTorch for SASA')
parser.add_argument('--config-file',
                    default="",
                    metavar="FILE",
                    help="path to config file",
                    type=str,
                    )
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--gpu_ids', default="-1", help="gpu id", type=str)
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--data', metavar='DIR', help='path to datasets',
                    default=os.getenv('PT_DATA_DIR', './datasets'))
parser.add_argument('--output_dir', type=str,
                    default=os.getenv('PT_OUTPUT_DIR', '/tmp'))
parser.add_argument('--backup_log_dir', type=str,
                    default=os.getenv('PHILLY_LOG_DIRECTORY', None))
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()

# Set the random seed manually for reproducibility.
if args.seed != 0:
    torch.manual_seed(args.seed)

if args.gpu_ids != '-1':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert (device == 'cuda')
num_gpus = torch.cuda.device_count()
print("Number of GPUs available = {}".format(num_gpus))

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)

cfg.DATA.PATH = args.data
cfg.OUTPUT_DIR = args.output_dir
if args.backup_log_dir is not None:
    cfg.BACKUP_LOG_DIR = args.backup_log_dir


##################### Helper functions ############################
def extend_stats(t_us, t_vs, diff=None, dmean=None, uci=None, lci=None,
                 rhs=None, stds=None,
                 dfs=None):
    if dfs is None:
        dfs = []
    if stds is None:
        stds = []
    if rhs is None:
        rhs = []
    if lci is None:
        lci = []
    if uci is None:
        uci = []
    if dmean is None:
        dmean = []
    if diff is None:
        diff = []
    for u in t_us:
        u_writer.add_scalar('stat', u)
    for v in t_vs:
        v_writer.add_scalar('stat', v)
    for d in diff:
        diff_writer.add_scalar('stat', d)
    for d in dmean:
        dmean_writer.add_scalar('dstat', d)
    for u in uci:
        uci_writer.add_scalar('dstat', u)
    for l in lci:
        lci_writer.add_scalar('dstat', l)
    for r in rhs:
        r_writer.add_scalar('dstat', r)
        r2_writer.add_scalar('dstat', -r)
    if stds:
        iid_writer.add_scalar("std", stds['iid'])
        bm_writer.add_scalar("std", stds['bm'])
        olbm_writer.add_scalar("std", stds['olbm'])
    checkpointer.us.extend(t_us)
    checkpointer.vs.extend(t_vs)


##################### Data ############################
print('==> Preparing data..')
batch_size = cfg.DATALOADER.BSZ
is_replacement = cfg.DATALOADER.RE == "yes"
# use half the bsz for sasapflug.
if 'pflug' in cfg.OPTIM.OPT:
    print("GOT PFLUG")
    batch_size = int(batch_size / 2)

if cfg.DATA.DATASET == "mnist":
    kwargs = {'num_workers': cfg.DATALOADER.WORKERS, 'pin_memory': True}
    trainset = torchvision.datasets.MNIST(root=cfg.DATA.PATH, train=True,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,),
                                                                   (0.3081,))
                                          ]))
    testset = torchvision.datasets.MNIST(root=cfg.DATA.PATH, train=False,
                                         download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,),
                                                                  (0.3081,))
                                         ]))
    sampler = torch.utils.data.sampler.RandomSampler(trainset,
                                                     replacement=is_replacement)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              sampler=sampler, **kwargs)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False, **kwargs)
    num_classes = 10
elif cfg.DATA.DATASET == "cifar":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=cfg.DATA.PATH, train=True,
                                            download=True,
                                            transform=transform_train)
    sampler = torch.utils.data.sampler.RandomSampler(trainset,
                                                     replacement=is_replacement)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=sampler,
                                              num_workers=cfg.DATALOADER.WORKERS)

    testset = torchvision.datasets.CIFAR10(root=cfg.DATA.PATH, train=False,
                                           download=True,
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=cfg.DATALOADER.WORKERS)

    classes = (
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
        'truck')
    num_classes = 10
elif cfg.DATA.DATASET == "cifar100":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR100(root=cfg.DATA.PATH, train=True,
                                             download=True,
                                             transform=transform_train)
    sampler = torch.utils.data.sampler.RandomSampler(trainset,
                                                     replacement=is_replacement)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=sampler,
                                              num_workers=cfg.DATALOADER.WORKERS)

    testset = torchvision.datasets.CIFAR100(root=cfg.DATA.PATH, train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=cfg.DATALOADER.WORKERS)
    num_classes = 100
elif cfg.DATA.DATASET == "imagenet":
    trainpath = os.path.join(cfg.DATA.PATH, 'train.zip')
    train_map = os.path.join(cfg.DATA.PATH, 'train_map.txt')
    valpath = os.path.join(cfg.DATA.PATH, 'val.zip')
    val_map = os.path.join(cfg.DATA.PATH, 'val_map.txt')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainset = ZipData(
        trainpath, train_map,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    testset = ZipData(
        valpath, val_map,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    sampler = torch.utils.data.sampler.RandomSampler(trainset,
                                                     replacement=is_replacement)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=sampler,
        num_workers=cfg.DATALOADER.WORKERS)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=cfg.DATALOADER.WORKERS)
    num_classes = 1000
elif cfg.DATA.DATASET == "wikitext-2":
    corpus = Corpus(cfg.DATA.PATH)
    eval_batch_size = 10
    train_data = batchify(corpus.train, batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)
else:
    raise ValueError("Unimplemented dataset: {}".format(cfg.DATA.DATASET))

## fix warmup based on trainset size, batch size.
if cfg.DATA.DATASET == "wikitext-2":
    steps_per_epoch = len(range(0, train_data.size(0) - 1, cfg.MODEL.RNN.BPTT))
    ntrain = steps_per_epoch * batch_size
else:
    ntrain = len(trainset)
    steps_per_epoch = len(trainloader)
logs_per_epoch = steps_per_epoch / cfg.LOG_FREQ
warmup = cfg.OPTIM.WARMUP * steps_per_epoch
cfg.OPTIM.WARMUP = warmup

# set testing frequency for sequential method.
cfg.OPTIM.SASA.TESTFREQ = int(steps_per_epoch / cfg.OPTIM.SASA.TESTS_PER_EPOCH)
print(cfg.OPTIM.SASA.TESTFREQ)

print("Experiment settings:")
print(cfg)

if cfg.OUTPUT_DIR:
    mkdir(cfg.OUTPUT_DIR)
    # save full config to a file in output_dir for future reference
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(str(cfg))

cfg.freeze()

##################### Model ############################
print('==> Building model..')
if cfg.MODEL.ARCH in model_names:
    logging.info("Use torchvision predefined model")
    if cfg.MODEL.PRETRAINED:
        logging.info("=> using pre-trained model '{}'".format(cfg.MODEL.ARCH))
        net = models.__dict__[cfg.MODEL.ARCH](pretrained=True,
                                              num_classes=num_classes)
    else:
        logging.info("=> creating model '{}'".format(cfg.MODEL.ARCH))
        net = models.__dict__[cfg.MODEL.ARCH](num_classes=num_classes)
elif cfg.MODEL.ARCH in my_model_names:
    logging.info("Use our own customized model")
    if cfg.MODEL.ARCH == 'mylinear':
        if cfg.DATA.DATASET == "mnist":
            input_dim = 784
            output_dim = 10
            bias = True
        else:
            raise ValueError(
                "{} is not supported for linear model!".format(
                    cfg.DATA.DATASET))
        net = mymodeldict[cfg.MODEL.ARCH](input_dim, output_dim, bias)
    elif cfg.MODEL.ARCH == 'myvgg':
        net = mymodeldict[cfg.MODEL.ARCH]('VGG16', num_classes=num_classes)
    elif 'mysresnet' in cfg.MODEL.ARCH:
        net = mymodeldict[cfg.MODEL.ARCH](num_classes=num_classes,
                                          nonlinearity=cfg.MODEL.NONLINEARITY)
    else:
        net = mymodeldict[cfg.MODEL.ARCH](num_classes=num_classes)
elif cfg.MODEL.ARCH in seq_model_names:
    logging.info("Use sequential models")
    assert cfg.DATA.DATASET == "wikitext-2"
    ntokens = len(corpus.dictionary)
    net = RNNModel(cfg.MODEL.ARCH, ntokens, cfg.MODEL.RNN.EMSIZE,
                   cfg.MODEL.RNN.NHID, cfg.MODEL.RNN.NLAYERS,
                   cfg.MODEL.RNN.DROPOUT, cfg.MODEL.RNN.TIED).to(device)
else:
    raise ValueError(
        "Unimplemented model architecture: {}".format(cfg.MODEL.ARCH))
net = net.to(device)

if device == 'cuda':
    if cfg.DATA.DATASET == "wikitext-2":
        net = torch.nn.DataParallel(net, dim=1)
    elif num_gpus > 1:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

##################### Loss function and optimizer ############################
if cfg.LOSS == 'xentropy':
    criterion = nn.CrossEntropyLoss()
elif cfg.LOSS == 'mse':
    criterion = nn.MSELoss()
elif cfg.LOSS == 'bce':
    criterion = nn.BCEWithLogitsLoss()
else:
    raise ValueError("Unimplemented loss: {}".format(cfg.LOSS))

optimizer = get_opt(cfg, net)
scheduler = None

##################### make a checkpoint ############################
if cfg.DATA.DATASET == "wikitext-2":
    # a super small number for neg_best_val_loss
    best_acc = -1.0e7  # neg_best_val_loss
else:
    best_acc = 0.0
checkpointer = Checkpointer(net, cfg.MODEL.ARCH, best_acc=best_acc,
                            optimizer=optimizer, scheduler=scheduler,
                            save_dir=cfg.OUTPUT_DIR,
                            is_test=cfg.EVALUATE,
                            only_save_last=cfg.ONLY_SAVE_LAST)

filepath = cfg.DATA.DATA_DIR + '/{}'.format(cfg.MODEL.MODEL_PATH)
extra_checkpoint_data = checkpointer.load(filepath)

############## tensorboard writers #############################
logstats_freq = cfg.OPTIM.SASA.LOGSTATS if cfg.OPTIM.SASA.LOGSTATS \
    else cfg.OPTIM.SASA.TESTFREQ

if cfg.BACKUP_LOG_DIR:
    train_writer = SasaWriter(os.path.join(cfg.OUTPUT_DIR, 'logs', 'train'),
                              phillypath=os.path.join(cfg.BACKUP_LOG_DIR,
                                                      'logs', 'train'))
    test_writer = SasaWriter(os.path.join(cfg.OUTPUT_DIR, 'logs', 'test'),
                             step_multiplier=logs_per_epoch,
                             phillypath=os.path.join(cfg.BACKUP_LOG_DIR,
                                                     'logs', 'test'))
    u_writer = SasaWriter(os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'u'),
                          phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs',
                                                  'stats', 'u'))
    v_writer = SasaWriter(os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'v'),
                          phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs',
                                                  'stats', 'v'))
    diff_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'diff'),
        phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs', 'stats', 'diff'))
    dmean_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'dmean'),
        step_multiplier=logstats_freq,
        phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs', 'stats', 'dmean'))
    uci_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'uci'),
        step_multiplier=logstats_freq,
        phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs', 'stats', 'uci'))
    lci_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'lci'),
        step_multiplier=logstats_freq,
        phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs', 'stats', 'lci'))
    r_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'rhs'),
        step_multiplier=logstats_freq,
        phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs', 'stats', 'rhs'))
    r2_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', '-rhs'),
        step_multiplier=logstats_freq,
        phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs', 'stats', '-rhs'))
    iid_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'iidvar'),
        step_multiplier=logstats_freq,
        phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs', 'stats', 'iidvar'))
    bm_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'bmvar'),
        step_multiplier=logstats_freq,
        phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs', 'stats', 'bmvar'))
    olbm_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'olbmvar'),
        step_multiplier=logstats_freq,
        phillypath=os.path.join(cfg.BACKUP_LOG_DIR, 'logs', 'stats',
                                'olbmvar'))
else:
    train_writer = SasaWriter(os.path.join(cfg.OUTPUT_DIR, 'logs', 'train'))
    test_writer = SasaWriter(os.path.join(cfg.OUTPUT_DIR, 'logs', 'test'),
                             step_multiplier=logs_per_epoch)
    u_writer = SasaWriter(os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'u'))
    v_writer = SasaWriter(os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'v'))
    diff_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'diff'))
    dmean_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'dmean'),
        step_multiplier=logstats_freq)
    uci_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'uci'),
        step_multiplier=logstats_freq)
    lci_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'lci'),
        step_multiplier=logstats_freq)
    r_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'rhs'),
        step_multiplier=logstats_freq)
    r2_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', '-rhs'),
        step_multiplier=logstats_freq)
    iid_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'iidvar'),
        step_multiplier=logstats_freq)
    bm_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'bmvar'),
        step_multiplier=logstats_freq)
    olbm_writer = SasaWriter(
        os.path.join(cfg.OUTPUT_DIR, 'logs', 'stats', 'olbmvar'),
        step_multiplier=logstats_freq)

if cfg.EVALUATE:
    if cfg.DATA.DATASET == "wikitext-2":
        seq_evaluate(test_data, net, criterion, ntokens, eval_batch_size, -1,
                     cfg, test_writer, checkpointer)
    else:
        validate(testloader, net, criterion, cfg,
                 test_writer, checkpointer, device)

############## training code #############################
# start from epoch 0 or last checkpoint epoch
start_epoch = checkpointer.epoch
for epoch in range(start_epoch, cfg.OPTIM.EPOCHS):
    # anneal by hand for sgd
    if (not cfg.OPTIM.VAL) and (cfg.OPTIM.OPT in ['sgd', 'qhm']):
        adjust_learning_rate(optimizer, epoch, cfg)
    # train for one epoch
    if cfg.DATA.DATASET == "wikitext-2":
        seq_train(train_data, net, criterion, optimizer, epoch, ntokens,
                  batch_size, cfg, checkpointer, extend_stats, train_writer)
    else:
        train(trainloader, net, criterion, optimizer, epoch,
              cfg, extend_stats, train_writer, checkpointer, device)

    # evaluate on validation set
    if cfg.DATA.DATASET == "wikitext-2":
        val_loss = seq_evaluate(val_data, net, criterion, ntokens,
                                eval_batch_size, epoch,
                                cfg, test_writer, checkpointer)
        acc = -val_loss
    else:
        acc = validate(testloader, net, criterion, cfg,
                       test_writer, checkpointer, device)

    with open(cfg.OUTPUT_DIR + '/trainacc.txt', 'w') as f:
        f.write(",".join([str(i) for i in checkpointer.trainacc]))
    with open(cfg.OUTPUT_DIR + '/trainloss.txt', 'w') as f:
        f.write(",".join([str(i) for i in checkpointer.trainloss]))
    with open(cfg.OUTPUT_DIR + '/testacc.txt', 'w') as f:
        f.write(",".join([str(i) for i in checkpointer.testacc]))
    with open(cfg.OUTPUT_DIR + '/testloss.txt', 'w') as f:
        f.write(",".join([str(i) for i in checkpointer.testloss]))
    with open(cfg.OUTPUT_DIR + '/lrs.txt', 'w') as f:
        f.write(",".join([str(i) for i in checkpointer.lrs]))
    with open(cfg.OUTPUT_DIR + '/moms.txt', 'w') as f:
        f.write(",".join([str(i) for i in checkpointer.moms]))
    with open(cfg.OUTPUT_DIR + '/us.txt', 'w') as f:
        f.write(",".join([str(i) for i in checkpointer.us]))
    with open(cfg.OUTPUT_DIR + '/vs.txt', 'w') as f:
        f.write(",".join([str(i) for i in checkpointer.vs]))

    # remember best prec@1 and save checkpoint
    is_best = acc > checkpointer.best_acc
    if is_best:
        checkpointer.best_acc = acc
    elif cfg.OPTIM.VAL and cfg.OPTIM.OPT in \
            ['sgd', 'qhm', 'adam', 'sasa', 'yaida_seq', 'sasa_xd', 'salsa']:
        print("DROPPING LEARNING RATE")
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        for group in optimizer.param_groups:
            group['lr'] = group['lr'] * 1.0 / cfg.OPTIM.DROP_FACTOR
    checkpointer.epoch = epoch + 1
    checkpointer.save(is_best)

# Run on test data.
if cfg.DATA.DATASET == "wikitext-2":
    test_loss = seq_evaluate(test_data, net, criterion, ntokens,
                             eval_batch_size, checkpointer.epoch - 1,
                             cfg, test_writer, checkpointer)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, test_loss))
    print('=' * 89)
