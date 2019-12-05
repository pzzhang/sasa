# Copyright (c) Microsoft. All rights reserved.
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import errno
import logging
import shutil
import optim
import torch
from tensorboardX import SummaryWriter


def mkdir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class SasaWriter():
    def __init__(self, path, step_multiplier=1, phillypath=None):
        self.steps = {}
        self.sm = step_multiplier
        self.writer = SummaryWriter(path)
        self.phillywriter = None
        if phillypath:
            self.phillywriter = SummaryWriter(phillypath)

    def add_scalar(self, name, val):
        if name not in self.steps:
            self.steps[name] = 0
        self.writer.add_scalar(name, val, self.sm * self.steps[name])
        if self.phillywriter:
            self.phillywriter.add_scalar(name, val, self.sm * self.steps[name])
        self.steps[name] += 1


def get_opt(cfg, net):
    lr = cfg.OPTIM.LR
    momentum = cfg.OPTIM.MOM

    if cfg.OPTIM.OPT == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.OPT == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.OPTIM.LR,
                                     betas=(
                                         cfg.OPTIM.ADAM.BETA1,
                                         cfg.OPTIM.ADAM.BETA2),
                                     weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.OPT == 'qhm':
        optimizer = optim.QHM(net.parameters(), lr=cfg.OPTIM.LR,
                              momentum=momentum,
                              nu=cfg.OPTIM.NU, weight_decay=cfg.OPTIM.WD)

    elif cfg.OPTIM.OPT == 'yaida_seq':
        optimizer = optim.SASAYaida(net.parameters(), lr=lr,
                                    momentum=momentum,
                                    weight_decay=cfg.OPTIM.WD,
                                    minN=cfg.OPTIM.SASA.N,
                                    maxN=20 * cfg.OPTIM.SASA.N,
                                    warmup=cfg.OPTIM.WARMUP,
                                    sigma=cfg.OPTIM.SASA.SIGMA,
                                    zeta=1.0 / cfg.OPTIM.DROP_FACTOR,
                                    delta=cfg.OPTIM.SASA.DELTA,
                                    testfreq=cfg.OPTIM.SASA.TESTFREQ,
                                    mode=cfg.OPTIM.SASA.MODE,
                                    logstats=cfg.OPTIM.SASA.LOGSTATS)
    elif cfg.OPTIM.OPT == 'yaida_ratio':
        optimizer = optim.Yaida(net.parameters(), lr=lr, momentum=momentum,
                                weight_decay=cfg.OPTIM.WD,
                                minN=cfg.OPTIM.SASA.N,
                                maxN=20 * cfg.OPTIM.SASA.N,
                                warmup=cfg.OPTIM.WARMUP,
                                sigma=cfg.OPTIM.SASA.SIGMA,
                                zeta=1.0 / cfg.OPTIM.DROP_FACTOR,
                                delta=cfg.OPTIM.SASA.DELTA,
                                testfreq=cfg.OPTIM.SASA.TESTFREQ)
    elif cfg.OPTIM.OPT == 'sasa_xd':
        optimizer = optim.SASA_xd(net.parameters(), lr=lr, momentum=momentum,
                                  nu=cfg.OPTIM.NU,
                                  weight_decay=cfg.OPTIM.WD,
                                  drop_factor=cfg.OPTIM.DROP_FACTOR,
                                  sigma=cfg.OPTIM.SASA.SIGMA,
                                  var_mode=cfg.OPTIM.SASA.MODE,
                                  leaky_ratio=cfg.OPTIM.SASA.LEAKY_RATIO,
                                  minN=cfg.OPTIM.SASA.N,
                                  warmup=cfg.OPTIM.WARMUP,
                                  testfreq=cfg.OPTIM.SASA.TESTFREQ,
                                  logstats=cfg.OPTIM.SASA.LOGSTATS)
    elif cfg.OPTIM.OPT == 'sgd_sls':
        optimizer = optim.SGD_SLS(net.parameters(), lr=lr, momentum=momentum,
                                  nu=cfg.OPTIM.NU,
                                  weight_decay=cfg.OPTIM.WD,
                                  gamma=cfg.OPTIM.LS.GAMMA,
                                  ls_evl=cfg.OPTIM.LS.EVAL,
                                  ls_sdc=cfg.OPTIM.LS.SDC,
                                  ls_inc=cfg.OPTIM.LS.INC,
                                  ls_dec=cfg.OPTIM.LS.DEC,
                                  ls_max=cfg.OPTIM.LS.MAX,
                                  ls_ign=cfg.OPTIM.LS.IGN)
    elif cfg.OPTIM.OPT == 'salsa':
        optimizer = optim.SALSA(net.parameters(), lr=lr, momentum=momentum,
                                nu=cfg.OPTIM.NU,
                                weight_decay=cfg.OPTIM.WD,
                                gamma=cfg.OPTIM.LS.GAMMA,
                                ls_evl=cfg.OPTIM.LS.EVAL,
                                ls_sdc=cfg.OPTIM.LS.SDC,
                                ls_inc=cfg.OPTIM.LS.INC,
                                ls_dec=cfg.OPTIM.LS.DEC,
                                ls_max=cfg.OPTIM.LS.MAX,
                                warmup=cfg.OPTIM.WARMUP,
                                drop_factor=cfg.OPTIM.DROP_FACTOR,
                                sigma=cfg.OPTIM.SASA.SIGMA,
                                var_mode=cfg.OPTIM.SASA.MODE,
                                leaky_ratio=cfg.OPTIM.SASA.LEAKY_RATIO,
                                minN=cfg.OPTIM.SASA.N,
                                testfreq=cfg.OPTIM.SASA.TESTFREQ,
                                logstats=cfg.OPTIM.SASA.LOGSTATS)

    return optimizer


def get_lr_mom(optimizer, cfg):
    lr, mom = None, None
    # todo: better logging for adam.
    if cfg.OPTIM.OPT == 'adam':
        lr = optimizer.param_groups[0]['lr']
        mom = optimizer.param_groups[0]['betas'][
            0]  # this isn't right to compare.
    else:
        lr = optimizer.param_groups[0]['lr']
        mom = optimizer.param_groups[0]['momentum']

    assert (lr is not None)
    assert (mom is not None)
    return lr, mom


def adjust_learning_rate(optimizer, epoch, cfg):
    """Sets the learning rate to the initial LR decayed by cfg.OPTIM.DROP_FACTOR every 30 epochs"""
    lr = cfg.OPTIM.LR * 1.0 / (
            cfg.OPTIM.DROP_FACTOR ** (epoch // cfg.OPTIM.DROP_FREQ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Checkpointer(object):
    def __init__(
            self,
            model,
            arch,
            optimizer=None,
            scheduler=None,
            save_dir="",
            logger=None,
            is_test=False,
            epoch=0,
            best_acc=0.,
            trainloss=None,
            trainacc=None,
            testloss=None,
            testacc=None,
            lrs=None,
            moms=None,
            us=None,
            vs=None,
            only_save_last=0
    ):
        self.model = model
        self.arch = arch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.is_test = is_test
        self.resume = False
        self.epoch = epoch
        self.best_acc = best_acc
        self.only_save_last = only_save_last
        if vs is None:
            vs = []
        if us is None:
            us = []
        if moms is None:
            moms = []
        if lrs is None:
            lrs = []
        if testacc is None:
            testacc = []
        if testloss is None:
            testloss = []
        if trainacc is None:
            trainacc = []
        if trainloss is None:
            trainloss = []
        self.trainloss = trainloss
        self.trainacc = trainacc
        self.testloss = testloss
        self.testacc = testacc
        self.lrs = lrs
        self.moms = moms
        self.us = us
        self.vs = vs

    def save(self, is_best, **kwargs):
        name = 'checkpoint_{}'.format(self.epoch)
        if self.only_save_last:
            name = 'checkpoint_last'

        if not self.save_dir:
            return

        data = {"net": self.model.state_dict(), "arch": self.arch,
                "epoch": self.epoch, "best_acc": self.best_acc,
                "trainloss": self.trainloss, "trainacc": self.trainacc,
                "testloss": self.testloss, "testacc": self.testacc,
                "lrs": self.lrs, "moms": self.moms,
                "us": self.us, "vs": self.vs}
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        # self.tag_last_checkpoint(save_file)
        # use relative path name to save the checkpoint
        self.tag_last_checkpoint("{}.pth".format(name))

        if is_best:
            shutil.copyfile(save_file,
                            os.path.join(self.save_dir, "model_best.pth"))

    def load(self, f=None):
        if self.is_test and os.path.isfile(f):
            # load the weights in config file if it is specified in testing
            # stage otherwise it will load the lastest checkpoint in
            # output_dir for testing
            self.logger.info("Loading checkpoint from {}".format(f))
            checkpoint = self._load_file(f)
            self._load_model(checkpoint)
            return checkpoint

        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
            # get the absolute path
            f = os.path.join(self.save_dir, f)
            self.resume = True
        if not os.path.isfile(f):
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from "
                             "scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        # if resume training, load optimizer and scheduler,
        # otherwise use the specified LR in config yaml for fine-tuning
        if self.resume:
            self.trainloss = checkpoint.pop('trainloss')
            self.trainacc = checkpoint.pop('trainacc')
            self.testloss = checkpoint.pop('testloss')
            self.testacc = checkpoint.pop('testacc')
            self.lrs = checkpoint.pop('lrs')
            self.moms = checkpoint.pop('moms')
            self.us = checkpoint.pop('us')
            self.vs = checkpoint.pop('vs')
            if "epoch" in checkpoint:
                self.epoch = checkpoint.pop('epoch')
            if "best_acc" in checkpoint:
                self.best_acc = checkpoint.pop('best_acc')
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f.strip(), map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        assert checkpoint.pop('arch') == self.arch
        self.model.load_state_dict(checkpoint.pop("net"))
