# Copyright (c) Microsoft. All rights reserved.
import time
import logging
import torch

from rnndata import repackage_hidden, clone_hidden, get_batch
from utils import get_lr_mom, AverageMeter


def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        if type(output) is not torch.Tensor:
            # inception v3 model
            output = output[0]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_accuracy_multi_binary_label_with_logits(output, target, topk=(40, 13)):
    with torch.no_grad():
        if type(output) is not torch.Tensor:
            # inception v3 model
            output = output[0]
        target = target.type(torch.int)
        acc_all = torch.mean(((output > 0.0) == (target > 0.5)).type(torch.float), dim=0)
        res = []
        for k in topk:
            acc_k = torch.mean(acc_all[:k], dim=0, keepdim=True)
            res.append(acc_k.mul_(100.0))
        return res


def seq_train(train_data, model, criterion, optimizer, epoch, ntokens,
              batch_size, cfg, checkpointer, extend_stats, train_writer):
    total_loss = 0.
    start_time = time.time()
    hidden = model.module.init_hidden(batch_size)
    data_batches = range(0, train_data.size(0) - 1, cfg.MODEL.RNN.BPTT)
    if cfg.MODEL.RNN.SHUFFLE:
        if cfg.DATALOADER.RE == 'yes':
            data_sampler = torch.randint(high=len(data_batches),
                                         size=(len(data_batches),),
                                         dtype=torch.int64).tolist()
        elif cfg.DATALOADER.RE == 'no':
            data_sampler = torch.randperm(len(data_batches)).tolist()
        else:
            raise ValueError(
                "Invalid cfg.DATALOADER.RE input {}".format(cfg.DATALOADER.RE))
    else:
        data_sampler = range(0, len(data_batches))
    for batch, data_i in enumerate(data_sampler):
        i = data_batches[data_i]
        # Turn on training mode which enables dropout.
        model.train()
        # get data
        data, targets = get_batch(train_data, i, cfg.MODEL.RNN.BPTT)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # When cfg.MODEL.RNN.SHUFFLE is true, not initializing with 0 does not
        # make sense. However, we just keep it here.
        hidden = repackage_hidden(hidden, cfg.MODEL.RNN.INIT0)
        if cfg.OPTIM.OPT in ['sgd_sls', 'salsa', 'ssls', 'salsa_new']:
            hidden_clone = clone_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MODEL.RNN.CLIP)

        # closure function defined for line search used in SGD_SLS
        def eval_loss():
            #if cfg.ls_eval:
            if cfg.OPTIM.LS.EVAL:
                model.eval()
            with torch.no_grad():
                output, _ = model(data, hidden_clone)
                loss = criterion(output.view(-1, ntokens), targets)
            return loss

        if cfg.OPTIM.OPT in ['yaida_diag', 'yaida_seq', 'pflug_bat', 'pflug_seq',
                        'sasa_xd_seq', 'sasa_xd']:
            optimizer.step(closure=extend_stats)
        elif cfg.OPTIM.OPT in ['sgd_sls', 'salsa', 'ssls', 'salsa_new']:
            optimizer.step(loss, closure=eval_loss)
        else:
            optimizer.step(closure=None)

        total_loss += loss.item()

        if batch % cfg.LOG_FREQ == 0 and batch > 0:
            cur_loss = total_loss / cfg.LOG_FREQ
            elapsed = time.time() - start_time
            lr, mom = get_lr_mom(optimizer, cfg)
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // cfg.MODEL.RNN.BPTT, lr,
                                  elapsed * 1000 / cfg.LOG_FREQ, cur_loss,
                    cur_loss))
            total_loss = 0
            start_time = time.time()
            train_writer.add_scalar("metrics/top1", cur_loss)
            train_writer.add_scalar("metrics/loss", cur_loss)

            lr, mom = get_lr_mom(optimizer, cfg)

            train_writer.add_scalar("params/lr", lr)
            train_writer.add_scalar("params/mom", mom)
            checkpointer.trainacc.append(cur_loss)
            checkpointer.trainloss.append(cur_loss)
            checkpointer.lrs.append(lr)
            checkpointer.moms.append(mom)


# Training
def train(train_loader, model, criterion, optimizer, epoch,
          cfg, extend_stats, train_writer, checkpointer, device):
    print('\nEpoch: %d' % epoch)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # switch to train mode
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output and record loss
        input, target = input.to(device), target.to(device)
        output = model(input)
        if cfg.LOSS == "bce":
            target = target.type(torch.float32)
        if cfg.MODEL.ARCH == 'inception_v3':
            loss = 0.5 * (criterion(output[0], target) + criterion(output[1], target))
        else:
            loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # measure and record accuracy
        if cfg.LOSS == "xentropy":
            prec1, prec5 = compute_accuracy(output, target, topk=(1, 5))
            top1.update(prec1[0].item(), input.size(0))
            top5.update(prec5[0].item(), input.size(0))
        elif cfg.LOSS == "bce":
            prec1, prec5 = mean_accuracy_multi_binary_label_with_logits(output, target, topk=(40, 13))
            top1.update(prec1[0].item(), input.size(0))
            top5.update(prec5[0].item(), input.size(0))
        else:
            top1.update(0.0, input.size(0))
            top5.update(0.0, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # closure function defined for line search used in SGD_SLS
        def eval_loss():
            #if cfg.ls_eval:
            if cfg.OPTIM.LS.EVAL:
                model.eval()
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target)
            return loss

        if cfg.OPTIM.OPT in ['yaida_diag', 'yaida_seq', 'pflug_bat', 'pflug_seq',
                        'sasa_xd_seq', 'sasa_xd']:
            optimizer.step(closure=extend_stats)
        elif cfg.OPTIM.OPT in ['sgd_sls', 'salsa', 'ssls', 'salsa_new']:
            optimizer.step(loss, closure=eval_loss)
        else:
            optimizer.step(closure=None)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # only log once per cfg.LOG_FREQ param updates. adjust factor because pflug uses
        # 3 batches to make 1 param update.
        if i % cfg.LOG_FREQ == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            train_writer.add_scalar("metrics/top1", top1.val)
            train_writer.add_scalar("metrics/top5", top5.val)
            train_writer.add_scalar("metrics/loss", losses.val)

            lr, mom = get_lr_mom(optimizer, cfg)

            train_writer.add_scalar("params/lr", lr)
            train_writer.add_scalar("params/mom", mom)
            checkpointer.trainacc.append(top1.val)
            checkpointer.trainloss.append(losses.val)
            checkpointer.lrs.append(lr)
            checkpointer.moms.append(mom)


def seq_evaluate(data_source, model, criterion, ntokens, eval_batch_size,
                 epoch, cfg, test_writer, checkpointer):
    # Turn on evaluation mode which disables dropout.
    eval_start_time = time.time()
    model.eval()
    total_loss = 0.
    hidden = model.module.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, cfg.MODEL.RNN.BPTT):
            data, targets = get_batch(data_source, i, cfg.MODEL.RNN.BPTT)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden, 0)

    val_loss = total_loss / (len(data_source) - 1)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - eval_start_time),
                                     val_loss, val_loss))

    test_writer.add_scalar("metrics/top1", val_loss)
    test_writer.add_scalar("metrics/loss", val_loss)

    checkpointer.testloss.append(val_loss)
    checkpointer.testacc.append(val_loss)

    return val_loss


def validate(val_loader, model, criterion,
             cfg, test_writer, checkpointer, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)
            # compute output and record loss
            output = model(input)
            if cfg.LOSS == "bce":
                target = target.type(torch.float32)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure and record accuracy
            if cfg.LOSS == "xentropy":
                prec1, prec5 = compute_accuracy(output, target, topk=(1, 5))
                top1.update(prec1[0].item(), input.size(0))
                top5.update(prec5[0].item(), input.size(0))
            elif cfg.LOSS == "bce":
                prec1, prec5 = mean_accuracy_multi_binary_label_with_logits(output, target, topk=(40, 13))
                top1.update(prec1[0].item(), input.size(0))
                top5.update(prec5[0].item(), input.size(0))
            else:
                top1.update(0.0, input.size(0))
                top5.update(0.0, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.LOG_FREQ == 0:
                logging.info('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        test_writer.add_scalar("metrics/top1", top1.avg)
        test_writer.add_scalar("metrics/top5", top5.avg)
        test_writer.add_scalar("metrics/loss", losses.avg)

        checkpointer.testloss.append(losses.avg)
        checkpointer.testacc.append(top1.avg)

    return top1.avg