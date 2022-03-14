import argparse
import os, sys
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models

import torchvision.models as models_torch

from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from torch.nn.parameter import Parameter

#progress bar
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='garbage',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet18_binary',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18_binary)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=5e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.25e-4, type=float,
                    metavar='W', help='weight decay (default: 0.25e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('-prt', '--pretrained', type=str, metavar='FILE',
                    help='pretrained model FILE')
parser.add_argument('-wb', '--wbits', default=1, type=int,
                    help='bitwidth for weights')
parser.add_argument('-acc', '--acc_bits', default=8, type=int,
                    help='bitwidth for accumulator')
parser.add_argument('-t', '--t', default=64, type=int,
                    help='size of Tile (default: 64)')
parser.add_argument('-k', '--k', default=2, type=int,
                    help='WrapNet slope (default: 2)')
parser.add_argument('-s', '--s', default=2.0, type=float,
                    help='psum step size (default: 2.0)')

def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = './results'
    if args.save is '':
        args.save = datetime.now().strftime('/garbage')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset,
                    'nbits': args.wbits, 'T': args.t, 'nbits_acc': args.acc_bits,
                    'k': args.k, 's': args.s}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    print(model)
    logging.info("created model with configuration: %s", model_config)


    if args.pretrained:
        if not os.path.isfile(args.pretrained):
            parser.error('invalid checkpoint: {}'.format(args.pretrained))

        checkpoint = torch.load(args.pretrained)
        #load_my_state_dict(model, checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.pretrained, checkpoint['epoch'])
        #print(models_torch.vgg16())
        #return

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        #model.load_state_dict(checkpoint['state_dict'], strict=False)
        load_my_state_dict(model, checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)

    #regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
    #                                       'lr': args.lr,
    #                                       'momentum': args.momentum,
    #                                       'weight_decay': args.weight_decay}})
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()

    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    #num, mean, std
    psums = [[0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0]]

    if args.evaluate:
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, psums=psums)
        print('Best Accuracy:', val_prec1)
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=args.start_epoch-1)

    #logging.info('training regime: %s', regime)


    for epoch in range(args.start_epoch, args.epochs):
        #for param_group in optimizer.param_groups:
        #    print('learning rate =', param_group['lr'])
        print('learning_rate=', scheduler.get_last_lr())

        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)


        #if is_best:
        #    torch.save(model.state_dict(), '~/results/model_best.pth.tar')

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1
            #'regime': regime
        }, is_best, path=save_path)

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5)
        #results.plot(x='epoch', y=['train_loss', 'val_loss'],
        #             title='Loss', ylabel='loss')
        #results.plot(x='epoch', y=['train_error1', 'val_error1'],
        #             title='Error@1', ylabel='error %')
        #results.plot(x='epoch', y=['train_error5', 'val_error5'],
        #             title='Error@5', ylabel='error %')
        results.save()

        print('Best Precision:', best_prec1)
        scheduler.step()


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, psums=None):
    if args.gpus and len(args.gpus) > 1:
        '''
        torch.cuda.set_device(args.gpus)
        model.cuda(args.gpus)
        args.batch_size = int(args.batch_size / len(args.gpus))
        args.workers = int((args.workers + len(args.gpus) - 1)/len(args.gpus))

        model = torch.nn.parallel.DistrbutedDataParallel(model, device_ids=[args.gpus])
        '''
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=len(data_loader))
    for i, (inputs, target) in enumerate(data_loader):
        #if i > 10:
        #    break
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda()

        if not training:
            with torch.no_grad():
                input_var = Variable(inputs.type(args.type), volatile=not training)
                target_var = Variable(target)
                # compute output
                output, _ = model(input_var, psums)

                #num, mean, std
                psums = [[0, 0, 0],[0, 0, 0],
                         [0, 0, 0],[0, 0, 0],
                         [0, 0, 0],[0, 0, 0],
                         [0, 0, 0],[0, 0, 0],
                         [0, 0, 0],[0, 0, 0],
                         [0, 0, 0],[0, 0, 0],
                         [0, 0, 0],[0, 0, 0]]
        else:
            input_var = Variable(inputs.type(args.type), volatile=not training)
            target_var = Variable(target)
            # compute output
            output = model(input_var)

        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '{phase} - ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | p: {p: .4f}'.format(
                    phase='TRAINING' if training else 'EVALUATING',
                    batch=i + 1,
                    size=len(data_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ss=16.0,
                    p=psums[0][1],
                    #ss=model.layer1[0].conv1.step_size_psum[0],
                    )
        bar.next()

        #if i % args.print_freq == 0:
        #    logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
        #                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #                 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #                     epoch, i, len(data_loader),
        #                     phase='TRAINING' if training else 'EVALUATING',
        #                     batch_time=batch_time,
        #                     data_time=data_time, loss=losses, top1=top1, top5=top5))
    bar.finish()

    print(psums)
    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch, psums):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None, psums=psums)

def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()

    for name, param in state_dict.items():
        if 'layer' in name:
            temp = name.split('layer')
            list1 = list(temp[1])
            list1[1] = '_'
            temp[1] = ''.join(list1)
            name = 'layer' + temp[1]

        if 'downsample' in name:
            temp = name.split('downsample')
            temp1 = temp[1].split('.')
            if '0' in temp[1]:
                temp[1] = '.conv1.' + temp1[2]
            elif '1' in temp[1]:
                temp[1] = '.bn1.' + temp1[2]

            name = temp[0] + 'downsample' + temp[1]
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

if __name__ == '__main__':
    main()
