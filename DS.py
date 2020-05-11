import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.datasets as Datasets

import os
import sys
import argparse
import time
import logging

from models import resnet
from utils import utils

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 training')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=233, help='random seed')
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='the diretory to save cifar100 dataset')
    parser.add_argument('--dataset', '-d', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'],
                        help='dataset choice')
    parser.add_argument('--epoch', default=200, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--step_ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--report_freq', default=200, type=int,
                        help='report frequency (default: 20)')
    parser.add_argument('--exp_dir', default='resnet34_DS', type=str,
                        help='folder to save the checkpoints')

    # kd parameter
    parser.add_argument('--beta', default=1e-6, type=float,
                        help='weight of feature loss')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    exp_dir = '{}_{}'.format(args.exp_dir, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(exp_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
        cudnn.benchmark = True
        cudnn.enable = True
        logging.info('using gpu : {}'.format(args.gpu))
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
        logging.info('using cpu')

    if args.dataset == 'cifar100':
        train_transform, valid_transform = utils.cifar100_transform()
        train_data = Datasets.CIFAR100(root=args.data_dir, train=True, transform=train_transform, download=True)
        valid_data = Datasets.CIFAR100(root=args.data_dir, train=False, transform=valid_transform, download=True)
    else:
        raise NotImplementedError

    if not torch.cuda.is_available():
        args.batch_size = 1
        train_data = torch.utils.data.Subset(train_data, range(1))
        valid_data = torch.utils.data.Subset(valid_data, range(1))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=2
    )
    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=2
    )

    if args.dataset == 'cifar100':
        model = resnet.multi_resnet34_kd(100).to(device)
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.epoch):
        lr, beta = utils.adjust_learning_rate(args, optimizer, epoch)
        logging.info('epoch {} lr {:.2e} beta {:.2e}'.format(epoch, lr, beta))

        acc1, acc2, acc3, acc4 = train(args, train_queue, model, criterion, optimizer, device)
        logging.info('train acc {:.4f} {:.4f} {:.4f} {:.4f}'.format(acc1, acc2, acc3, acc4))

        acc1, acc2, acc3, acc4 = infer(valid_queue, model, device)
        logging.info('valid acc {:.4f} {:.4f} {:.4f} {:.4f}'.format(acc1, acc2, acc3, acc4))

        if (epoch+1)%50 == 0:
            utils.save_model(model, os.path.join(exp_dir, 'latest.pt'))

def train(args, train_queue, model, criterion, optimizer, device):
    model.train()
    avg_acc1, avg_acc2, avg_acc3, avg_acc4 = 0, 0, 0, 0
    batch_num = len(train_queue)

    for batch, (input, target) in enumerate(train_queue):
        input = Variable(input, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)

        optimizer.zero_grad()

        logits1, logits2, logits3, logits4, \
        _, _, _, _ = model(input)

        PLoss1, PLoss2, PLoss3, PLoss4 = criterion(logits1, target), criterion(logits2, target), \
                                         criterion(logits3, target), criterion(logits4, target)

        acc1, acc2, acc3, acc4 = utils.accuracy(logits1.data, target, topk=(1,)), \
                                 utils.accuracy(logits2.data, target, topk=(1,)), \
                                 utils.accuracy(logits3.data, target, topk=(1,)), \
                                 utils.accuracy(logits4.data, target, topk=(1,))

        total_loss = PLoss1 + PLoss2 + PLoss3 + PLoss4

        total_loss.backward()
        optimizer.step()

        avg_acc1 += acc1
        avg_acc2 += acc2
        avg_acc3 += acc3
        avg_acc4 += acc4

        if batch % args.report_freq == 0:
            logging.info('train {:0>3d} loss {:.4f} acc {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch,
                                                                                            total_loss.item(),
                                                                                            acc1, acc2, acc3, acc4))

    return avg_acc1 / batch_num, avg_acc2 / batch_num, avg_acc3 / batch_num, avg_acc4 / batch_num

def infer(valid_queue, model, device):
    model.eval()
    avg_acc1, avg_acc2, avg_acc3, avg_acc4 = 0, 0, 0, 0
    batch_num = len(valid_queue)

    for batch, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input, requires_grad=False).to(device)
            target = Variable(target, requires_grad=False).to(device)

            logits1, logits2, logits3, logits4, \
            _, _, _, _ = model(input)
        acc1, acc2, acc3, acc4 = utils.accuracy(logits1.data, target, topk=(1,)), \
                                utils.accuracy(logits2.data, target, topk=(1,)), \
                                utils.accuracy(logits3.data, target, topk=(1,)), \
                                utils.accuracy(logits4.data, target, topk=(1,))

        avg_acc1 += acc1
        avg_acc2 += acc2
        avg_acc3 += acc3
        avg_acc4 += acc4

    return avg_acc1 / batch_num, avg_acc2 / batch_num, avg_acc3 / batch_num, avg_acc4 / batch_num

if __name__ == '__main__':
    main()

