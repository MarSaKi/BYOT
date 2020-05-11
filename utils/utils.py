import torch
import numpy as np
import torchvision.transforms as transforms

def cifar100_transform():
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def count_params(model):
    return sum(np.prod(v.shape) for name,v in model.named_parameters())/1e6

def kd_loss_function(output, target_output, args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output_t = output / args.temperature
    output_log_softmax = torch.log_softmax(output_t, dim=1)
    target_output_t = target_output / args.temperature
    target_output_softmax = torch.softmax(target_output_t, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output_softmax, dim=1))
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


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


def adjust_learning_rate(args, optimizer, epoch):
    if epoch < 75:
        lr = args.lr
        beta = args.beta
    elif 75 <= epoch < 130:
        lr = args.lr * (args.step_ratio ** 1)
        beta = args.beta * (args.step_ratio ** 1)
    elif 130 <= epoch < 180:
        lr = args.lr * (args.step_ratio ** 2)
        beta = args.beta * (args.step_ratio ** 2)
    elif epoch >= 180:
        lr = args.lr * (args.step_ratio ** 3)
        beta = args.beta * (args.step_ratio ** 3)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr, beta


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))

    return res[0].item()

def save_model(model, model_path):
  torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
  model.load_state_dict(torch.load(model_path))