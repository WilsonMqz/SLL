import time
import os
import argparse
import torch
from torch import nn
from torch.backends import cudnn
from my_utils import utils_dataset as dataset
from models.wideresnet import WideResNet
from models.lenet import LeNet
import logging
import numpy as np
from copy import deepcopy
from my_utils.utils_loss import calculate_loss


parser = argparse.ArgumentParser(description='Learning from Stochastic Labels')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--num-workers', default=2, type=int)
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar100', 'combine_mnist', 'mnist', 'fashion', 'kuzushiji', 'cifar10', 'svhn', 'tiny'])
parser.add_argument('--model', type=str, choices=['widenet', 'lenet'], default='widenet')
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--ldr', default=0.1, type=float)
parser.add_argument('--lds', default=80, type=int)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--rate', default=0.3, type=float, help='0.x for random')
parser.add_argument('--seed', help='seed', default='100', type=int)
# parser.add_argument('--data-dir', default='./data/', type=str)
parser.add_argument('--data-dir', default='D:\datasets\source datasets\cifar10', type=str)
parser.add_argument('--lo', default='gce', choices=['ce', 'mae', 'mse', 'gce'], type=str)

args = parser.parse_args()

best_prec1 = 0
num_classes = 10
milestones = [50, 80]
args.epochs = 100
if args.dataset == 'combine_mnist':
    num_classes = 30
if args.dataset == 'cifar100':
    num_classes = 100
    milestones = [100, 150]
    args.epochs = 200
elif args.dataset == 'tiny':
    num_classes = 200
    milestones = [100, 150]
    args.epochs = 200

logging.basicConfig(format='[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.DEBUG,
                    handlers=[logging.StreamHandler()])

args.model_name = 'model_{}_dataset_{}_binomial_{}'.format(args.model, args.dataset, args.rate)

torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# fix random seed
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

save_dir = "./results" + '_' + str(args.seed) + "/res"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = "res_ds_{}_mo_{}_me_SLL_lr_{}_wd_{}_e_{}_bs_{}_rate_{}_lo_{}.csv".format(args.dataset, args.model,
                                                                                  args.lr, args.wd, args.epochs,
                                                                                  args.batch_size, args.rate, args.lo)
model_save_dir = "./results" + '_' + str(args.seed) + "/best_model"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
save_model_name = "best_model_rate_{}_lr_{}_wd_{}.pt".format(args.rate, args.lr, args.wd)
model_save_path = os.path.join(model_save_dir, save_model_name)
record_save_path = os.path.join(save_dir, save_name)
with open(record_save_path, 'a') as f:
    f.writelines("epoch,train_loss,val_loss,val_acc1, val_acc5\n")


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, optimizer, criterion):
    """
        Run one train epoch
    """

    losses = AverageMeter()
    begin_time = time.time()
    time_cost = 0

    model.train()
    for i, (x_aug0, x_aug1, x_aug2, y, TL, SL, index) in enumerate(train_loader):
        x_aug0, x_aug1, x_aug2, TL, SL = \
            x_aug0.to(device), x_aug1.to(device), x_aug2.to(device), TL.to(device), SL.to(device)
        # compute output
        y_pred_aug0 = model(x_aug0)
        y_pred_aug1 = model(x_aug1)
        y_pred_aug2 = model(x_aug2)

        # record loss
        loss1 = calculate_loss(y_pred_aug0, TL, SL, args.rate, num_classes, criterion, args.lo)
        loss2 = calculate_loss(y_pred_aug1, TL, SL, args.rate, num_classes, criterion, args.lo)
        loss3 = calculate_loss(y_pred_aug2, TL, SL, args.rate, num_classes, criterion, args.lo)
        final_loss = (loss1 + loss2 + loss3) / 3

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        losses.update(final_loss.item(), x_aug0.size(0))
        # measure elapsed time
        time_cost = time.time() - begin_time

    return losses.avg, time_cost


def validate(valid_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            input, target = input.to(device), target.to(device)
            input_var = input
            target_var = target

            # compute output
            if args.dataset == 'combine_mnist':
                input_var = input_var.unsqueeze(1)
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, losses.avg, top5.avg


def validate_mnist(valid_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, _, _, target, _, _, index) in enumerate(valid_loader):
            input, target = input.to(device), target.to(device)
            input_var = input
            target_var = target

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, losses.avg, top5.avg



lr_plan = [args.lr] * args.epochs
for i in range(0, args.epochs):
    lr_plan[i] = args.lr * args.ldr ** (i / args.lds)


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]


if __name__ == '__main__':
    # load data
    if args.dataset == 'cifar100':
        train_loader, test_loader = dataset.cifar100_dataloaders(args.data_dir, args.rate, args.batch_size,
                                                                 args.num_workers)
    elif args.dataset == 'combine_mnist':
        train_loader, test_loader = dataset.combine_mnist_dataloaders(args.data_dir, args.rate, args.batch_size,
                                                                 args.num_workers)
    elif args.dataset == 'tiny':
        train_loader, test_loader = dataset.tiny_imagenet_dataloaders(args.data_dir, args.rate, args.batch_size,
                                                                      args.num_workers)
    elif args.dataset == 'fashion':
        train_loader, test_loader = dataset.fashion_dataloaders(args.data_dir, args.rate, args.batch_size,
                                                                      args.num_workers)
    elif args.dataset == 'kuzushiji':
        train_loader, test_loader = dataset.kuzushiji_dataloaders(args.data_dir, args.rate, args.batch_size,
                                                                      args.num_workers)
    elif args.dataset == 'cifar10':
        train_loader, test_loader = dataset.cifar10_dataloaders(args.data_dir, args.rate, args.batch_size,
                                                                      args.num_workers)
    elif args.dataset == 'svhn':
        train_loader, test_loader = dataset.svhn_dataloaders(args.data_dir, args.rate, args.batch_size,
                                                                      args.num_workers)

    # load model
    if args.model == 'widenet':
        model = WideResNet(34, num_classes, widen_factor=10, dropRate=0.0)
    elif args.model == 'lenet':
        model = LeNet(out_dim=num_classes, in_channel=1, img_sz=28)
    else:
        assert "Unknown model"
    model = model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    cudnn.benchmark = True

    # Train loop
    for epoch in range(0, args.epochs):
        logging.info('current lr {:.6f}'.format(optimizer.param_groups[0]['lr']))
        # training
        train_loss, time_cost = train(train_loader, model, optimizer, criterion)
        # lr_step
        scheduler.step()
        # adjust_learning_rate(optimizer, epoch)
        # evaluate on validation set
        if args.dataset == 'combine_mnist':
            val_acc1, val_loss, val_acc5 = validate_mnist(test_loader, model, criterion)
        else:
            val_acc1, val_loss, val_acc5 = validate(test_loader, model, criterion)

        # save best
        if best_prec1 < val_acc1:
            best_prec1 = val_acc1
            best_model_state = deepcopy(model.state_dict())
            torch.save(best_model_state, model_save_path)

        logging.info('Epoch: [{}],  Time_cost {:.3f},  Train_loss {:.4f},  Val_loss {:4f},  Val_acc1 {:.3f}, '
                     'Val_acc5 {:.3f}'.format(epoch, time_cost, train_loss, val_loss, val_acc1, val_acc5))
        with open(record_save_path, 'a') as f:
            f.writelines("{},{:.4f},{:.4f},{:.2f},{:.2f}\n".format(epoch+1, train_loss, val_loss, val_acc1, val_acc5))

    with open(record_save_path, 'a') as f:
        f.writelines("max_acc,{:.2f}\n".format(best_prec1))