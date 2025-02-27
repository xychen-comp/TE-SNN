from __future__ import print_function
import os,time
import torch
import torch.nn as nn
import data
from utils.load_gg12 import GCommandLoader
from utils.utils import AverageMeter
from utils.lib import dump_json, set_seed, count_para
from torch.optim.lr_scheduler import StepLR
from model.spiking_seqmnist_model import TESNN
from model.Hyperparameters import args
from tqdm import tqdm

current_dir = os.path.dirname(os.getcwd())
set_seed(1111)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    end = time.time()
    for i, (text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        text = text.cuda(device,non_blocking=True)  # text:[bs, 1, 80, T]
        target = target.cuda(device,non_blocking=True)

        text_in = text.squeeze(1).permute(2, 0, 1)  # text:[T, bs, 80]
        output = model(text_in)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), text.size(0))
        top1.update(acc1[0], text.size(0))
        top5.update(acc5[0], text.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % 100 == 0:
          print('Train Epoch: {}, Step: [{}/{}] lr: {:.6f}, top1: {:.4f}'.format(epoch, i + 1, len(train_loader), optimizer.param_groups[0]['lr'],top1.avg))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (text, target) in enumerate(val_loader):
            text = text.cuda(device,non_blocking=True)
            target = target.cuda(device,non_blocking=True)

            text_in = text.squeeze(1).permute(2, 0, 1)  # text:[T, bs, 80]

            output = model(text_in)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), text.size(0))
            top1.update(acc1[0], text.size(0))
            top5.update(acc5[0], text.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":

    # CUDA configuration
    if torch.cuda.is_available():
        device = 'cuda:'+str(args.cuda) 
        print('GPU is available')
    else:
        device = 'cpu'
        print('GPU is not available')
    
    torch.set_num_threads(10)
    home_dir = current_dir  # Relative path
    data_path = args.data_path
    snn_ckp_dir = os.path.join(home_dir, 'exp/GSC/checkpoint/')
    snn_rec_dir = os.path.join(home_dir, 'exp/GSC/record/')
    print(snn_ckp_dir)

    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    arch_name = '-'.join([str(s) for s in args.fc])

    print('algo is %s, dropout = %.2f, thresh = %.2f, lens = %.2f, decay = %.2f, lr = %.5f, wd = %.5f, beta = %.2f' %
          (args.te, args.dropout, args.thresh, args.lens, args.decay, args.lr, args.wd, args.beta))
    print('Arch: {0}'.format(arch_name))
    data.google12_v2(version='v2')
    torch.backends.cudnn.benchmark = True

    # GSC V2
    train_dataset = GCommandLoader(data_path + '/google_speech_command_2/processed/train',
                                   window_size=.02, max_len=args.time_window)
    test_dataset = GCommandLoader(data_path + '/google_speech_command_2/processed/test',
                                  window_size=.02, max_len=args.time_window)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None, num_workers=8, pin_memory=True, sampler=None)

    if args.recurrent:
        train_record_path = 'GSC_srnn_step_TESNN_{0}_dropout{1}_decay{2}_beta{3}_arch{4}_lr{5}.txt' \
            .format(args.te, args.dropout, args.decay, args.beta, arch_name, learning_rate)
        train_chk_pnt_path = 'GSC_srnn_step_TESNN_{0}_dropout{1}_decay{2}_beta{3}_arch{4}_lr{5}.pt' \
            .format(args.te, args.dropout, args.decay, args.beta, arch_name, learning_rate)
    else:
        train_record_path = 'GSC_sfnn_step_TESNN_{0}_dropout{1}_decay{2}_beta{3}_arch{4}_lr{5}.txt' \
            .format(args.te, args.dropout, args.decay, args.beta, arch_name, learning_rate)
        train_chk_pnt_path = 'GSC_sfnn_step_TESNN_{0}_dropout{1}_decay{2}_beta{3}_arch{4}_lr{5}.pt' \
            .format(args.te, args.dropout, args.decay, args.beta, arch_name, learning_rate)
    

    # Network initialization
    net = TESNN(in_size=40, output_size=12, dropout=args.dropout)
    net = net.to(device)

    para=count_para(net)
    print(net,'Parameter counts:',para)

    criterion = nn.CrossEntropyLoss()
    train_best_acc = 0
    test_best_acc = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    train_acc_record = list([])
    test_acc_record = list([])
    loss_train_record = list([])
    loss_test_record = list([])
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=args.wd)  # define weight update method
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in tqdm(range(start_epoch, start_epoch + args.epochs)):
        starts = time.time()
        train_acc, train_loss = train(train_loader, net, criterion, optimizer, epoch, args)
        acc1, acc5 = validate(test_loader, net, criterion, args)
        scheduler.step()
        if acc1.item() >= test_best_acc:
            test_best_acc = acc1.item()
            # Save Model
            print("Saving the model.")
            if not os.path.isdir(snn_ckp_dir):
                os.makedirs(snn_ckp_dir)
            state = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': test_best_acc,
            }
            torch.save(state, snn_ckp_dir + train_chk_pnt_path)
        
        print('Test Epoch: [{}/{}], lr: {:.6f}, train acc: : {:.4f}, test acc: {:.4f}, best: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], train_acc.item(), acc1.item(), test_best_acc))


        loss_train_record.append(train_loss)
        train_acc_record.append(train_acc.item())
        test_acc_record.append(acc1.item())

        if train_best_acc < train_acc.item():
            train_best_acc = train_acc.item()

        elapsed = time.time() - starts
        print('Time past: ', elapsed, 's', 'Iter number:', epoch+1)

        if not os.path.isdir(snn_ckp_dir):
            os.makedirs(snn_ckp_dir)

        training_record = {
            'learning_rate': args.lr,
            'algo': args.algo,
            'thresh': args.thresh,
            'lens': args.lens,
            'decay': args.decay,
            'architecture': args.fc,
            'loss_train_record': loss_train_record,
            'test_acc_record': test_acc_record,
            'train_acc_record': train_acc_record,
            'train_best_acc': train_best_acc,
            'test_best_acc': test_best_acc,
        }
        dump_json(training_record, snn_rec_dir, train_record_path)

    print(" Best Train Acc: ", train_best_acc)
    print(" Best Test Acc: ", test_best_acc)
    print('=====================End of trail=============================\n\n')

