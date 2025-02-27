import argparse
import json
import os
import logging
import math
import time
from functools import partial

import torch
from torch.cuda import amp
from datetime import datetime
import utils.distributed as distributed
from utils.utils import set_random_seed, setup_logging, save_checkpoint, \
    AverageMeter, ProgressMeter
from model.surrogate import SurrogateGradient
from model.te_lif import TELIF
from model.lm_snn import LMSNN
parser = argparse.ArgumentParser(description='PyTorch Training')
# args of datasets
parser.add_argument('--dataset', default='PTB', type=str,
                    help='dataset: [PTB]')
parser.add_argument('--data_path', default='/path',
                    help='path to dataset,')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--use-ddp', action='store_true', default=False, help='')
parser.add_argument('--amp', action='store_true', help='automatic mixed precision training')
parser.add_argument('--save-path', default='', type=str, help='the directory used to save the trained models')
parser.add_argument('--name', default='', type=str,
                    help='name of experiment')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-ckpt', action='store_true', default=True, help='')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# args of optimizer
parser.add_argument('--lr', '--learning-rate', default=3., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1.2e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# Cosine learning rate
parser.add_argument('--cos-lr', action='store_true', default=True,
                    help='whether to use cosine learning rate')

# args of spiking neural networks
parser.add_argument('--threshold', type=float, default=0.5, help='neuronal threshold (default: 1)')
parser.add_argument('--time-window', type=int, default=70, help='total time steps (default: 10)')
parser.add_argument('--decay', type=float, default=0.5, help='decay factor (default: 5)')
parser.add_argument('--learning-rule', default='STBP', type=str, help='')
parser.add_argument('--detach-mem', action='store_true', default=False, help='')
parser.add_argument('--detach-reset', action='store_true', default=False, help='')
parser.add_argument('--grad-clip', type=float, default=0.25)

parser.add_argument('--dropout-emb', type=float, default=0.4, help='default: 0.4 on PTB')
parser.add_argument('--dropout-words', type=float, default=0.1, help='default: 0.1 on PTB')
parser.add_argument('--dropout-forward', type=float, default=0.25, help='default: 0.25 on PTB')
parser.add_argument('--dropout', type=float, default=0.4, help='default: 0.4 on PTB')
parser.add_argument('--emb-dim', type=int, default=400, help='total time steps (default: 10)')
parser.add_argument('--hidden-dim', type=int, default=1100, help='total time steps (default: 10)')

parser.add_argument('--te', default='TE-N', type=str, choices=['TE-N', 'TE-R'])
parser.add_argument('--recurrent', default=False, action='store_true', help='Feedforward or recurrent')

parser.add_argument('--beta', type=float, default=0.1, help='decay factor of TE')
def main():
    args = parser.parse_args()
    if args.save_path == '':
        save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = save_path + args.name + '_' + str(args.seed)
        if args.amp:
            save_path += '_amp'
    else:
        save_path = args.save_path

    if args.use_ddp:
        distributed.init_distributed_mode(args)

    if args.use_ddp:
        torch.distributed.barrier()
        if distributed.is_main_process():
            from pathlib import Path

            Path(save_path).mkdir(parents=True, exist_ok=True)
            # Logging settings
            setup_logging(os.path.join(save_path, 'log.txt'))
            logging.info('saving to:' + str(save_path))
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Logging settings
        setup_logging(os.path.join(save_path, 'log.txt'))
        logging.info('saving to:' + str(save_path))

    if args.use_ddp:
        is_cuda = torch.cuda.is_available()
        assert is_cuda
        device = torch.device('cuda', args.gpu)
        set_random_seed(seed=args.seed, is_ddp=True)
        torch.backends.cudnn.benchmark = True
    else:
        is_cuda = torch.cuda.is_available()
        assert is_cuda, 'CPU is not supported!'
        device = torch.device('cuda' if is_cuda else 'cpu')
        set_random_seed(seed=args.seed, is_ddp=False)
        torch.backends.cudnn.benchmark = False
        args.gpu = 'cuda'

    if not args.use_ddp or (args.use_ddp and distributed.is_main_process()):
        with open(save_path + '/args.json', 'w') as fid:
            json.dump(args.__dict__, fid, indent=2)

    logging.info('args:' + str(args))

    from data.penn_treebank import PennTreebank
    T = args.time_window
    B = args.batch_size
    train_dataset = PennTreebank(root="/benchmark_data/PennTreebank", subset="train", time_step=T, chunk_num=B, device=device)
    val_dataset   = PennTreebank(root="/benchmark_data/PennTreebank", subset="valid", time_step=T, chunk_num=10, device=device)
    test_dataset  = PennTreebank(root="/benchmark_data/PennTreebank", subset="test",  time_step=T, chunk_num=1, device=device)
    vocab_size = 10000
    ######

    logging.info(f"Dataset {args.dataset} has {vocab_size} tokens")

    # TODO: Build spiking model
    args.surrogate = 'rectangle'
    surro_grad = SurrogateGradient(func_name=args.surrogate, a=1.0)
    exec_mode = "serial"
    if args.te == 'TE-N':
        beta = args.beta
        spiking_neuron = partial(TELIF,
                                 te_type='N',
                                 decay=args.decay,
                                 threshold=args.threshold,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad,
                                 exec_mode=exec_mode,
                                 recurrent=args.recurrent,
                                 beta=beta
                                 )
    elif args.te == 'TE-R':
        beta = args.beta
        spiking_neuron = partial(TELIF,
                                 te_type='R',
                                 decay=args.decay,
                                 threshold=args.threshold,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad,
                                 exec_mode=exec_mode,
                                 recurrent=args.recurrent,
                                 beta=beta
                                 )
    else:
        raise NotImplementedError
    args.multi_step = True

    model = LMSNN(rnn_type=args.te,
                  nlayers=2,
                  emb_dim=args.emb_dim,
                  hidden_dim=args.hidden_dim,
                  vocab_size=vocab_size,
                  dropout_words=args.dropout_words,
                  dropout_embedding=args.dropout_emb,
                  dropout_forward=args.dropout_forward,
                  dropout=args.dropout,
                  spiking_neuron=spiking_neuron,
                  args=args,
                  )

    logging.info(str(model))
    logging.info(f"Model number of parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1024 / 1024):.4f} M")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.0
                                )

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = None
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
    best_val_ppl = float('inf')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            logging.info(f"best_val_ppl: {checkpoint['best_val_ppl']}")
            args.start_epoch = checkpoint['epoch']
            best_val_ppl = checkpoint['best_val_ppl']
            state_dict = checkpoint['state_dict']
            for (key, value) in list(state_dict.items()):
                if key.startswith('module.'):
                    state_dict[key.replace("module.", "")] = value
                del state_dict[key]

            msg = model.load_state_dict(state_dict, strict=False)
            logging.info(msg)
            logging.info("=> loading optimizer of checkpoint)")
            optimizer.load_state_dict(checkpoint['optimizer'])
            for k, v in optimizer.state.items():  # key is Parameter, val is a dict {key='momentum_buffer':tensor(...)}
                if 'momentum_buffer' not in v:
                    continue
                optimizer.state[k]['momentum_buffer'] = optimizer.state[k]['momentum_buffer'].cuda(args.gpu)
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                         .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            find_unused_parameters=True,
        )
    else:
        model = model.cuda()
    try:
        standard_train(train_dataset, val_dataset, model, criterion, optimizer, scheduler, save_path, best_val_ppl, scaler, vocab_size,
                       args)
    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')
    ######################################################################
    # Evaluate the best model on the test dataset
    # -------------------------------------------
    if not args.use_ddp or (args.use_ddp and distributed.is_main_process()):
        best_model_checkpoint = torch.load(os.path.join(save_path, 'model_best.pth.tar'))
        model.load_state_dict(best_model_checkpoint['state_dict'])
        test_ppl = validate_one_epoch(test_dataset, model, criterion, vocab_size, 1, args)
        logging.info('=' * 89)
        logging.info(f'| End of training | test ppl {test_ppl:8.2f}')
        logging.info('=' * 89)


def standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_val_ppl, scaler, vocab_size,
                   args):
    for epoch in range(args.start_epoch, args.epochs):
        if args.use_ddp:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_ppl, train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, vocab_size, args)
        if not args.use_ddp or (args.use_ddp and distributed.is_main_process()):
            # evaluate on validation set
            val_ppl = validate_one_epoch(val_loader, model, criterion, vocab_size, 10, args)
            out_string = 'Train ppl. {:8.2f} Val ppl {:8.2f} \t'.format(train_ppl, val_ppl)
            logging.info(out_string)
            # remember best acc@1 and save checkpoint
            is_best = val_ppl < best_val_ppl
            best_val_ppl = min(val_ppl, best_val_ppl)
            if args.save_ckpt:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_ppl': best_val_ppl,
                    'optimizer': optimizer.state_dict(),
                }, is_best, filename=os.path.join(save_path, 'checkpoint.pth.tar'), save_path=save_path)
    logging.info(f'Best best_val_ppl: {best_val_ppl}')


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is not None:
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, ntokens, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    num_batches = len(train_loader)

    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    hidden = model.init_hidden(args.batch_size)

    for batch_index, (data, targets) in enumerate(train_loader): 
        # measure data loading time
        data_time.update(time.time() - end)
        data = data.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)

        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)

        loss.backward()
        # for param in model.parameters():
        #     grad_norm = torch.norm(param.grad, p=2)
        #     print(grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_rec = criterion(output.view(-1, ntokens), targets)

        losses.update(loss_rec.item(), data.numel())

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_index + 1) % args.print_freq == 0 or (batch_index + 1) == num_batches:
            progress.display(batch_index + 1)
            # logging.info(f"ppl: {math.exp(losses.avg)}")

    return math.exp(losses.avg), losses.avg


def validate_one_epoch(val_loader, model, criterion, ntokens, eval_batch_size, args):
    losses = AverageMeter('Loss', ':.4e')

    # switch to evaluate mode
    model.eval()
    seq_length = args.time_window
    # iter_range = range(0, val_loader.size(0) - 1, seq_length)

    with torch.no_grad():
        # initialize hidden states
        hidden = model.init_hidden(eval_batch_size)
        for _, (data, targets) in enumerate(val_loader): 
            data = data.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
            output, hidden = model(data, hidden) # [T, B, N]
            loss = criterion(output.view(-1, ntokens), targets)
            losses.update(loss.item(), data.numel())
    return math.exp(losses.avg)


if __name__ == '__main__':
    main()
