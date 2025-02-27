from __future__ import print_function
import os,time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils.lib import dump_json, set_seed, count_para
from torch.optim.lr_scheduler import StepLR
from model.spiking_seqmnist_model import TESNN
from model.Hyperparameters import args
current_dir = os.path.dirname(os.getcwd())

set_seed(args.seed)

if __name__ == "__main__":

    # CUDA configuration
    if torch.cuda.is_available():
        device = 'cuda'
        print('GPU is available')
    else:
        device = 'cpu'
        print('GPU is not available')

    home_dir = current_dir  # Relative path
    data_path = args.data_path
    snn_ckp_dir = os.path.join(home_dir, 'exp/SeqMNIST/checkpoint/')
    snn_rec_dir = os.path.join(home_dir, 'exp/SeqMNIST/record/')
    print(snn_ckp_dir)

    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    arch_name = '-'.join([str(s) for s in args.fc])

    print('Recurrent architecture is %s, algo is %s, permuted = %s thresh = %.2f, lens = %.2f, decay = %.2f, in_size = %d, lr = %.5f, wd = %.5f, beta = %.2f, drop = %.2f' %
          (args.recurrent, args.te, args.permute,args.thresh, args.lens, args.decay, args.in_size, args.lr, args.wd, args.beta, args.dropout))
    print('Arch: {0}'.format(arch_name))

    # Data preprocessing
    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                               transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if args.permute:
        train_record_path = 'PSMNIST_TESNN_{0}_T{1}_decay{2}_beta{3}_arch{4}_lr{5}_drop{6}' \
            .format( args.te, (784 / args.in_size), args.decay, args.beta, arch_name, learning_rate,args.dropout)
        train_chk_pnt_path = 'PSMNIST_TESNN_{0}_T{1}_decay{2}_beta{3}_arch{4}_lr{5}_drop{6}.pt' \
            .format(args.te, (784 / args.in_size), args.decay, args.beta, arch_name, learning_rate,args.dropout)

        permute = torch.Tensor(np.random.permutation(int(784 / args.in_size)).astype(np.float64)).long()

        # Network initialization
        net = TESNN(in_size=1, hidden_size=args.fc, permute=True, permute_matrix=permute, te_type=args.te, beta=args.beta, decay=args.decay, dt_lim=args.dt_lim)


    else:
        train_record_path = 'SeqMNIST_TESNN_{0}_T{1}_decay{2}_beta{3}_arch{4}_lr{5}_drop{6}' \
            .format(args.te, (784 / args.in_size), args.decay, args.beta, arch_name, learning_rate,args.dropout)
        train_chk_pnt_path = 'SeqMNIST_TESNN_{0}_T{1}_decay{2}_beta{3}_arch{4}_lr{5}_drop{6}.pt' \
            .format(args.te, (784 / args.in_size), args.decay, args.beta, arch_name, learning_rate,args.dropout)

        # Network initialization
        net = TESNN(in_size=1, hidden_size=args.fc, permute=False, te_type=args.te, beta=args.beta, decay=args.decay, dt_lim=args.dt_lim)

    if args.recurrent:
        train_record_path = 'srnn_' + train_record_path
        train_chk_pnt_path = 'srnn_' + train_chk_pnt_path
    else:
        train_record_path = 'sfnn_' + train_record_path
        train_chk_pnt_path = 'sfnn_' + train_chk_pnt_path


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


    # Training
    def train(epoch):
        print('\nEpoch: %d' % (epoch + 1))
        global train_best_acc
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)  # [72, 500, 2, 32, 32]
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs.cpu(), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.cpu().max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % (len(trainloader) // 20) == 0:
                elapsed = time.time() - starts

                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f | Acc: %.5f%% (%d/%d)'
                      % (epoch + 1, num_epochs, batch_idx + 1, len(trainloader), train_loss / (batch_idx + 1),
                         100. * correct / total, correct, total))

        print('Train time past: ', elapsed, 's', 'Iter number:', epoch + 1)
        train_acc = 100. * correct / total
        loss_train_record.append(train_loss / (batch_idx + 1))
        train_acc_record.append(train_acc)
        if train_best_acc < train_acc:
            train_best_acc = train_acc


    def test(epoch):
        global test_best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)

                loss = criterion(outputs.cpu(), targets)
                test_loss += loss.item()
                _, predicted = outputs.cpu().max(1)
                total += targets.size(0)

                correct += predicted.eq(targets).sum().item()

                if (batch_idx + 1) % (len(testloader) // 1) == 0:
                    print(batch_idx + 1, '/', len(testloader), 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        acc = 100. * correct / total
        test_acc_record.append(acc)
        loss_test_record.append(test_loss / (batch_idx + 1))
        if test_best_acc < acc:
            test_best_acc = acc

            # Save Model
            print("Saving the model.")
            if not os.path.isdir(snn_ckp_dir):
                os.makedirs(snn_ckp_dir)
            state = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'acc': test_best_acc,
            }
            torch.save(state, snn_ckp_dir + train_chk_pnt_path)

    if not os.path.isdir(snn_ckp_dir):
        os.makedirs(snn_ckp_dir)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        starts = time.time()
        train(epoch)
        test(epoch)
        elapsed = time.time() - starts
        scheduler.step()
        print('Time past: ', elapsed, 's', 'Iter number:', epoch+1)

        training_record = {
            'learning_rate': args.lr,
            'algo': args.algo,
            'thresh': args.thresh,
            'lens': args.lens,
            'decay': args.decay,
            'architecture': args.fc,
            'loss_test_record': loss_test_record,
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


