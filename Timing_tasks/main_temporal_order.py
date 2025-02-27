from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
from data.data_order import data_generator
from utils.lib import dump_json, set_seed, count_para
from torch.optim.lr_scheduler import StepLR
from model.spiking_seqmnist_model import TESNN
from model.Hyperparameters_order import args
from model.ALIF import RNN_custom
from model.lstm import lstm

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

    # Data preprocessing
    home_dir = current_dir  # relative path
    snn_ckp_dir = os.path.join(home_dir, 'exp/Temporal_Order/checkpoint/')
    snn_rec_dir = os.path.join(home_dir, 'exp/Temporal_Order/record/')

    seq_length = args.seq_len
    max_duration = args.max_duration
    batch_size=args.batch_size
    epochs=args.epochs
    lr=args.lr
    X_train, Y_train = data_generator(seq_length-20, 10, 50000,encode=True)
    X_test, Y_test = data_generator(seq_length-20, 10, 1000,encode=True)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    arch_name = '-'.join([str(s) for s in args.fc])


    circle=1.

    print('Sequence length is %.1f,algo is %s, thresh = %.2f, lens = %.2f, decay = %.2f, in_size = %d, lr = %.5f, co2 = %.2f' %
          (args.seq_len, args.te, args.thresh, args.lens, args.decay, args.in_size, args.lr,  args.beta))
    print('Arch: {0}'.format(arch_name))

    if args.te == 'LSTM':
        train_record_path = 'LSTM_seq{0}_co{1}_lens{2}_arch{3}_lr{4}' \
            .format(args.seq_len, args.beta, args.lens, arch_name, learning_rate)
        net = lstm(INPUT_SIZE=9, OUT_SIZE=10,LAYERS=len(args.fc), HIDDEN_SIZE=args.fc[0])
    elif args.te == 'ALIF':
        train_record_path = 'ALIF_seq{0}_co{1}_lens{2}_arch{3}_lr{4}' \
            .format(args.seq_len, args.beta, args.lens, arch_name, learning_rate)
        net = RNN_custom(time_window=seq_length, input_size=9, output_size=10, hidden_dims=args.fc)
    else:
        train_record_path = 'TESNN_{0}_seq{1}_co{2}_lens{3}_arch{4}_lr{5}' \
            .format(args.te,args.seq_len, args.beta, args.lens, arch_name, learning_rate)
        net = TESNN(in_size=9, time_window=seq_length, out_size=10, hidden_size=args.fc, te_type=args.te, beta=args.beta, decay=args.decay)

    train_record_path = train_record_path + '_seed_' + str(args.seed)
    train_chk_pnt_path = train_record_path

    net = net.to(device)
    print(net)
    para=count_para(net)
    print(net,'para:',para)
    # pretrain = torch.load(snn_ckp_dir + 'SRNN_v2_in8_T784.0_decay0.5_thr0.3_lens0.2_arch64-256-256_lr0.0002.pt')
    #
    # net.load_state_dict(pretrain['model_state_dict'], strict=False)

    # n = count_parameters(net)
    # print("Number of parameters: %s" % n)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # define weight update method
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    best_loss=1000
    loss_train_record = []  # list([])
    loss_test_record = []  # list([])
    acc_train_record = []
    acc_test_record = []
    acc_decode_record = []
    # Training
    def train(epoch):
        correct = 0
        counter = 0
        correct_decode = 0
        net.train()
        batch_idx = 1
        total_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            if i + batch_size > X_train.size(0):
                x, y = X_train[i:], Y_train[i:]
            else:
                x, y = X_train[i:(i + batch_size)], Y_train[i:(i + batch_size)]
            optimizer.zero_grad()
            output = net(x,'order')
            output = output.transpose(1,2)
            loss = criterion(output.reshape(-1, 10), y.view(-1))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            batch_idx += 1
            total_loss += loss.item()
            pred = output[:,-10:,:].data.max(2, keepdim=False)[1] # only count the accuracy of the last 10 timesteps
            correct += pred.eq(y[:,-10:].data).cpu().sum()
            counter += pred.view(-1).size(0)


            if batch_idx % 20 == 0:
                avg_loss = total_loss / 20
                print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | '
                      'loss {:5.8f} | accuracy {:5.4f}'.format(
                    ep, batch_idx, 50000 // batch_size + 1, args.lr,
                    avg_loss, 100. * correct / counter))
                loss_train_record.append(avg_loss)
                acc_train_record.append((100. * correct / counter).item())
                total_loss = 0
                correct = 0
                counter = 0

    def evaluate(epoch):
        global best_loss
        net.eval()
        with torch.no_grad():
            output = net(X_test,'order')
            test_loss = criterion(output.transpose(1,2).reshape(-1, 10), Y_test.view(-1))
            pred = output[...,-10:].data.max(1, keepdim=False)[1] # only count the accuracy of the last 10 timesteps
            acc = pred.eq(Y_test[:,-10:].data).cpu().sum() / (output.size(0) * 10.)

            print('\nTest set: Average loss: {:.6f}, Accuracy: {:5.4f}\n'.format(test_loss.item(), 100. * acc.item()))

            if test_loss < best_loss:
                if not os.path.isdir(snn_ckp_dir):
                    os.makedirs(snn_ckp_dir)
                best_loss=test_loss
                state = {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                }
                torch.save(state, snn_ckp_dir + train_chk_pnt_path)
                print('Saving model')
            loss_test_record.append(test_loss.item())
            acc_test_record.append(100. * acc.item())
            return test_loss.item()



    for ep in range(1, epochs + 1):
        train(ep)
        loss = evaluate(ep)
        scheduler.step()
        if not os.path.isdir(snn_ckp_dir):
            os.makedirs(snn_ckp_dir)

        training_record = {
            'learning_rate': args.lr,
            'algo': args.algo,
            'thresh': args.thresh,
            'lens': args.lens,
            'decay': args.decay,
            'architecture': args.fc,
            'loss_test_record': loss_test_record,
            'loss_train_record': loss_train_record,
            'acc_train_record': acc_train_record,
            'acc_test_record': acc_test_record

        }
        dump_json(training_record, snn_rec_dir, train_record_path)