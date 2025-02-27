from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn.functional as F
from data.data_interval import data_generator
from utils.lib import dump_json, set_seed, count_para
from torch.optim.lr_scheduler import StepLR
from model.spiking_seqmnist_model import TESNN
from model.Hyperparameters_interval import args
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
    snn_ckp_dir = os.path.join(home_dir, 'exp/Interval/checkpoint/')
    snn_rec_dir = os.path.join(home_dir, 'exp/Interval/record/')

    seq_length = args.seq_len
    max_duration = args.max_duration
    batch_size=args.batch_size
    epochs=args.epochs
    lr=args.lr
    X_train, Y_train = data_generator(50000, seq_length)  #[N,1,Seq_len],[N,2,Seq_len]
    X_test, Y_test = data_generator(1000, seq_length)
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
        net = lstm(INPUT_SIZE=2, LAYERS=len(args.fc), HIDDEN_SIZE=args.fc[0])
    elif args.te == 'ALIF':
        train_record_path = 'ALIF_seq{0}_co{1}_lens{2}_arch{3}_lr{4}' \
            .format(args.seq_len, args.beta, args.lens, arch_name, learning_rate)
        net = RNN_custom(time_window=seq_length, input_size=2, hidden_dims=args.fc)
    else:
        train_record_path = 'TESNN_{0}_seq{1}_co{2}_lens{3}_arch{4}_lr{5}' \
            .format(args.te,args.seq_len, args.beta, args.lens, arch_name, learning_rate)
        net = TESNN(in_size=2, time_window=seq_length, hidden_size=args.fc, te_type=args.te, beta=args.beta, decay=args.decay)

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

    best_loss=1000
    loss_train_record = []  # list([])
    loss_test_record = []  # list([])
    MAE_record = []
    # Training
    def train(epoch):
        net.train()
        batch_idx = 1
        total_loss = 0
        total_MAE = 0
        for i in range(0, X_train.size(0), batch_size):
            if i + batch_size > X_train.size(0):
                x, y = X_train[i:], Y_train[i:]
            else:
                x, y = X_train[i:(i + batch_size)], Y_train[i:(i + batch_size)]  # [N,2,200], [N,1]
            optimizer.zero_grad()
            output = net(x,'interval')
            MAE = (output - y).abs().mean()
            loss = F.mse_loss(output, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            batch_idx += 1
            total_loss += loss.item()
            total_MAE += MAE.item()

            if batch_idx % 20 == 0:
                cur_loss = total_loss / 20
                processed = min(i + batch_size, X_train.size(0))
                print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}\tMAE:{:.4f}'.format(
                    epoch, processed, X_train.size(0), 100. * processed / X_train.size(0), lr, cur_loss, total_MAE / 20))
                loss_train_record.append(total_loss / 20)
                MAE_record.append(total_MAE / 20)
                total_loss = 0
                total_MAE = 0


    def evaluate(epoch):
        global best_loss
        net.eval()
        with torch.no_grad():
            output = net(X_test,'interval')
            test_loss = F.mse_loss(output, Y_test)
            print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))

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
            'MAE_record': MAE_record
        }
        dump_json(training_record, snn_rec_dir, train_record_path)