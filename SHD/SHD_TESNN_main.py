from __future__ import print_function
import os,time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import h5py
import torch
import torch.nn as nn
import numpy as np
from utils.lib import dump_json, set_seed, count_para
from torch.optim.lr_scheduler import StepLR
from model.spiking_seqmnist_model import TESNN
from model.Hyperparameters import args
from tqdm import tqdm
current_dir = os.path.dirname(os.getcwd())

set_seed(1111)

def runTrain(epoch, train_ldr, optimizer, model, evaluator, args=None, encoder=None):
    loss_record = []
    predict_tot = []
    label_tot = []
    model.train()
    start_time = time.time()
    for idx, (ptns, labels) in enumerate(train_ldr):
        ptns, labels = ptns.to(device), labels.to(device).long()
        if encoder is not None:
            ptns = encoder(ptns)
        optimizer.zero_grad()

        B,length,frame=ptns.size()
        input_bin=torch.zeros(B,length,140, device=device)
        n_bin = frame // 140
        for i in range(140):
            input_bin[:, :, i ] = ptns[:,:, n_bin * i: n_bin * (i + 1)].sum(axis=2)
        #[N,T,H]
        output = model(input_bin.permute(0,2,1))  #[N,cls]
        loss = evaluator(output, labels)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        predict = torch.argmax(output, axis=1)
        # record results
        loss_record.append(loss.detach().cpu())
        predict_tot.append(predict)
        label_tot.append(labels)
        if (idx + 1) % 100 == 0:
            print('\nEpoch [%d/%d], Step [%d/%d], Loss: %.5f'
                  % (epoch, args.epochs + start_epoch, idx + 1, len(train_ldr) // args.batch_size,
                     loss_record[-1] / args.batch_size))
            print('Time elasped:', time.time() - start_time)
    predict_tot = torch.cat(predict_tot)
    label_tot = torch.cat(label_tot)
    train_acc = torch.mean((predict_tot == label_tot).float())
    train_loss = torch.tensor(loss_record).sum() / len(label_tot)
    return train_acc, train_loss
def runTest(val_ldr, model, evaluator, args=None, encoder=None):
    model.eval()
    with torch.no_grad():
        predict_tot = {}
        label_tot = []
        loss_record = []
        key = 'ann' if encoder is None else 'snn'
        for idx, (ptns, labels) in enumerate(val_ldr):
            # ptns: batch_size x num_channels x T x nNeu ==> batch_size x T x (nNeu*num_channels)
            ptns, labels = ptns.to(device), labels.to(device).long()
            if encoder is not None:
                ptns = encoder(ptns)
            B, length, frame = ptns.size()
            input_bin = torch.zeros(B, length, 140, device=device)
            n_bin = frame // 140
            for i in range(140):
                input_bin[:, :, i] = ptns[:, :, n_bin * i: n_bin * (i + 1)].sum(axis=2)
            # [N,T,H]
            output = model(input_bin.permute(0, 2, 1))  # [N,cls]
            if isinstance(output, dict):
                for t in output.keys():
                    if t not in predict_tot.keys():
                        predict_tot[t] = []
                    predict = torch.argmax(output[t], axis=1)
                    predict_tot[t].append(predict)
                loss = evaluator(output[encoder.nb_steps], labels)

            else:
                if key not in predict_tot.keys():
                    predict_tot[key] = []
                loss = evaluator(output, labels)
                # snn.clamp()
                predict = torch.argmax(output, axis=1)
                predict_tot[key].append(predict)
            loss_record.append(loss)
            label_tot.append(labels)

        label_tot = torch.cat(label_tot)
        val_loss = torch.tensor(loss_record).sum() / len(label_tot)
        if 'ann' not in predict_tot.keys() and 'snn' not in predict_tot.keys():
            val_acc = {}
            for t in predict_tot.keys():
                val_acc[t] = torch.mean((torch.cat(predict_tot[t]) == label_tot).float())

        else:
            predict_tot = torch.cat(predict_tot[key])
            val_acc = torch.mean((predict_tot == label_tot).float())
        return val_acc, val_loss


class SpikeIterator:
    def __init__(self, X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        # self.max_time = max_time
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=np.int64)
        self.num_samples = len(self.labels_)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.labels_))
        # compute discrete firing times
        self.firing_times = X['times']
        self.units_fired = X['units']
        self.time_bins = np.linspace(0, max_time, num=nb_steps)
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.counter < self.number_of_batches:
            batch_index = self.sample_index[
                          self.batch_size * self.counter:min(self.batch_size * (self.counter + 1), self.num_samples)]
            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins)
                units = self.units_fired[idx]
                batch = [bc for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

            X_batch = torch.sparse.FloatTensor(i, v, torch.Size(
                [len(batch_index), self.nb_steps, self.nb_units])).to_dense().to(
                device)
            y_batch = torch.tensor(self.labels_[batch_index], device=device)
            self.counter += 1
            return X_batch.to(device=device), y_batch.to(device=device)

        else:
            raise StopIteration


def sparse_data_generator_from_hdf5_spikes(X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True):
    """ This generator takes a spike dataset and generates spiking network input as sparse tensors.

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    labels_ = np.array(y, dtype=np.int)
    number_of_batches = len(labels_) // batch_size
    sample_index = np.arange(len(labels_))

    # compute discrete firing times
    firing_times = X['times']
    units_fired = X['units']

    time_bins = np.linspace(0, max_time, num=nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index], device=device)

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1

def getData():
    root_path = args.data_path + '/SHD'
    train_file = h5py.File(os.path.join(root_path, 'shd_train.h5'), 'r')
    test_file = h5py.File(os.path.join(root_path, 'shd_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":

    # CUDA configuration
    if torch.cuda.is_available():
        device = 'cuda'
        print('GPU is available')
    else:
        device = 'cpu'
        print('GPU is not available')

    home_dir = current_dir  # Relative path
    snn_ckp_dir = os.path.join(home_dir, 'exp/SHD/checkpoint/')
    snn_rec_dir = os.path.join(home_dir, 'exp/SHD/record/')
    print(snn_ckp_dir)

    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    arch_name = '-'.join([str(s) for s in args.fc])

    print('Recurrent architecture is %s, algo is %s, thresh = %.2f, dropout = %.4f, decay = %.2f, lr = %.5f, wd = %.5f, beta = %.2f' %
          (args.recurrent, args.te, args.thresh, args.dropout, args.decay, args.lr, args.wd, args.beta))
    print('Arch: {0}'.format(arch_name))

    globals().update(vars(args))
    (x_train, y_train), (x_test, y_test) = getData()
    train_ldr = SpikeIterator(x_train, y_train, args.batch_size, 250, 700, max_time=1.4, shuffle=True)
    val_ldr = SpikeIterator(x_test, y_test, args.batch_size, 250, 700, max_time=1.4, shuffle=False)

    if args.recurrent:
        train_record_path = 'SHD_srnn_TESNN_{0}_2dropout{1}_decay{2}_beta{3}_arch{4}_lr{5}.txt' \
            .format(args.te, args.dropout, args.decay, args.beta, arch_name, learning_rate)
        train_chk_pnt_path = 'SHD_srnn_TESNN_{0}_2dropout{1}_decay{2}_beta{3}_arch{4}_lr{5}.pt' \
            .format(args.te, args.dropout, args.decay, args.beta, arch_name, learning_rate)
    else:
        train_record_path = 'SHD_sfnn_TESNN_{0}_2dropout{1}_decay{2}_beta{3}_arch{4}_lr{5}.txt' \
            .format(args.te, args.dropout, args.decay, args.beta, arch_name, learning_rate)
        train_chk_pnt_path = 'SHD_sfnn_TESNN_{0}_2dropout{1}_decay{2}_beta{3}_arch{4}_lr{5}.pt' \
            .format(args.te, args.dropout, args.decay, args.beta, arch_name, learning_rate)


    # Network initialization
    net = TESNN(in_size=140, output_size=20, dropout=args.dropout)
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
        train_ldr.reset()
        train_acc, train_loss = runTrain(epoch, train_ldr, optimizer, net, criterion, args=args)
        train_ldr.reset()
        train_acc_, train_loss_ = runTest(train_ldr, net, criterion, args=args)
        scheduler.step()
        val_ldr.reset()
        val_acc, val_loss = runTest(val_ldr, net, criterion, args=args)
        print('validation record:', val_loss, val_acc)
        if (val_acc > test_best_acc):
            test_best_acc = val_acc

            # Save Model
            print("Saving the model.")
            if not os.path.isdir(snn_ckp_dir):
                os.makedirs(snn_ckp_dir)
            state = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'acc': test_best_acc,
            }
            torch.save(state, snn_ckp_dir + train_chk_pnt_path)

        print('Epoch %d: train acc %.5f, train acc with spike %.5f, test acc %.5f ' % (
            epoch, train_acc, train_acc_, val_acc))

        loss_train_record.append(train_loss.item())
        train_acc_record.append(train_acc.item())
        test_acc_record.append(val_acc.item())
        loss_test_record.append(val_loss.item())
        if train_best_acc < train_acc:
            train_best_acc = train_acc

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
            'loss_test_record': loss_test_record,
            'loss_train_record': loss_train_record,
            'test_acc_record': test_acc_record,
            'train_acc_record': train_acc_record,
            'train_best_acc': train_best_acc.item(),
            'test_best_acc': test_best_acc.item(),
        }
        dump_json(training_record, snn_rec_dir, train_record_path)

    print(" Best Train Acc: ", train_best_acc)
    print(" Best Test Acc: ", test_best_acc)
    print('=====================End of trail=============================\n\n')

