import argparse
"""
GSC
"""
parser = argparse.ArgumentParser(description='model')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N',
					help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=0, type=float,
					metavar='wd', help='weight decay', dest='wd')
parser.add_argument('--algo', default='model', type=str, metavar='N',
					help='algorithmn for learning')
parser.add_argument('--te', default='LIF', type=str, choices=['TE-N', 'TE-R', 'LIF'],
					help='algorithmn for learning')
parser.add_argument('--dropout','--drop', default=0.2, type=float, metavar='N',
					help='Dropout rate')
parser.add_argument('--beta', default=0.08, type=float, metavar='N',
					help='Decay factor of V')
parser.add_argument('--thresh', default=0.3, type=float, metavar='N',
					help='threshold of the neuron model')
parser.add_argument('--lens', default=0.2, type=float, metavar='N',
					help='lens of surrogate function')
parser.add_argument('--decay', default=0.5, type=float, metavar='N',
					help='decay of the neuron model')
parser.add_argument('--fc', nargs= '+', default=[296, 296], type=int, metavar='N',
					help='model architecture')
parser.add_argument('--cuda', default='0', type=str, help='gpu index')
parser.add_argument('--time-window', default=101, type=int, help='')
parser.add_argument('--recurrent', default=False, action='store_true',
					help='Feedforward or recurrent')
parser.add_argument('--data_path', default='/path', type=str, metavar='PATH',
					help='path to load dataset')

args = parser.parse_args()