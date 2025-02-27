import argparse
"""
SHD
"""
parser = argparse.ArgumentParser(description='model')

parser.add_argument('--epochs', default=150, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=40, type=int,metavar='N',
					help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=0, type=float,
					metavar='wd', help='weight decay', dest='wd')
parser.add_argument('--algo', default='model', type=str, metavar='N',
					help='algorithmn for learning')
parser.add_argument('--te', default='TE-N', type=str, choices=['TE-N', 'TE-R','LIF'],
					help='algorithmn for learning')
parser.add_argument('--dropout', default=0.3, type=float, metavar='N',
					help='Dropout rate')
parser.add_argument('--recurrent', default=False, action='store_true',
					help='Feedforward or recurrent')
parser.add_argument('--beta', default=0.02, type=float, metavar='N',
					help='Decay factor of V')
parser.add_argument('--thresh', default=0.3, type=float, metavar='N',
					help='threshold of the neuron model')
parser.add_argument('--lens', default=0.2, type=float, metavar='N',
					help='lens of surrogate function')
parser.add_argument('--decay', default=0.5, type=float, metavar='N',
					help='decay of the neuron model')
parser.add_argument('--grad-clip', type=float, default=0.0)
parser.add_argument('--in_size', default=1, type=int, metavar='N',
					help='model input size')
parser.add_argument('--out_size', default=20, type=int, metavar='N',
					help='model output size')
parser.add_argument('--fc', nargs= '+', default=[128, 128], type=int, metavar='N',
					help='model architecture')
parser.add_argument('--data_path', default='/path', type=str, metavar='PATH',
					help='path to load dataset')
parser.add_argument('--chkp_path', default='', type=str, metavar='PATH',
					help='path to save the training model (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
					help='path to save the training record (default: none)')
args = parser.parse_args()

