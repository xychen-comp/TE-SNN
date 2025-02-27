import os

### TE-SFNN-R
os.system(" python ./SMNIST_TESNN_main.py --permute --lr 5e-3 --te TE-R --dropout 0.05 --fc 200 200 200 --data_path /path")

### TE-SFNN-N
os.system(" python ./SMNIST_TESNN_main.py --permute --lr 5e-3 --te TE-N --dropout 0.2 --fc 88 88 88 --data_path /path")

### TE-SRNN-R
os.system(" python ./SMNIST_TESNN_main.py --recurrent --permute --lr 1.5e-3 --te TE-R --dropout 0.2 --fc 224 224 224 --data_path /path")

### TE-SRNN-N
os.system(" python ./SMNIST_TESNN_main.py --recurrent --permute --lr 1.5e-3 --te TE-N --dropout 0.2 --fc 128 128 128 --data_path /path")



