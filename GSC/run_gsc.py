import os


### TE-SRNN-R
os.system(" python ./GSC_TESNN_main.py --recurrent --fc 318 318 --lr 5e-4 --te TE-R --beta 0.08 --data_path /path")

### TE-SRNN-N
os.system(" python ./GSC_TESNN_main.py --recurrent --fc 256 256 --lr 1.5e-3 --te TE-N --beta 0.08 --data_path /path")

