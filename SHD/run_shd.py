import os



### TE-SFNN-R
os.system(" python ./SHD_TESNN_main.py --lr 5e-3 --fc 128 128 --te TE-R --data_path /path")

### TE-SFNN-N
os.system(" python ./SHD_TESNN_main.py --lr 5e-3 --fc 88 88 --te TE-N --data_path /path")

### TE-SRNN-R
os.system(" python ./SHD_TESNN_main.py --lr 1.5e-3 --fc 128 128 --te TE-R --recurrent --data_path /path")

### TE-SRNN-N
os.system(" python ./SHD_TESNN_main.py --lr 1e-3 --fc 100 100 --te TE-N  --recurrent --data_path /path")