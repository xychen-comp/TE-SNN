import os

### TE-SRNN-R
os.system(" python ./main_synchronization.py --te TE-R --fc 84 84")

### TE-SRNN-N
os.system(" python ./main_synchronization.py --te TE-N")

### LSTM
os.system(" python ./main_synchronization.py --fc 35 35 --te LSTM --grad-clip 1e-1")

### SRNN
os.system(" python ./main_synchronization.py --te LIF --fc 84 84")

### ASRNN
os.system(" python ./main_synchronization.py --te ALIF --fc 84 84")

