import os

### TE-SRNN-R
os.system(" python ./main_interval.py --te TE-R --fc 84 84 --lr 1e-3")

### TE-SRNN-N
os.system(" python ./main_interval.py --te TE-N")

### LSTM
os.system(" python ./main_interval.py --te LSTM --fc 35 35  --grad-clip 0.5")

### SRNN
os.system(" python ./main_interval.py --te LIF --fc 84 84")

### ASRNN
os.system(" python ./main_interval.py --te ALIF --fc 84 84")



