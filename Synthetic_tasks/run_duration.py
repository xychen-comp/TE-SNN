import os

### TE-SRNN-R
os.system(" python ./main_duration.py --te TE-R --fc 84 84")

### TE-SRNN-N
os.system(" python ./main_duration.py --te TE-N")

### LSTM
os.system(" python ./main_duration.py --te LSTM --fc 35 35")

### SRNN
os.system(" python ./main_duration.py --te LIF --fc 84 84")

### ASRNN
os.system(" python ./main_duration.py --te ALIF --fc 84 84")




