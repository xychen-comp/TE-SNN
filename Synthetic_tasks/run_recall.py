import os

### TE-SRNN-R
os.system(" python ./main_delayed_recll.py --te TE-R  --seq_len 50  --fc 280 280")
os.system(" python ./main_delayed_recll.py --te TE-R  --seq_len 100 --fc 280 280")
os.system(" python ./main_delayed_recll.py --te TE-R  --seq_len 200  --fc 280 280")
os.system(" python ./main_delayed_recll.py --te TE-R  --seq_len 400 --beta 0.0025  --fc 280 280")

### TE-SRNN-N
os.system(" python ./main_delayed_recll.py --te TE-N  --seq_len 50 --beta 0.0025 --fc 264 264")
os.system(" python ./main_delayed_recll.py --te TE-N  --seq_len 100 --beta 0.0025")
os.system(" python ./main_delayed_recll.py --te TE-N  --seq_len 200 --beta 0.0025 --fc 236 236")
os.system(" python ./main_delayed_recll.py --te TE-N  --seq_len 400 --beta 0.0025 --fc 200 200")

### LSTM
os.system(" python ./main_delayed_recll.py --te LSTM  --seq_len 50 --fc 114 114 --grad-clip 1e-1")
os.system(" python ./main_delayed_recll.py --te LSTM  --seq_len 100 --fc 114 114")
os.system(" python ./main_delayed_recll.py --te LSTM  --seq_len 200 --fc 114 114")
os.system(" python ./main_delayed_recll.py --te LSTM  --seq_len 400 --fc 114 114")

### SRNN
os.system(" python ./main_delayed_recll.py --te LIF  --seq_len 50  --fc 280 280")
os.system(" python ./main_delayed_recll.py --te LIF  --seq_len 100 --fc 280 280")
os.system(" python ./main_delayed_recll.py --te LIF  --seq_len 200  --fc 280 280")
os.system(" python ./main_delayed_recll.py --te LIF  --seq_len 400 --beta 0.0025  --fc 280 280")

### ASRNN
os.system(" python ./main_delayed_recll.py --te ALIF  --seq_len 50  --fc 280 280")
os.system(" python ./main_delayed_recll.py --te ALIF  --seq_len 100 --fc 280 280")
os.system(" python ./main_delayed_recll.py --te ALIF  --seq_len 200  --fc 280 280")
os.system(" python ./main_delayed_recll.py --te ALIF  --seq_len 400 --beta 0.0025  --fc 280 280")


