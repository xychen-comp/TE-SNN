import os

# os.system(" python ./main_synchronization.py --fc 35 35 --te LSTM --grad-clip 1e-1 --seed 1111")
# os.system(" python ./main_synchronization.py --fc 35 35 --te LSTM --grad-clip 1e-1 --seed 2222")
# os.system(" python ./main_synchronization.py --fc 35 35 --te LSTM --grad-clip 1e-1 --seed 3333")

os.system(" python ./main_synchronization.py --te TE-R --fc 84 84 --seed 1111")
os.system(" python ./main_synchronization.py --te TE-R --fc 84 84 --seed 2222")
os.system(" python ./main_synchronization.py --te TE-R --fc 84 84 --seed 3333")


os.system(" python ./main_synchronization.py --te LIF --fc 84 84 --seed 1111")
os.system(" python ./main_synchronization.py --te LIF --fc 84 84 --seed 2222")
os.system(" python ./main_synchronization.py --te LIF --fc 84 84 --seed 3333")

os.system(" python ./main_synchronization.py --te ALIF --fc 84 84 --seed 1111")
os.system(" python ./main_synchronization.py --te ALIF --fc 84 84 --seed 2222")
os.system(" python ./main_synchronization.py --te ALIF --fc 84 84 --seed 3333")

os.system(" python ./main_synchronization.py --te TE-N --seed 1111")
os.system(" python ./main_synchronization.py --te TE-N --seed 2222")
os.system(" python ./main_synchronization.py --te TE-N --seed 3333")