import os


os.system(" python ./main_interval.py --te LIF --fc 84 84 --seed 1111")
os.system(" python ./main_interval.py --te LIF --fc 84 84 --seed 2222")
os.system(" python ./main_interval.py --te LIF --fc 84 84 --seed 3333")
#
# os.system(" python ./main_interval.py --te TE-R --fc 84 84 --seed 1111")
os.system(" python ./main_interval.py --te TE-R --fc 84 84 --seed 2222")
os.system(" python ./main_interval.py --te TE-R --fc 84 84 --seed 3333")

os.system(" python ./main_interval.py --te ALIF --fc 84 84 --seed 1111")
os.system(" python ./main_interval.py --te ALIF --fc 84 84 --seed 2222")
os.system(" python ./main_interval.py --te ALIF --fc 84 84 --seed 3333")
#
# os.system(" python ./main_interval.py --te LSTM --fc 35 35 --seed 1111")
# os.system(" python ./main_interval.py --te LSTM --fc 35 35 --seed 2222")
# os.system(" python ./main_interval.py --te LSTM --fc 35 35 --seed 3333")

# os.system(" python ./main_interval.py --te TE-N --seed 1111")
os.system(" python ./main_interval.py --te TE-N --seed 2222")
os.system(" python ./main_interval.py --te TE-N --seed 3333")