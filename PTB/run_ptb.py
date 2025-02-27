import os


# TE-SFNN-R
os.system("python ./PTB_TESNN_main.py --name _TE_SFNN_R_PTB --te TE-R --data_path /path")

# # TE-SRNN-R
os.system("python ./PTB_TESNN_main.py --name _TE_SRNN_R_PTB --recurrent --decay 0.25 --beta 0.15  --te TE-R --data_path /path")

# TE-SFNN-N
os.system(" python ./PTB_TESNN_main.py --name _TE_SFNN_N_PTB --te TE-N --data_path /path")

#TE-SRNN-N
os.system(" python ./PTB_TESNN_main.py --name _tTE_SRNN_N_PTB --recurrent --decay 0.2 --te TE-N --data_path /path")

