# [Title] - Code Implementation

This repository contains the implementation of the experiments from the paper **"[Title]"**. The code is structured to reproduce all experiments efficiently using provided scripts.

## Dependencies
To run this code, you need the following dependencies:

- Python `3.9.13`
- NumPy `1.23.4`
- PyTorch `1.13.0`
- CUDA `12.2` 
- Librosa `0.8.0` (for audio processing in the GSC dataset)


## How to Reproduce Our Results

Each category of tasks is stored in a separate folder. Inside each folder, the script prefixed with `run_` can execute all relevant experiments.

```graphql
Project Root
 ├── GSC/  
 │    ├── run_gsc.py      # Script to run GSC experiments 
 ├── PTB/  
 │    ├── run_ptb.py      # Script to run PTB experiments 
 ├── SHD/  
 │    ├── run_shd.py      # Script to run SHD experiments 
 ├── Spiking_MNIST/      
 │    ├── run_smnist.py   # Script to run S-MNIST experiments
 │    ├── run_psmnist.py  # Script to run PS-MNIST experiments
 ├── Synthetic_tasks/      
 │    ├── run_interval.py # Script to run Interval Discrimination experiments
 │    ├── run_duration.py # Script to run Duration Discrimination experiments
 │    ├── run_syn.py      # Script to run Synchronization experiments
 │    ├── run_recall.py   # Script to run Delayed Recall experiments
 ├── README.md            # This file
 
```


### Example: Running Experiments on the S-MNIST Dataset

- Before running the experiments, you must specify the dataset path inside the `Spiking_MNIST/run_smnist.py` script by replacing all `/path` with your actual dataset location. 
  
  📌 *Note*: For scripts in `Synthetic_tasks`, no dataset path is required.


- Once the dataset path is correctly set, you can run all four experiments (TE-SFNN-R, TE-SFNN-N, TE-SRNN-R, TE-SRNN-N) sequentially using:

```sh
cd Spiking_MNIST
python run_smnist.py
```
- This script will sequentially execute all the experiments for the dataset.




