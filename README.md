# TE-SNN 

This repository contains the implementation of the experiments from the paper **"Temporal structure encoding for accurate and robust temporal processing in spiking neural networks"**. The code is structured to reproduce all experiments efficiently using provided scripts.

## Dependencies
To run this code, you need the following dependencies:

- Python `3.9.13`
- NumPy `1.23.4`
- PyTorch `1.13.0`
- CUDA `11.6` 
- Librosa `0.8.0` (for audio processing in the GSC dataset)

📌 *Note*: The autonomous driving task requires additional dependencies, which are listed in `Autonomous_driving/README.md`.

## Project Structure

Each category of tasks is stored in a separate folder.

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
 ├── Autonomous_driving/  
 │    ├── run_autonomous_driving.py   # Script to run autonomous driving experiments
 │    ├── README.md       # Instructions for environment configuration and running autonomous driving experiments  
 ├── README.md            # This file
 
```

## Runing Experiments
### Runing Standard Benchmarks 

The following method applies to all tasks except the autonomous driving task.
1. **Set the dataset path:** 

   Before running, update the corresponding script (e.g., `Spiking_MNIST/run_smnist.py`) by replacing all `/path` with your actual dataset location. 
  
    📌 *Note*: For scripts in `Synthetic_tasks`, no dataset path is required.


2. **Execute the experiment (e.g., S-MNIST):**

   Once the dataset path is correctly set, you can run all four experiments (TE-SFNN-R, TE-SFNN-N, TE-SRNN-R, TE-SRNN-N) sequentially using:

    ```sh
    cd Spiking_MNIST
    python run_smnist.py
    ```
    This approach applies to **GSC, PTB, SHD, Spiking_MNIST, and Synthetic_tasks**. 

### Running the Autonomous Driving Task

The autonomous driving task has unique dependencies and setup requirements. Please refer to `Autonomous_driving/README.md` for detailed instructions on installing the necessary packages and running the experiment.


