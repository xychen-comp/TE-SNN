# Autonomous Driving Task  

This directory contains the implementation of the **autonomous driving task** experiments from the paper **"Temporal structure encoding for accurate and robust temporal processing in spiking neural networks"**.  

## Code Base  

Our implementation is **developed based on [F1TENTH Gym](https://f1tenth-gym.readthedocs.io/en/latest/installation.html) and the [f1 racetrack database](https://github.com/TUMFTM/racetrack-database)**. The necessary components are already included for ease of use. We provide their **official README files** within respective folders to offer further instructions on their usage and setup.

## Dependencies  

This task requires the following libraries:  

- **Python 3.8** (required for compatibility with Windows systems)
- **Stable-Baselines3** (for reinforcement learning framework)  
- **TensorBoard** (for real-time training visualization and performance monitoring)  
- **PyBullet 3.2.6** (for physics-based simulation and collision detection)

Additionally, dependencies required for **F1TENTH Gym** include:

- **Gym 0.19.0** (for reinforcement learning environment compatibility)  
- **NumPy (>=1.18.0, <=1.22.0)**  
- **Pillow (>=9.0.1)**  
- **SciPy (>=1.7.3)**  
- **Numba (>=0.55.2)**  
- **PyYAML (>=5.3.1)**  
- **Pyglet (<1.5)**  
- **PyOpenGL**  

### Installation  

1. **F1TENTH Dependencies**: Install the necessary dependencies required for **F1TENTH Gym**. For additional setup issues, please refer to the official `f110_gym/README.md` file.


2. **Stable-Baselines3**: Follow the official [Installation Guide](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
   
   📌 *Known Issue*: Installing Stable-Baselines3 may overwrite the existing PyTorch installation. After installation, ensure that you have the correct CUDA version of PyTorch by reinstalling it if necessary.


3. After installing the above dependencies, you can install the remaining required Python packages:

   ```sh
   pip install tensorboard pybullet==3.2.6
   ```

## Running the Experiments 

Once all dependencies are installed, follow these steps to run the autonomous driving task:  

1. **Run the autonomous driving experiment**
   ```sh
   cd Autonomous_driving
   python run_autonomous_driving.py
   ```
   
   - Experimental logs and trained models will be saved in the `logs/` directory for further analysis.
2. **Monitor the training process**   
   - Initial **TensorBoard** for real-time visualization:

   ```sh
   tensorboard --logdir=logs
   ```
 
