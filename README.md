# Track-MDP: Object Tracking with Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ray RLlib](https://img.shields.io/badge/Ray-RLlib-orange.svg)](https://docs.ray.io/en/latest/rllib/index.html)

A modular framework for training reinforcement learning agents to track moving objects in grid environments using sensor networks. This implementation provides a comprehensive solution for object tracking with sensor selection using A2C (Advantage Actor-Critic) algorithms.

## 🌟 Features

- **Grid-based Object Tracking**: Stochastic object movement in configurable NxN grid environments
- **Sensor Network Optimization**: Intelligent sensor activation to minimize energy consumption while maximizing tracking accuracy
- **Ray RLlib Integration**: Built on Ray RLlib for scalable distributed training
- **Real-time Visualization**: Interactive pygame-based visualization of tracking performance
- **Comparative Evaluation**: Built-in QMDP baseline comparison for performance analysis
- **Modular Architecture**: Clean separation of environment, training, evaluation, and visualization components
- **Flexible Configuration**: Easy parameter tuning for different scenarios and environments

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Performance Evaluation](#performance-evaluation)
- [Visualization](#visualization)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Install from Source

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Track-MDP-final.git
   cd Track-MDP-final
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv track_mdp_env
   source track_mdp_env/bin/activate  # On Windows: track_mdp_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```

### Quick Install

```bash
pip install track-mdp
```

## 🏃 Quick Start

### Basic Training

Train a tracking agent with default parameters:

```python
from src.training.trainer import train

# Train with visualization during evaluation
train(visualization_mode='evaluation')
```

### Basic Evaluation

Evaluate a trained model:

```python
from src.evaluation.evaluator import evaluate_policy
from src.core.environment import learning_grid_sarsa_0
import ray
from ray.rllib.algorithms.a2c import A2C

# Load trained model
ray.init()
algo = A2C.from_checkpoint("./agent_run194_a2c/checkpoint_001000")

# Create environment
qobj = learning_grid_sarsa_0(194, 10, 4, [0.15, 0.3, 0.45, 0.6], 6, 6, 1, 1)

# Evaluate
accuracy, sensors = evaluate_policy(algo, qobj.grid_env, num_episodes=100)
print(f"Tracking Accuracy: {accuracy:.4f}, Avg Sensors: {sensors:.2f}")
```

### Visualization

Visualize a trained policy:

```bash
python visualize.py --checkpoint ./agent_run194_a2c --episodes 5 --fps 10
```

## 📖 Usage

### Training a New Model

```python
# Basic training
from src.training.trainer import train

train(
    visualization_mode='evaluation',  # Show visualization during evaluation
    viz_config={
        'episodes': 5,                # Episodes per evaluation visualization
        'fps': 10,                   # Visualization frame rate
        'step_by_step': False        # Real-time vs step-by-step
    }
)
```

### Custom Environment Configuration

```python
from src.core.environment import learning_grid_sarsa_0

# Create custom environment
qobj = learning_grid_sarsa_0(
    run_number=999,              # Unique run identifier
    N=10,                        # Grid size (10x10)
    num_trans=4,                 # Number of transition types
    state_trans_cum_prob=[0.15, 0.3, 0.45, 0.6],  # Transition probabilities
    max_sensors=6,               # Maximum sensors
    max_sensors_null=6,          # Maximum sensors for null state
    time_limit=1,                # Time limit before missing state
    time_limit_max=1             # Maximum time limit
)
```

### Advanced Evaluation

```python
from src.evaluation.evaluator import evaluate_policy_detailed
from src.evaluation.visualizer import evaluate_and_visualize

# Detailed evaluation with statistics
summary = evaluate_policy_detailed(algo, environment, num_episodes=100)

# Combined evaluation and visualization
eval_summary, viz_summary = evaluate_and_visualize(
    algo, environment, 
    num_episodes=5,           # Visualization episodes
    evaluation_episodes=100   # Evaluation episodes
)
```

## 📁 Project Structure

```
Track-MDP-final/
├── src/
│   ├── core/                    # Core environment and gym wrapper
│   │   ├── environment.py       # Grid environment implementation
│   │   └── gym_wrapper.py       # Gymnasium environment wrapper
│   ├── training/                # Training modules
│   │   ├── trainer.py           # Main training script
│   │   └── monitor.py           # Training monitoring and visualization
│   ├── evaluation/              # Evaluation and testing
│   │   ├── evaluator.py         # Policy evaluation functions
│   │   └── visualizer.py        # Interactive visualization
│   ├── visualization/           # Rendering and display
│   │   └── renderer.py          # Pygame-based visualization
│   └── utils/                   # Utility functions
├── assets/                      # Image assets for visualization
│   ├── robot4.png               # Robot/object sprite
│   ├── antenna_on.png           # Active sensor sprite
│   └── antenna_off.png          # Inactive sensor sprite
├── examples/                    # Usage examples
├── visualize.py                 # Standalone visualization script
├── comparative_evaluation.py    # QMDP vs Track-MDP comparison
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
└── README.md                    # This file
```

## ⚙️ Configuration

### Environment Parameters

Configure environments using inline parameters when creating them:

```python
from src.core.environment import learning_grid_sarsa_0

# Grid environment configuration
grid_size = 6                    # Size of NxN grid  
n_sensors = 4                    # Number of sensors
sensor_probabilities = [0.15, 0.3, 0.45, 0.6]  # Detection probabilities
transition_noise = 1             # Movement randomness
reward = 1                       # Reward for successful tracking

# Create environment with parameters
qobj = learning_grid_sarsa_0(
    seed=42,
    n=n_sensors,
    dim=grid_size, 
    probabilities=sensor_probabilities,
    transition_noise=transition_noise,
    reward=reward,
    n_cols=grid_size,
    n_rows=grid_size
)

# Training configuration  
training_iterations = 2000       # Number of training iterations
learning_rate = 0.0001          # A2C learning rate
NUM_ROLLOUT_WORKERS = 4         # Parallel rollout workers
```

### Reward Structure

The environment uses the following reward structure:
- **Successful detection**: +1.0
- **Missed detection**: 0.0  
- **Sensor activation**: -0.16 per sensor
- **Missing state penalty**: Configurable

## 💡 Examples

### Example 1: Basic Training and Evaluation

```python
from src.training.trainer import train
from src.evaluation.evaluator import evaluate_policy
from src.core.environment import learning_grid_sarsa_0
import ray
from ray.rllib.algorithms.a2c import A2C

# Train a model
train(visualization_mode='none')  # Train without visualization

# Load and evaluate
ray.init()
algo = A2C.from_checkpoint("./agent_run194_a2c/checkpoint_002000")
qobj = learning_grid_sarsa_0(194, 10, 4, [0.15, 0.3, 0.45, 0.6], 6, 6, 1, 1)

accuracy, sensors = evaluate_policy(algo, qobj.grid_env, num_episodes=1000)
print(f"Final Performance - Accuracy: {accuracy:.4f}, Sensors: {sensors:.2f}")
```

### Example 2: Comparative Analysis

```python
from comparative_evaluation import main

# Run QMDP vs Track-MDP comparison
qmdp_results, track_mdp_results = main()

# Results automatically printed with detailed comparison
```

### Example 3: Real-time Training Monitoring

```python
from src.training.monitor import TrainingMonitor
from src.training.trainer import train

# Train with real-time visualization
viz_config = {
    'enabled': True,
    'frequency': 10,      # Visualize every 10 iterations
    'episodes': 3,        # 3 episodes per visualization
    'fps': 15            # 15 FPS visualization
}

train(visualization_mode='evaluation', viz_config=viz_config)
```

## 📊 Performance Evaluation

### Metrics

The framework provides comprehensive performance metrics:

- **Tracking Accuracy**: Proportion of time steps where object is successfully detected
- **Sensor Efficiency**: Average number of sensors activated per time step
- **Episode Length**: Average duration of tracking episodes
- **Total Reward**: Cumulative reward including tracking success and sensor costs
- **Cost per Detection**: Average sensor cost per successful object detection

### Baseline Comparison

Use the built-in QMDP baseline for performance comparison:

```bash
python comparative_evaluation.py
```

This will output a detailed comparison showing:
- Tracking accuracy improvements
- Sensor efficiency gains  
- Reward optimization
- Statistical significance tests

## 🎮 Visualization

### Interactive Visualization

The framework includes rich visualization capabilities:

- **Real-time rendering** of the grid environment
- **Sensor activation patterns** shown as colored overlays
- **Object movement tracking** with trajectory history
- **Performance statistics** displayed in real-time
- **Step-by-step mode** for detailed analysis

### Visualization Modes

1. **Evaluation Mode**: Visualize during training evaluation cycles
2. **Real-time Mode**: Continuous visualization during training
3. **Standalone Mode**: Visualize pre-trained models

### Controls

- **ESC**: Exit visualization
- **Space**: Pause/resume (in step-by-step mode)
- **Arrow Keys**: Manual stepping (in step-by-step mode)

## 📚 API Reference

### Core Classes

#### `grid_env`
Main environment class implementing object tracking dynamics.

```python
env = grid_env(N, num_trans, state_trans_cum_prob, max_sensors, 
               max_sensors_null, missing_state, time_limit)
```

#### `grid_environment` 
Gymnasium wrapper for RL algorithm integration.

```python
gym_env = grid_environment(env_config)
```

#### `TrackingRenderer`
Pygame-based visualization renderer.

```python
renderer = TrackingRenderer(grid_size=10, sq_pixels=40, fps=5)
```

### Key Functions

#### Training
- `train()`: Main training function with visualization options
- `save_environment()`: Save environment configuration

#### Evaluation  
- `evaluate_policy()`: Basic policy evaluation
- `evaluate_policy_detailed()`: Comprehensive evaluation with statistics
- `visualize_policy_interactive()`: Interactive visualization

#### Visualization
- `TrackingRenderer.render()`: Render single frame
- `TrackingRenderer.update_sensors()`: Update sensor display

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `python -m pytest`
5. **Submit a pull request**

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Track-MDP-final.git
cd Track-MDP-final

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting  
- **MyPy** for type checking
- **Pytest** for testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [Ray RLlib](https://docs.ray.io/en/latest/rllib/) for distributed reinforcement learning
- Visualization powered by [Pygame](https://www.pygame.org/)
- Environment design inspired by POMDP object tracking research

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Track-MDP-final/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Track-MDP-final/discussions)
- **Wiki**: [Project Wiki](https://github.com/yourusername/Track-MDP-final/wiki)

## 🔗 Related Projects

- [Ray RLlib](https://github.com/ray-project/ray)
- [OpenAI Gym](https://github.com/openai/gym)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

---

**Happy Tracking!** 🎯