# Self-Modeling Robot Arm: Learning and Adaptation

A NumPy-based simulation exploring how a neural network can learn a robot arm's movement and adapt its predictions after a change in the arm's structure.

## Overview

This project simulates a two-dimensional robotic arm with three joints. A multilayer perceptron (MLP) learns to predict the position of the arm's hand from its joint angles.

After the model learns the original arm, the simulation changes the length of one arm segment. The model then receives new observations and updates its predictions.

The project investigates two ways of selecting these new observations:

- **Random sampling:** selects joint configurations randomly.
- **Uncertainty-guided sampling:** selects configurations where an ensemble of neural networks disagrees most about the predicted hand position.

The objective is to compare how these strategies affect adaptation accuracy as new observations become available.

## Research Question

Can uncertainty-guided observation selection improve the adaptation of a learned robot-arm model compared with random observation selection?

## How It Works

1. **Simulate the arm:** Calculate the hand's position using forward kinematics.
2. **Learn the original movement:** Train a neural network to predict the hand's two-dimensional position from joint angles.
3. **Simulate structural change:** Reduce the length of the second arm segment.
4. **Collect observations:** Select new joint configurations using either random or uncertainty-guided sampling.
5. **Adapt the model:** Train the neural networks using the newly collected observations and their simulated ground-truth positions.
6. **Evaluate performance:** Measure prediction error on held-out test configurations and compare the sampling strategies.

## Model Architecture

The neural network is implemented directly with NumPy.

- **Input:** Three joint angles represented by their sine and cosine values, producing six input features.
- **Hidden layers:** Two fully connected layers with 64 units each.
- **Activation:** Hyperbolic tangent (`tanh`).
- **Output:** Two values representing the predicted hand position, `(x, y)`.
- **Training:** Manual backpropagation with an Adam-style parameter update.
- **Ensemble:** Five independently initialized neural networks whose predictions are combined to estimate the mean prediction and model disagreement.

## Observation Selection

### Random Sampling

New joint configurations are drawn randomly from the allowed angle range. Their corresponding hand positions are calculated using the simulated arm.

### Uncertainty-Guided Sampling

The ensemble evaluates candidate joint configurations. Configurations with the greatest disagreement among the networks are selected for new observations.

This disagreement is used as an uncertainty proxy; it is not a guarantee that the selected observations will produce the greatest improvement.

## Evaluation

The primary metric is root mean squared Euclidean position error on held-out joint configurations. Lower error indicates more accurate position predictions.

Experiments use multiple random seeds to examine variability across independent trials. Results are recorded at increasing numbers of newly collected observations.

The experiment runner saves configuration information, tabular results, a summary, and a plot for each experiment.

## Project Structure

```text
Self Learning Robotic Arm/
├── lab.py
├── Experiment.py
├── Visualize.py
├── requirements.txt
├── .gitignore
└── experiments/
    └── <experiment-output>/
        ├── config.json
        ├── results.csv
        ├── summary.md
        └── plot.png
```

`Experiment.py` and `Visualize.py` use capital letters in their filenames in this project.

## Getting Started

### Requirements

- Python 3.10 or a compatible Python version
- NumPy
- Other dependencies listed in `requirements.txt`

### Installation

Clone the repository:

```bash
git clone https://github.com/paddy-max/self-modeling-robot-arm.git
cd self-modeling-robot-arm
```

Create and activate a virtual environment on Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Run an Experiment

Display the available command-line options:

```powershell
python Experiment.py --help
```

Run the baseline comparison:

```powershell
python Experiment.py --name baseline --seeds 3
```

Change the candidate pool size for uncertainty-guided sampling:

```powershell
python Experiment.py --name pool-100 --seeds 3 --pool 100
python Experiment.py --name pool-1000 --seeds 3 --pool 1000
```

Each experiment creates a separate output directory under `experiments/`.

### Visualize the Simulation

Check the available options:

```powershell
python Visualize.py --help
```

Run the visualization:

```powershell
python Visualize.py
```

For a shorter run:

```powershell
python Visualize.py --quick
```

To save a GIF:

```powershell
python Visualize.py --save demo.gif
```

## Preliminary Findings

In an initial comparison using three random seeds, both strategies reduced prediction error as additional observations were collected.

At 80 observations, the reported mean error was:

| Strategy                    | Mean error | Standard deviation |
| --------------------------- | ---------: | -----------------: |
| Random sampling             |      0.104 |              0.005 |
| Uncertainty-guided sampling |      0.094 |              0.010 |

These preliminary results suggest that uncertainty-guided sampling may help at some stages of adaptation. Random sampling performed better at several earlier stages in this experiment.

The result is not conclusive: more trials and parameter comparisons are needed before determining whether one strategy is consistently better.

## Current Limitations

- The arm is simulated; no physical robot is controlled.
- The ground-truth hand positions are calculated by the simulation.
- Ensemble disagreement is used as an uncertainty proxy and may not always identify the most informative observations.
- The current experiment runner saves experiment results but does not persist trained model weights for later reuse.
- Results depend on model initialization, sampling, training settings, and the simulated arm configuration.
- This project investigates model learning and adaptation; it does not demonstrate self-awareness or consciousness.

## Future Work

- Compare sampling strategies across more random seeds.
- Investigate the effect of candidate pool size and the severity of structural change.
- Add model checkpoint saving and loading.
- Improve experiment reproducibility and documentation.
- Explore more realistic robotic dynamics and, eventually, physical-robot evaluation.

## Reproducibility

Each experiment records its configuration and results in its output directory. Use the saved configuration and source code when reproducing a run, and compare strategies at the same observation counts.

## License

No license has been specified yet. Reuse and redistribution permissions will be defined when a license is added.
