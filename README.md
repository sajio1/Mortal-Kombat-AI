# Mortal Kombat II Reinforcement Learning AI

This project trains an AI to play *Mortal Kombat II* (Genesis) using Reinforcement Learning with the *Stable-Baselines3* framework. The AI utilizes Recurrent Proximal Policy Optimization (PPO with LSTM) to enhance gameplay through memory and sequential decision-making.

## Features

- **Custom OpenAI Gym Environment**: Wrapper for *Mortal Kombat II* using `gym-retro`.
- **Recurrent PPO (LSTM)**: Leverages past game states for better decision-making.
- **Parallel Training**: Utilizes `SubprocVecEnv` for efficient training across multiple environments.
- **Preprocessing & Reward Shaping**: Custom frame processing and reward functions tailored for fighting gameplay.

## Installation

### Requirements
Ensure you have Python 3.8+ and install the dependencies:

```bash
pip install -r requirements.txt
```

You'll also need `gym-retro` configured with *Mortal Kombat II* ROM.

## Usage

### Train the AI
Run the following command to start training:

```bash
python main.py --train
```

The model will be trained using Recurrent PPO and saved as `mortal_kombat_ppo_lstm`.

### Run the Trained AI
After training, you can test the AI using:

```bash
python main.py --run
```
This will load the trained model and have the AI play a match.

## Code Overview

### `environment.py`
Defines a custom Gym environment for *Mortal Kombat II*:

- **Frame Preprocessing**: Converts frames to grayscale, resizes, and normalizes.
- **Action Space**: Maps game buttons and combos into a simplified discrete action space.
- **Reward System**: Encourages attacking the opponent and penalizes taking damage.

### `train_run.py`
Handles model training and evaluation:

- **Train AI**: Uses `RecurrentPPO` with LSTM to learn optimal strategies.
- **Run AI**: Loads the trained model and executes gameplay.

### `main.py`
Entry point for training and running the AI. Uses command-line arguments:

- `--train`: Trains the model.
- `--run`: Runs the trained model.

## Future Improvements

- Implement Self-Play for better AI training.
- Enhance action space for more complex strategies.
- Improve reward shaping for better decision-making.

## License

This project is for research and educational purposes only. *Mortal Kombat II* is a copyrighted property of its respective owners.

