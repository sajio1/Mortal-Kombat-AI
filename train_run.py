import multiprocessing
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO
from environment import MortalKombatEnv

# -------------------- Environment Creation --------------------
def create_env():
    """
    Create an instance of the Mortal Kombat environment.
    """
    return MortalKombatEnv()

# -------------------- Training Function --------------------
def train_model():
    """
    Train the AI using Recurrent PPO (LSTM) with parallel environments.
    """
    num_envs = max(2, multiprocessing.cpu_count() - 2)  # Adjust based on CPU cores

    # Use SubprocVecEnv for parallelized training
    env = SubprocVecEnv([lambda: create_env() for _ in range(num_envs)])

    # Define LSTM-based PPO model
    model = RecurrentPPO(
        "MlpLstmPolicy",  # Changed from "CnnLstmPolicy" to "MlpLstmPolicy"
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs={"lstm_hidden_size": 256},  # Enables LSTM memory
        tensorboard_log="./ppo_mk2_tensorboard/"
    )

    # Train the model
    model.learn(total_timesteps=1_00_000)

    # Save the trained model
    model.save("mortal_kombat_ppo_lstm")
    env.close()

# -------------------- Running Function --------------------
def run_model():
    """
    Run the trained AI in Mortal Kombat II.
    """
    env = MortalKombatEnv()

    # Load the trained model
    model = RecurrentPPO.load("mortal_kombat_ppo_lstm")

    obs = env.reset()
    lstm_states = None  # Track LSTM states
    episode_start = True  # Used for LSTM rollout
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            lstm_states = None  # Reset LSTM memory
            episode_start = True
