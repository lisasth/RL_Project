"""
Training DQN model on the Riverraid environment for obs_type='ram' observations for 
1_000_000 timesteps. This time using MlpPolicy instead of CnnPolicy for RAM 
observations (no need for convolutional layers here).
"""

import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.spaces import Box
import gymnasium as gym
import torch
import wandb
from wandb.integration.sb3 import WandbCallback


def make_ram_env(env_id, seed=0):
    """
    create environment for RAM observations
    """

    def make_env():
        '''
        initialize environment iwth RAM observations
        '''
        atari_env = gym.make(env_id, obs_type="ram")
        atari_env.seed(seed)
        return atari_env

    # initialize environment with multiple environments in parallel by vectorizing them
    atari_env = DummyVecEnv([make_env]) #using DummyVecEnv to run multiple environments in parallel
    return atari_env


def get_device():
    """
    get device GPU or CPU
    """
    return "mps" if torch.backends.mps.is_available() else "cpu"


DEVICE = get_device()

# initialize and train DQN model on RAM observations
wandb.init(
    project="RL_DQN_RAM",
    entity="lisa-stuch",
    sync_tensorboard=True,
    mode="offline",
    name="DQN_RAM",
)
# seed=0 ensures reproducibility in the environment's random elements (initial states, etc.)
env = make_ram_env("ALE/Riverraid-ram-v5", seed=0)

# ensure that observation space is an instance of Gym's Box class
# Box space represents n-dimensional box (continuous values within specified bounds)
assert isinstance(env.observation_space, Box) and env.observation_space.shape == (
    128, # checks that shape of observation space is a 1D array with 128 elements
), "Observation space should be 128-dim ram"

# this time using MlpPolicy for RAM observations (no need for convolutional layers here)
model_dqn = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/dqn_tensorboard/",
    device=DEVICE,
)
model_dqn.learn(
    total_timesteps=1_000_000, # train for 1,000,000 timesteps
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path="./models/dqn_ram",
        verbose=2,
    ),
)

model_dqn.save("dqn_rr_ram") # save model
wandb.finish() # finish logging
os.system("wandb sync wandb/offline-run-dqn") # sync offline run
print("Training completed")
