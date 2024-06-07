"""
Training DQN model on the Riverraid environment for obs_type='grayscale' 
observations for 1_000_000 timesteps.
"""

import os
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import AtariPreprocessing
import wandb
from wandb.integration.sb3 import WandbCallback


def make_atari_environment(env_id, n_envs, seed=0, grayscale=True, frame_stack=4):
    """
    create environment with preprocessing including grayscale transformation
    """

    def make_env():
        atari_env = gym.make(env_id) # create atari environment
        atari_env.seed(seed)  # set seed
        if grayscale:
            # atari preprocessing including grayscaling and resizing
            atari_env = AtariPreprocessing(
                atari_env, screen_size=84, grayscale_newaxis=True, grayscale_obs=True
            )  # grayscale observations
        return atari_env

    # initialize environment with multiple environments in parallel by vectorizing them
    atari_env = DummyVecEnv([make_env() for _ in range(n_envs)])  # DummyVecEnv allows to run multiple environments in parallel
    if frame_stack > 1: # wraps the environment to stack consecutive frames together
        atari_env = VecFrameStack(atari_env, n_stack=frame_stack)  # stack frames
    return atari_env


def get_device():
    """
    get device GPU or CPU
    """
    return "mps" if torch.backends.mps.is_available() else "cpu"


DEVICE = get_device()

# initialize and train DQN model using grayscale observations for 1,000,000 timesteps
# this time using CnnPolicy for convolutional layers
# RiveraidNoFrameskip-v4 environment is used because it has no frame skipping
wandb.init(
    project="Riverraid_Models",
    entity="lisa-stuch",
    sync_tensorboard=True,
    mode="offline",
    name="DQN_Grayscale",
)

# initialize environment and model for training DQN with grayscale observations
env = make_atari_environment(
    "RiverraidNoFrameskip-v4", n_envs=1, seed=0, grayscale=True, frame_stack=4
)
model_dqn = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/dqn_tensorboard_greyscale/",
    device=DEVICE,
)
model_dqn.learn(
    total_timesteps=1_000_000,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path="./models/dqn_grayscale",
        verbose=2,
    ),
)
model_dqn.save("dqn_rr_grayscale") # save model
wandb.finish() # finish logging
os.system("wandb sync wandb/offline-run-dqn") # sync offline run
print("Training completed for all models")
