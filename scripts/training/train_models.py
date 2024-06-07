"""
training of different models on the Riverraid environment
the different models are A2C, PPO, and DQN which are trained for 1,000,000 timesteps each
"""

import os
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import torch
import wandb
from wandb.integration.sb3 import WandbCallback


def init_env(env_name, n_envs, n_stack, seed=0):
    """
    initialize environment with given parameters
    """
    atari_env = make_atari_env(env_name, n_envs=n_envs, seed=seed) # create atari environment
    atari_env = VecFrameStack(atari_env, n_stack=n_stack) # stack frames
    return atari_env


def get_device():
    """
    get device GPU or CPU
    """
    return "mps" if torch.backends.mps.is_available() else "cpu"


DEVICE = get_device()

# initialize and train A2C model for 1,000,000 timesteps
print("Initializing A2C model")
wandb.init(
    project="RL_Comparison",
    entity="lisa-stuch",
    sync_tensorboard=True,
    mode="offline",
    name="A2C_RR_Training",
)
env = init_env("ALE/Riverraid-v5", n_envs=4, n_stack=4)
model_a2c = A2C(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/a2c_tensorboard/",
    device=DEVICE,
)
model_a2c.learn(
    total_timesteps=1_000_000, # train for 1,000,000 timesteps
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path="./models/a2c",
        verbose=2,
    ),
)
model_a2c.save("a2c_rr")
wandb.finish()
os.system("wandb sync wandb/offline-run-a2c")


# initialize and train PPO modelfor 1,000,000 timesteps
print("Initializing PPO model")
wandb.init(
    project="RL_Comparison_Models",
    entity="lisa-stuch",
    sync_tensorboard=True,
    mode="offline",
    name="PPO_RR_Training",
)
env = init_env("ALE/Riverraid-v5", n_envs=4, n_stack=4)
model_ppo = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/ppo_tensorboard/",
    n_steps=512,
    batch_size=16,
    n_epochs=2,
    device=DEVICE,
)
model_ppo.learn(
    total_timesteps=1_000_000, # train for 1,000,000 timesteps
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path="./models/ppo",
        verbose=2,
    ),
)
model_ppo.save("ppo_rr")
wandb.finish()
os.system("wandb sync wandb/offline-run-ppo")

# initialize and train DQN model for 1,000,000 timesteps
print("Initializing DQN model")
wandb.init(
    project="RL_Comparison",
    entity="lisa-stuch",
    sync_tensorboard=True,
    mode="offline",
    name="DQN_RR_Training",
)
env = make_atari_env("ALE/Riverraid-v5", n_envs=1, seed=0)
model_dqn = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/dqn_tensorboard/",
    device=DEVICE,
)
model_dqn.learn(
    total_timesteps=1_000_000, # train for 1,000,000 timesteps
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path="./models/dqn",
        verbose=2,
    ),
)
model_dqn.save("dqn_rr") # save model
wandb.finish() # finish logging
os.system("wandb sync wandb/offline-run-dqn") # sync offline run

print("Training completed for all models")
