"""
Training of different models on the Riverraid environment for only 5 minutes
the different models are A2C, PPO, and DQN which are trained for 50,000 timesteps each.
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
    atari_env = make_atari_env(
        env_name, n_envs=n_envs, seed=seed
    )  # create atari environment
    atari_env = VecFrameStack(
        atari_env, n_stack=n_stack
    )  # use VecFrameStack to stack frames together
    return atari_env


def get_device():
    """
    get device GPU or CPU
    """
    return "mps" if torch.backends.mps.is_available() else "cpu"


DEVICE = get_device()

# initialize and train A2C model for 5 min training
print("Initializing A2C model")
wandb.init(
    project="Riverraid_Models",
    entity="lisa-stuch",
    sync_tensorboard=True,
    mode="offline",
    name="A2C_5min",
)
env = init_env("ALE/Riverraid-v5", n_envs=4, n_stack=4)
model_a2c = A2C(
    "CnnPolicy", env, verbose=1, tensorboard_log="./logs/a2c_5min/", device=DEVICE
)
model_a2c.learn(
    total_timesteps=50_000,  # 50,000 timesteps equal to 5 minutes
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path="./models/5min/a2c",
        verbose=2,
    ),
)
model_a2c.save("a2c_5min")
wandb.finish()
os.system("wandb sync ./wandb/offline-run-a2c-5")

# initialize and train PPO model for 5 min training
print("Initializing PPO model")
wandb.init(
    project="Riverraid_Models",
    entity="lisa-stuch",
    sync_tensorboard=True,
    mode="offline",
    name="PPO_5min",
)
env = init_env("ALE/Riverraid-v5", n_envs=4, n_stack=4)
model_ppo = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/ppo_5min/",
    n_steps=512,
    batch_size=16,
    n_epochs=2,
    device=DEVICE,
)
model_ppo.learn(
    total_timesteps=50_000,  # 50,000 timesteps equal to 5 minutes
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path="./models/5min/ppo",
        verbose=2,
    ),
)
model_ppo.save("ppo_5min")
wandb.finish()
os.system("wandb sync ./wandb/offline-run-ppo-5")

# initialize and train DQN model for 5 min training
print("Initializing DQN model")
wandb.init(
    project="Riverraid_Models",
    entity="lisa-stuch",
    sync_tensorboard=True,
    mode="offline",
    name="DQN_5min",
)
env = make_atari_env("ALE/Riverraid-v5", n_envs=1, seed=0)
model_dqn = DQN(
    "CnnPolicy", env, verbose=1, tensorboard_log="./logs/dqn_5min/", device=DEVICE
)
model_dqn.learn(
    total_timesteps=50_000,  # 50,000 timesteps equal to 5 minutes
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path="./models/5min/dqn",
        verbose=2,
    ),
)
model_dqn.save("dqn_5min")
wandb.finish()
os.system("wandb sync ./wandb/offline-run-dqn-5")

print("Training completed for all models for 5 minutes")
