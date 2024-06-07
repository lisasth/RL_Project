"""
Train DQN model on RiverRaid-v5 environment with RGB observations
"""

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import wandb
from wandb.integration.sb3 import WandbCallback


def init_env(env_name, n_envs, seed=0):
    """
    initialize environment with given parameters
    """
    atari_env = make_atari_env(env_name, n_envs=n_envs, seed=seed)
    return atari_env


def get_device():
    """
    get device GPU or CPU
    """
    return "mps" if torch.backends.mps.is_available() else "cpu"


DEVICE = get_device()

# custom callback to log action distribution
class CustomCallback(BaseCallback):
    """
    custom callback to log action distribution
    """

    def _on_step(self):
        """
        on step callback function
        """
        action = self.model.predict(self.model.policy.observation_space.sample())
        if action is not None and not isinstance(
            action, tuple
        ):  # check if action is not None and not a tuple
            action_counts = np.bincount(
                action, minlength=self.model.action_space.n
            )  # count actions
            action_probs = action_counts / np.sum(
                action_counts
            )  # calculate action probabilities
            self.logger.record(
                "Action Distribution", wandb.Histogram(action_probs)
            )  # log action distribution
        return True


# initialize wandb
wandb.init(
    project="RL_Models",
    entity="lisa-stuch",
    sync_tensorboard=True,
    mode="offline",
    name="DQN_RR_Training",
)

# initialize environment and train DQN model
env = init_env("ALE/Riverraid-v5", n_envs=1, seed=0)

model_dqn = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/dqn_tensorboard/",
    device=DEVICE,
)

# train DQN model with custom callback
model_dqn.learn(total_timesteps=1_000_000, callback=[WandbCallback(), CustomCallback()]) # train
model_dqn.save("dqn_rr_trained")  # save model
wandb.finish()  # finish logging
