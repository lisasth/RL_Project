"""
run demo of the DQN agent trained on the Riverraid environment to see performance
"""

from stable_baselines3 import DQN
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env(env_name):
    """
    create environment with preprocessing and enable rendering
    """

    def _init():
        """
        initialize environment
        """
        env = gym.make(
            env_name, render_mode="human"
        )  # render_mode='human' to enable visual rendering
        env = ResizeObservation(env, (84, 84))  # resize observations to 84x84
        env = GrayScaleObservation(
            env, keep_dim=True
        )  # keep_dim=True to keep the channel dimension
        return env

    return _init


# initialize environment within DummyVecEnv to make it compatible with SB3 and load DQN model
env_dqn = DummyVecEnv([make_env("ALE/Riverraid-v5")])  # initialize environment
model_dqn = DQN.load("models/dqn_rr", env=env_dqn)  # load trained DQN model


def run_demo(model, env):
    """
    run episode with model and render each step
    """
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward[0]
    print(f"Game finished with score: {total_reward}")


run_demo(model_dqn, env_dqn)  # run demo
env_dqn.close()  # close environment
