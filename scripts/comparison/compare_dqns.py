"""
Compares trained DQN models with different observation tyoes
ram, rgb, and greyscale and the random agent on their 
performance on the Riverraid environment. Each model
is ran for 100 episodes and the scores are plotted to compare the 
performance of the models.
"""

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing


def init_env(env_name, n_envs, seed=0, stack_frames=True, n_stack=4):
    """
    initialize environment with given parameters

    :param env_name: (str) name of the environment
    :param n_envs: (int) number of environments
    :param seed: (int) seed for environment
    :param stack_frames: (bool) whether to stack frames or not
    :param n_stack: (int) number of frames to stack (if stacking is enabled)
    :return: environment (either VecFrameStack or basic depending on settings)
    """
    env = make_atari_env(env_name, n_envs=n_envs, seed=seed)
    if stack_frames:
        env = VecFrameStack(env, n_stack=n_stack)
    return env


def init_ram_env(env_name, n_envs, seed=0):
    """
    initialize RAM environment with the given parameters

    :param env_name: (str) name of the environment
    :param n_envs: (int) number of environments
    :param seed: (int) seed for the environment
    :return: environment (DummyVecEnv) configured for RAM observations
    """

    def make_env():
        env = gym.make(env_name, obs_type="ram")
        env.seed(seed)
        return env

    env = DummyVecEnv([make_env])
    return env


def random_agent(env, n_episodes=100):
    """
    run 100 episodes with random agent in environment and collect scores

    :param env: (gym.Env) environment
    :param n_episodes: (int) number of episodes to run
    :return: list of total rewards per episode
    """
    scores = []
    for episode in range(n_episodes):
        observation, info = env.reset()
        DONE = False
        total_reward = 0
        while not DONE:
            action = env.action_space.sample()
            observation, reward, DONE, info, _ = env.step(action)
            total_reward += reward
        if (episode + 1) % 4 == 0:  # collect score only for every 4th episode
            scores.append(total_reward)
            print(f"Episode {episode + 1}, Score: {total_reward}")
    return scores


def init_greyscale_env(env_name, n_envs, seed=0, stack_frames=True, n_stack=4):
    """
    initialize greyscale environment with preprocessing and optional frame stacking
    """

    def make_env():
        """
        creates environment with preprocessing
        """
        env = gym.make(env_name)
        # wraps the environment with preprocessing from atari (gymnasium wrapper)
        # to preprocess raw images into grayscale and resize them to 84x84 pixels
        env = AtariPreprocessing(
            env, screen_size=84, grayscale_newaxis=True, grayscale_obs=True
        )
        return env

    env = DummyVecEnv([make_env for _ in range(n_envs)])  # create multiple environments
    if stack_frames:
        env = VecFrameStack(env, n_stack=n_stack)  # stack frames
    return env


# initialize all environments
env_dqn = init_env("ALE/Riverraid-v5", n_envs=1, stack_frames=False)  # rgb observations
env_dqn_ram = init_ram_env("ALE/Riverraid-ram-v5", n_envs=1, seed=0)  # ram observations
env_dqn_greyscale = init_greyscale_env(
    "RiverraidNoFrameskip-v4",
    n_envs=1,
    stack_frames=True,
    n_stack=4,  # greyscale observations
)
env_random = gym.make("ALE/Riverraid-v5")

# load trained models for different DQN models
model_dqn = DQN.load("models/dqn_rr", env=env_dqn)  # rgb observations
model_dqn_ram = DQN.load("models/dqn_rr_ram", env=env_dqn_ram)  # ram observations
model_dqn_greyscale = DQN.load(
    "models/dqn_rr_grayscale", env=env_dqn_greyscale
)  # greyscale observations


def run_episodes(model, env, n_episodes=100, frame_skipping=True):
    """
    run 100 episodes for each model and collect scores

    :param model: the model to run
    :param env: the environment to run the model in
    :param n_episodes: the number of episodes to run
    :param frame_skipping: flag to indicate if frame skipping is used (affects episode recording)
    """
    scores = []
    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0] if isinstance(rewards, np.ndarray) else rewards
            done = dones.any() if isinstance(dones, np.ndarray) else dones
        # conditions changed based on whether frame skipping is used or not
        if not frame_skipping or (episode + 1) % 4 == 0:
            if "episode" in infos[0]:
                episode_score = infos[0]["episode"]["r"]
            else:
                episode_score = total_reward
            scores.append(episode_score)
            print(f"Episode {episode + 1}: score: {episode_score}")
    return scores


# run episodes and collect scores for both models
print("Run DQN model")
scores_dqn = run_episodes(model_dqn, env_dqn, n_episodes=100)
print("Run DQN RAM model")
scores_dqn_ram = run_episodes(model_dqn_ram, env_dqn_ram, n_episodes=100)
print("Run DQN Greyscale model")
scores_dqn_greyscale = run_episodes(
    model_dqn_greyscale, env_dqn_greyscale, n_episodes=100
)
print("Run Random Agent")
scores_random = random_agent(env_random, n_episodes=100)

# plot scores and save image
plt.figure(figsize=(12, 6))

if scores_dqn:
    plt.plot(range(len(scores_dqn)), scores_dqn, label="DQN", color="g", marker="*")
    avg_dqn = np.mean(scores_dqn)
    plt.axhline(y=avg_dqn, color="g", linestyle="-", label=f"Avg DQN: {avg_dqn:.2f}")

if scores_dqn_ram:
    plt.plot(
        range(len(scores_dqn_ram)),
        scores_dqn_ram,
        label="DQN RAM",
        color="c",
        marker="s",
    )
    avg_dqn_ram = np.mean(scores_dqn_ram)
    plt.axhline(
        y=avg_dqn_ram, color="c", linestyle="-", label=f"Avg DQN RAM: {avg_dqn_ram:.2f}"
    )

if scores_dqn_greyscale:
    plt.plot(
        range(len(scores_dqn_greyscale)),
        scores_dqn_greyscale,
        label="DQN Greyscale",
        color="black",
        marker="D",
    )
    avg_dqn_greyscale = np.mean(scores_dqn_greyscale)
    plt.axhline(
        y=avg_dqn_greyscale,
        color="black",
        linestyle="-",
        label=f"Avg DQN Greyscale: {avg_dqn_greyscale:.2f}",
    )

if scores_random:
    plt.plot(
        range(len(scores_random)), scores_random, label="Random", color="m", marker="D"
    )
    avg_random = np.mean(scores_random)
    plt.axhline(
        y=avg_random, color="m", linestyle="-", label=f"Avg Random: {avg_random:.2f}"
    )

plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Performance comparison between different DQN models")
plt.legend()
plt.savefig("images/compare_DQNs_different.png")
plt.show()

env_dqn.close()
env_dqn_ram.close()
env_random.close()
env_dqn_greyscale.close()
