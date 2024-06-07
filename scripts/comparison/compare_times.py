"""
Compares trained DQN models for 5 vs. 30 minutes on their 
performance on the Riverraid environment. They are loaded from the 
saved models "dqn_rr_5min", "dqn_rr_30min", "dqn_rr" and "dqn_ram". 
Each model is ran for 100 episodes and the scores are plotted to 
compare the performance of the models.
"""

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


def init_env(env_name, n_envs, seed=0, stack_frames=True, n_stack=4):
    """
    initialize environment with given parameters

    :param env_name: (str) name of environment
    :param n_envs: (int) number of environments
    :param seed: (int) seed for the environment
    :param stack_frames: (bool) whether to stack frames or not
    :param n_stack: (int) number of frames to stack if stacking is enabled
    :return: environment (either VecFrameStack or basic depending on settings)
    """
    env = make_atari_env(env_name, n_envs=n_envs, seed=seed)
    if stack_frames:
        env = VecFrameStack(env, n_stack=n_stack)
    return env


def init_ram_env(env_name, n_envs, seed=0):
    """
    initialize RAM environment with given parameters

    :param env_name: (str) name of environment
    :param n_envs: (int) number of environments
    :param seed: (int) seed for environment
    :return: environment (DummyVecEnv) configured for RAM observations
    """

    def make_env():
        env = gym.make(env_name, obs_type="ram") # create RAM environment
        env.seed(seed) # set seed for environment
        return env

    env = DummyVecEnv([make_env]) # create DummyVecEnv to enable multiple environments
    return env


def random_agent(env, n_episodes=100):
    """
    run episodes with random agent in environment and collect scores

    :param env: (gym.Env) environment to run the episode in
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
        scores.append(total_reward)
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
    return scores


# initialize environments for DQN models and random agent
env_dqn = init_env(
    "ALE/Riverraid-v5", n_envs=1, stack_frames=False
)  # dqn model trained for 1,000,000 timesteps
env_dqn_5 = init_env(
    "ALE/Riverraid-v5", n_envs=1, stack_frames=False
)  # dqn model trained for 5 minutes
env_dqn_30 = init_env(
    "ALE/Riverraid-v5", n_envs=1, stack_frames=False
)  # dqn model trained for 30 minutes
env_random = gym.make("ALE/Riverraid-v5")  # random agent

custom_objects = {"lr_schedule": lambda x: x, "clip_range": lambda x: x}

# load the trained models
model_dqn = DQN.load("models/dqn_rr", env=env_dqn)  # trained for 1,000,000 timesteps
model_dqn_5 = DQN.load("models/dqn_rr_5min", env=env_dqn_5)  # trained for 5 minutes
model_dqn_30 = DQN.load("models/dqn_rr_30min", env=env_dqn_30)  # trained for 30 minutes


def run_episodes(model, env, n_episodes=100):
    """
    run 100 episodes with model and collect scores considering only every 4th episode
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
        if (episode + 1) % 4 == 0:  # consider only every 4th episode because of frameskipping=4
            if "episode" in infos[0]:
                episode_score = infos[0]["episode"]["r"]
            else:
                episode_score = total_reward
            scores.append(episode_score)
            print(f"Episode {episode + 1}: score: {episode_score}")
    return scores


# run episodes and collect scores for all models
print("Run DQN_5 model")
scores_dqn_5 = run_episodes(model_dqn_5, env_dqn_5, n_episodes=100)
print("Run DQN_30 model")
scores_dqn_30 = run_episodes(model_dqn_30, env_dqn_30, n_episodes=100)
print("Run DQN model")
scores_dqn = run_episodes(model_dqn, env_dqn, n_episodes=100)
print("Run random agent")
scores_random = random_agent(env_random, n_episodes=100)

plt.figure(figsize=(12, 6))

# plot scores
if scores_dqn_5:
    plt.plot(
        range(4, 4 * len(scores_dqn_5) + 1, 4),
        scores_dqn_5,
        label="5 min",
        color="b",
        marker="o",
    )
    avg_dqn_5 = np.mean(scores_dqn_5)
    plt.axhline(
        y=avg_dqn_5, color="b", linestyle="-", label=f"Avg DQN 5 min: {avg_dqn_5:.2f}"
    )

if scores_dqn_30:
    plt.plot(
        range(4, 4 * len(scores_dqn_30) + 1, 4),
        scores_dqn_30,
        label="30 min",
        color="r",
        marker="x",
    )
    avg_dqn_30 = np.mean(scores_dqn_30)
    plt.axhline(
        y=avg_dqn_30,
        color="r",
        linestyle="-",
        label=f"Avg DQN 30 min: {avg_dqn_30:.2f}",
    )

if scores_dqn:
    plt.plot(
        range(4, 4 * len(scores_dqn) + 1, 4),
        scores_dqn,
        label="DQN",
        color="g",
        marker="*",
    )
    avg_dqn = np.mean(scores_dqn)
    plt.axhline(y=avg_dqn, color="g", linestyle="-", label=f"Avg DQN: {avg_dqn:.2f}")

if scores_random:
    plt.plot(
        range(1, len(scores_random) + 1),
        scores_random,
        label="Random",
        color="m",
        marker="D",
    )
    avg_random = np.mean(scores_random)
    plt.axhline(
        y=avg_random, color="m", linestyle="-", label=f"Avg Random: {avg_random:.2f}"
    )

plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Performance comparison between DQN´s")
plt.legend()
plt.savefig("compare_models.png")
plt.show()

env_dqn.close()
env_dqn_5.close()
env_dqn_30.close()
env_random.close()
