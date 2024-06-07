"""
Compares trained PPO, A2C and DQN models and the random agent on their 
performance on the Riverraid environment. They are loaded from the 
saved models "ppo_rr", "dqn_rr", "dqn_rr_ram" and "a2c_rr". Each model
is ran for 100 episodes and the scores are plotted to compare the 
performance of the models.
"""

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


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
    env = make_atari_env(env_name, n_envs=n_envs, seed=seed)  # create atari environment
    if stack_frames:
        env = VecFrameStack(env, n_stack=n_stack)  # stack frames
    return env # return environment (can be vectorized or basic, and frames can be stacked or not)


def random_agent(env, n_episodes=25):
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
        scores.append(total_reward)
        print(f"Episode {episode + 1}, score: {total_reward}")
    return scores


# initialize all environments
env_ppo = init_env("ALE/Riverraid-v5", n_envs=1, stack_frames=True, n_stack=4)
env_a2c = init_env("ALE/Riverraid-v5", n_envs=1, stack_frames=True, n_stack=4)
env_dqn = init_env("ALE/Riverraid-v5", n_envs=1, stack_frames=False)
env_random = gym.make("ALE/Riverraid-v5")

# custom objects for loading the models (no custom objects needed)
custom_objects = {
    "lr_schedule": lambda x: x, # no transformation applied to learning rate schedule
    "clip_range": lambda x: x # no transformation applied to clipping range (uses directly the value from the model)
    }

# load PPO and A2C models with custom objects
model_ppo = PPO.load("models/ppo_rr", env=env_ppo, custom_objects=custom_objects)
model_a2c = A2C.load("models/a2c_rr", env=env_a2c, custom_objects=custom_objects)
# load DQN model
model_dqn = DQN.load("models/dqn_rr", env=env_dqn)


def run_episodes(model, env, n_episodes=100):
    """
    run 100 episodes for each model and collect scores considering only every 4th episode
    because of the frameskipping=4 in the environment
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
        if (episode + 1) % 4 == 0:  # consider only every 4th episode
            if "episode" in infos[0]:
                episode_score = infos[0]["episode"]["r"]
            else:
                episode_score = total_reward
            scores.append(episode_score)
            print(f"Episode {episode + 1}: Score: {episode_score}")
    return scores


# run episodes and collect scores for both models
print("Run PPO model")
scores_ppo = run_episodes(model_ppo, env_ppo, n_episodes=100)
print("Run A2C model")
scores_a2c = run_episodes(model_a2c, env_a2c, n_episodes=100)
print("Run DQN model")
scores_dqn = run_episodes(model_dqn, env_dqn, n_episodes=100)
print("Run Random Agent")
scores_random = random_agent(env_random, n_episodes=100)

# plot scores and save image
plt.figure(figsize=(12, 6))

if scores_ppo:
    plt.plot(
        range(4, 4 * len(scores_ppo) + 1, 4),
        scores_ppo,
        label="PPO",
        color="b",
        marker="o",
    )
    avg_ppo = np.mean(scores_ppo)
    plt.axhline(y=avg_ppo, color="b", linestyle="-", label=f"Avg PPO: {avg_ppo:.2f}")

if scores_a2c:
    plt.plot(
        range(4, 4 * len(scores_a2c) + 1, 4),
        scores_a2c,
        label="A2C",
        color="r",
        marker="x",
    )
    avg_a2c = np.mean(scores_a2c)
    plt.axhline(y=avg_a2c, color="r", linestyle="-", label=f"Avg A2C: {avg_a2c:.2f}")

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
plt.title("Performance comparison between PPO, A2C, and DQN agents")
plt.legend()
plt.savefig("images/compare_PPO_A2C_DQN.png")
plt.show()

env_ppo.close()
env_a2c.close()
env_dqn.close()
env_random.close()
