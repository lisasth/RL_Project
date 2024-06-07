'''
random agent for the Riverraid environment which runs an episode with a random agent
'''

import gymnasium as gym

# run episode with random agent and render environment (human mode)
env = gym.make('ALE/Riverraid-v5', render_mode='human')
observation, info = env.reset() # reset environment
DONE = False
TOTAL_REWARD = 0
STEP_COUNT = 0

while not DONE:
    action = env.action_space.sample() # random action from action space
    observation, reward, DONE, info, _ = env.step(action) # take action in environment
    env.render() # render environment
    TOTAL_REWARD += reward # accumulate reward
    STEP_COUNT += 1  # count steps
    print(f"Step {STEP_COUNT}, Points: {TOTAL_REWARD}")

env.close() # close environment
print(f"Game finished with score: {TOTAL_REWARD}")
