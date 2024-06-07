'''
this script trains a DQN model on the Riverraid environment for RGB observations from scratch
'''

import random
from collections import deque
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class DQN(nn.Module):
    '''
    defines a simple CNN for Q-learning inheriting from nn.Module
    '''
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__() # initialize super class
        self.conv = nn.Sequential( # sequential container for convolutional layers
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # 1st convolutional layer
            nn.ReLU(), # ReLU activation function
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 2nd convolutional layer
            nn.ReLU(), # ReLU activation function
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 3rd convolutional layer
            nn.ReLU() # ReLU activation function
        )
        self.fc = nn.Sequential( # sequential container for fully connected layers
            nn.Linear(self.feature_size(input_shape), 512), # 1st fully connected layer
            nn.ReLU(), # ReLU activation function
            nn.Linear(512, num_actions) # output layer
        )

    def forward(self, x):
        '''
        forward pass through network
        '''
        x = self.conv(x) # apply convolutional layers
        x = x.view(x.size(0), -1) # flatten tensor
        x = self.fc(x) # apply fully connected layers
        return x

    def feature_size(self, input_shape):
        '''
        calculate size of flattened feature tensor
        '''
        return self.conv(torch.zeros(1, *input_shape)).view(1, -1).size(1) # pass dummy input through convolutional layers

# setup environment
env = gym.make('ALE/Riverraid-v5', render_mode='rgb_array')  # using rgb rendering mode
input_shape = (3, 210, 160)  # input shape (C, H, W) (3 channels, 210x160 pixels)
num_actions = env.action_space.n # number of actions from env

# initialize network and optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # zse GPU if available
print(f"Training on: {device}")
DQN_MODEL = DQN(input_shape, num_actions).to(device) # initialize DQN model
optimizer = optim.Adam(DQN_MODEL.parameters()) # initialize Adam optimizer
loss_function = nn.SmoothL1Loss() # initialize loss function

# Preprocess observations
def preprocess(obs):
    '''
    preprocess observation for input to network
    '''
    if isinstance(obs, tuple): # check if observation is a tuple, then get 1st element
        obs = obs[0]
        
    if not isinstance(obs, torch.Tensor): 
        obs = torch.from_numpy(obs).to(device) # if observation is not tensor convert to tensor

    if obs.max() > 1.0:
        obs = obs.float() / 255.0 # normalize observation values to 0-1

    if obs.dim() == 3 and obs.shape[2] == 3:
        obs = obs.permute(2, 0, 1)  # change shape to (C, H, W)
    elif obs.dim() == 3:
        raise ValueError(f"Expected 3 channels, got {obs.shape[2]}")

    if obs.dim() == 3:
        obs = obs.unsqueeze(0) # if there is no batch dimension add it

    return obs


# experience replay buffer
class ReplayBuffer:
    '''
    experience replay buffer
    '''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # create deque to store experiences

    def push(self, state, action, reward, next_state, done):
        '''
        push experience to buffer
        '''
        self.buffer.append((state, action, reward, next_state, done)) # append experience to buffer

    def sample(self, batch_size):
        '''
        sample batch of experiences from buffer
        '''
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size)) #sample batch randomly
        return np.array(state), action, reward, np.array(next_state), done # return sampled batch

    def __len__(self):
        return len(self.buffer)

# define hyperparameters
EPSILON_START = 1.0 # initial epsilon value
EPSILON_FINAL = 0.01 # final epsilon value
EPSILON_DECAY = 500 # epsilon decay rate
EPSILON_BY_FRAME = lambda frame_idx: EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * np.exp(-1. * frame_idx / EPSILON_DECAY) # decay function
BATCH_SIZE = 32
BUFFER_SIZE = 10000 # size of replay buffer
REPLAY_BUFFR = ReplayBuffer(BUFFER_SIZE)
NUM_FRAMES = 100000
GAMMA = 0.99  # discount factor

# training function
def compute_td_loss(batch_size):
    '''
    compute temporal differene loss for training
    '''
    if len(REPLAY_BUFFR) < batch_size: # chekc if enough experiences in buffer
        return None

    state, action, reward, next_state, done = REPLAY_BUFFR.sample(batch_size) # sample batch from buffer

    #print("Sampled state shape and type:", state[0].shape, state[0].dtype)

    try:
        state = torch.cat([preprocess(s) for s in state], dim=0) # preprocess state
    except Exception as e:
        print("Error during preprocessing:", e)
        raise

    next_state = torch.cat([preprocess(s) for s in next_state], dim=0) # preprocess next state
    action = torch.tensor(action, device=device, dtype=torch.long) # convert action to tensor
    reward = torch.tensor(reward, device=device, dtype=torch.float32) # convert reward to tensor
    done = torch.tensor(done, device=device, dtype=torch.float32) # convert done to tensor
    
    #print("Preprocessed state tensor dtype:", state_tensor.dtype)
    q_values = DQN_MODEL(state) # get current q values
    next_q_values = DQN_MODEL(next_state) # get next q values
 
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) # get q value for action
    next_q_value = next_q_values.max(1)[0] # get max q value for next state
    expected_q_value = reward + GAMMA * next_q_value * (1 - done) # calculate expected q value
    
    loss = loss_function(q_value, expected_q_value) # calculate loss

    optimizer.zero_grad() # zero gradients
    loss.backward() # backpropagate loss
    optimizer.step() # update weights
    return loss.item()

# training loop
losses = [] # store losses
all_rewards = [] # store rewards
state = env.reset() # reset environment

if isinstance(state, tuple):
    state = state[0] # get 1st element of tuple
EPISODE_REWARD = 0
#print("Initial state type:", type(state))
#print("Initial state content:", state)

# loop through each frame
for frame_idx in tqdm(range(1, NUM_FRAMES + 1)):
    epsilon = EPSILON_BY_FRAME(frame_idx) # get epsilon value
    state_tensor = preprocess(state) # preprocess state
    
    # select action either randomly or by policy
    if random.random() > epsilon:
        q_values = DQN_MODEL(state_tensor)
        action = q_values.max(1)[1].item()
    else:
        action = env.action_space.sample()

    # step in the environment
    next_state, reward, done, info, _ = env.step(action) # take action
    if isinstance(next_state, tuple):
        next_state = next_state[0]
    EPISODE_REWARD += reward # getfirst element of tuple and add reward

    # process next state and store experience in buffer
    REPLAY_BUFFR.push(state_tensor.cpu().numpy(), action, reward, preprocess(next_state).cpu().numpy(), done)
    
    # update state to next state
    state = next_state
    
    # check if epoisode is done
    if done:
        all_rewards.append(EPISODE_REWARD)
        print(f"Episode completed with total reward {EPISODE_REWARD}")
        state = env.reset()
        EPISODE_REWARD = 0

    # train model if enough data is available
    if len(REPLAY_BUFFR) > BATCH_SIZE:
        loss = compute_td_loss(BATCH_SIZE)
        if loss is not None:
            losses.append(loss)

# plot results
plt.figure(figsize=(20, 5))
plt.subplot(121)

if all_rewards:
    plt.title(f'Frame {frame_idx}. Last 10 avg reward: {np.mean(all_rewards[-10:]):.2f}')
else:
    plt.title('Frame {frame_idx}. No rewards.')
plt.plot(all_rewards)
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.subplot(122)

if losses:
    plt.title('Loss over time')
else:
    plt.title('No losses')
plt.plot(losses)
plt.xlabel('Training steps')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
