import gym
import pandas as pd
import numpy as np

#set up 
#env = gym.make('Taxi-v3')
env = gym.make('Taxi-v3', render_mode='rgb_array')

# Print the number of states and actions
print("Observation space:", env.observation_space)    # 500 states
print("Action space:", env.action_space)         # 6 actions

env.reset()
#env.render()

def create_qtable(n_states, n_actions):
    qtable = pd.DataFrame(np.zeros((n_states, n_actions)), columns=range(n_actions))       #values are 0
    #qtable = pd.DataFrame(np.random.random((n_states, n_actions)), columns=range(1, n_actions+1))    #values are random
    return qtable

n_states = env.observation_space.n
n_actions = env.action_space.n

qtable = create_qtable(n_states, n_actions)

#hyperparameters
alpha = 0.2
gamma = 0.95
epsilon = 0.1
num_training_episodes = 1000

def choose_action_e_greedy(state, qtable):
    if np.random.uniform() < epsilon:       #random actions
        action = np.random.choice(qtable.columns)
    else:                                   #greedy actions
        action = qtable.loc[state].argmax()
    return action

def update_qtable(qtable, state, action, reward, next_state, next_action):
    qtable.loc[state, action] += alpha * (reward + gamma * qtable.loc[next_state, next_action] - qtable.loc[state, action])
    return qtable


for episode in range(1, num_training_episodes+1):

    print("EPISODE: ", episode)

    state, info = env.reset()
    terminated = False

    while not terminated:

        current_action = choose_action_e_greedy(state = state, qtable = qtable)
        next_state, reward, terminated, done, info = env.step(action=current_action)
        next_action = choose_action_e_greedy(state = next_state, qtable = qtable)
        update_qtable(qtable, state, current_action, reward, next_state, next_action)
        state = next_state


print("QTABLE:", qtable)

env.close()

env = gym.make('Taxi-v3', render_mode='human')

# Test the agent
state, info = env.reset()
env.render()
epsilon = 0   # no exploration
terminated = False    
while not terminated:
    action = choose_action_e_greedy(state = state, qtable = qtable)
    state, reward, terminated, done, info = env.step(action=action)

env.close()
    