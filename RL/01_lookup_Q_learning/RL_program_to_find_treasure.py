import pandas as pd
import numpy as np
import time

N_STATES = 6 # the length of the 1 dimensional world
ACTIONS = ['left', 'right'] # available actions
EPSILON = 0.9 # greedy policy (exploration - exploitation)
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # discount factor
FRESH_TIME = 0.3 # fresh time for one move
MAX_EPISODES = 50 # maximum episodes


# Building the Q table for state-action pairs
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), # q_table initial values
        columns = actions, # actions's name
    )
    #print(table)
    return table


# Choose the action according to the epsilon-greedy policy
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :] # current state's actions
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()): # choose random action --> exploration
        action_name = np.random.choice(ACTIONS)
    else: # choose best action  --> exploitation
        action_name = ACTIONS[state_actions.argmax()]
    return action_name


def get_env_feedback(S, A):
    if A == 'right': # move right
        if S == N_STATES - 2: # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else: # move left
        R = 0
        if S == 0:
            S_ = S # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):

    env_list = ['-']*(N_STATES-1) + ['T'] # '---------T' our environment

    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)         #Join all items in a tuple into a string, using a hash character as separator
        print('\r{}'.format(interaction), end='')   #\r will just work as you have shifted your cursor to the beginning of the string or line
        time.sleep(FRESH_TIME)


def rl():

    q_table = build_q_table(N_STATES, ACTIONS)

    for episode in range(MAX_EPISODES):

        time.sleep(FRESH_TIME)

        step_counter = 0
        S = 0 # initial state
        is_terminated = False

        update_env(S, episode, step_counter)

        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A) # take action & get next state and reward

            q_predict = q_table.loc[S, A] # current state's action value

            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max() # next state is not terminal

            else:
                q_target = R # next state is terminal
                is_terminated = True # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict) # update

            S = S_ # move to next state
            update_env(S, episode, step_counter+1)
            step_counter += 1

    return q_table
        


q_table = rl()
print('\r\nQ-table:\n')
print(q_table)