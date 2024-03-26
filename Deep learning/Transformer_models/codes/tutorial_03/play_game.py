import torch
import torch.nn as nn
import gym
import time

from decision_transformer import DecisionTransformer
from transformer_model import PositionalEncoding
import numpy as np


class time_step_encoder(nn.Module):

    def __init__(self, max_sequence_length, d_model):
        super(time_step_encoder, self).__init__()
        self.layer = nn.Embedding(max_sequence_length, d_model)

    def forward(self, time_steps):
        embedd = self.layer(time_steps)
        return embedd



class embedding_layer(nn.Module):

    def __init__(self, d_in, d_out):
        super(embedding_layer, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=True)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x



def subsequent_mask(size):
    """ Mask out subsequent positions """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    mask = mask == 0
    return mask.bool()



token_size = 64
d_ff = 512
n_head = 4
num_layers = 2
max_suquence_length = 100


#make model
model = DecisionTransformer(d_model=token_size*3,
                            num_head=n_head,
                            d_ff=d_ff,
                            num_layers=num_layers,
                            dropout=0.1,
                            vocab_size=None,
                            src_embed=embedding_layer(d_in=token_size*3, d_out=token_size*3),
                            tgt_embed=embedding_layer(d_in=token_size*3, d_out=token_size*3),
                            positional_encoding=time_step_encoder(max_sequence_length=max_suquence_length, d_model=token_size))


#load the model
model.load_state_dict(torch.load('Deep learning/Transformer_models/codes/tutorial_03/trained_model/decision_transformer_24_.pth'))

import random

def play_game(n_games=100, Rewards_to_go=50, render=False):

    game_rewards = []

    for game in range(n_games):

        env = gym.make('CartPole-v1', render_mode='human')

        observation, _  = env.reset()
        rewards_to_go = [Rewards_to_go]

        #choose a random action
        random_action = random.randint(0, 1)

        action = [random_action]
        steps_count = [0]
        context_length = 10

        #get the state
        state = torch.tensor(observation).unsqueeze(0).unsqueeze(1).float()
        rewards = torch.tensor(rewards_to_go).unsqueeze(0).unsqueeze(1).float()
        action = torch.tensor(action).unsqueeze(0).unsqueeze(1).float()
        steps = torch.tensor(steps_count).unsqueeze(0).unsqueeze(1).int()

        while True:

            if render:
                env.render()
                time.sleep(0.1)

            #cut the context length
            if steps_count[0] > context_length:
                state = state[:, -context_length:, :]
                rewards = rewards[:, -context_length:, :]
                action = action[:, -context_length:, :]
                steps = steps[:, -context_length:, :]

            src_mask = None
            batch_size, seq_len, d_model = state.shape
            tgt_mask = subsequent_mask(seq_len)

            with torch.no_grad():
                model.eval()
                hidden = model.encode(state, action, rewards, steps, src_mask)
                output = model.decode(state, action, rewards, hidden, steps, src_mask, tgt_mask)
                action_pred = model.make_prediction(output)

            action_prob = action_pred[0, -1, 0].item()

            if action_prob > 0.5:
                action_next = 1
            else:
                action_next = 0

            observation, reward, done, terminated, info = env.step(action_next)
            rewards_to_go[0] -= reward
            steps_count[0] += 1

            if done or steps_count[0] + 1 > max_suquence_length:
                print('Episode finished after {} timesteps'.format(steps_count[0]))
                game_rewards.append(steps_count[0])
                break

            action_next = torch.tensor([action_next]).unsqueeze(0).unsqueeze(1).float()

            #add observation to the state
            state = torch.cat((state, torch.tensor(observation).unsqueeze(0).unsqueeze(1).float()), dim=1)
            rewards = torch.cat((rewards, torch.tensor(rewards_to_go).unsqueeze(0).unsqueeze(1).float()), dim=1)
            action = torch.cat((action, action_next), dim=1)
            steps = torch.cat((steps, torch.tensor(steps_count).unsqueeze(0).unsqueeze(1)), dim=1).int()

        env.close()
        del state, rewards, action, steps


    print('Average reward: {}'.format(np.mean(game_rewards)))
    
    return





play_game(n_games=100, Rewards_to_go=50, render=True)


