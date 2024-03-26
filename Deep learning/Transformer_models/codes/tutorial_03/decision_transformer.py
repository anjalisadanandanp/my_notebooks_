import sys
sys.path.append('/mnt/data2/Deep learning/Transformer_models/codes/tutorial_03')

import torch
import torch.nn as nn
from transformer_model import Encoder, Decoder, LayerNorm, clones
class DecisionTransformer(nn.Module):

    def __init__(self, 
                 d_model, 
                 num_head, 
                 d_ff, 
                 num_layers, 
                 dropout, 
                 vocab_size, 
                 src_embed, 
                 tgt_embed, 
                 positional_encoding):
        
        super(DecisionTransformer, self).__init__()

        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.src_embedding = src_embed
        self.tgt_embedding = tgt_embed
        self.positional_encoding = positional_encoding

        self.encoder = Encoder(d_model, num_head, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_head, d_ff, num_layers, dropout)

        n_observations = 4
        self.d_in = d_model//3

        self.linear_state_embed = nn.Linear(n_observations, self.d_in)
        self.state_embed = nn.Tanh()
        self.linear_action_embed = nn.Linear(1, self.d_in)
        self.action_embed = nn.Tanh()
        self.linear_rewards_embed = nn.Linear(1, self.d_in)
        self.rewards_embed = nn.Tanh()

        self.layer_norm_actions = LayerNorm(self.d_model//3)
        self.layer_norm_rewards = LayerNorm(self.d_model//3)
        self.layer_norm_states = LayerNorm(self.d_model//3)

        self.state_pred = nn.Linear(self.d_model//3, n_observations)
        self.action_pred = nn.Sequential(nn.Linear(self.d_model, 1), nn.Sigmoid())
        self.reward_pred = nn.Linear(self.d_model//3, 1)
        
    def state_emdedding(self, state):
        state = self.linear_state_embed(state)
        state = self.state_embed(state)
        return state

    def action_embedding(self, action):
        action = self.linear_action_embed(action)
        action = self.action_embed(action)
        return action

    def rewards_embedding(self, rewards):
        rewards = self.linear_rewards_embed(rewards)
        rewards = self.rewards_embed(rewards)
        return rewards
    
    def merge(self, observations, actions, rewards, time_steps):

        observations = self.state_emdedding(observations) 
        actions = self.action_embedding(actions)
        rewards = self.rewards_embedding(rewards)

        observations = self.layer_norm_states(observations)
        actions = self.layer_norm_actions(actions)
        rewards = self.layer_norm_rewards(rewards)

        time_embedding = self.positional_encoding(time_steps).squeeze(2)

        observations = observations + time_embedding
        actions = actions + time_embedding
        rewards = rewards + time_embedding

        x = torch.cat((rewards, observations, actions), dim=2)

        return x
    
    def split(self, x):
        observations = x[:, :, :self.d_model//3]
        actions = x[:, :, self.d_model//3:2*self.d_model//3]
        rewards = x[:, :, 2*self.d_model//3:]
        return rewards, observations, actions
    
    def encode(self, observations, actions, rewards, time_steps, src_mask=None):
        x = self.merge(observations, actions, rewards, time_steps)
        x = self.src_embedding(x)
        x = self.encoder(x, src_mask)
        return x
    
    def decode(self, observations, actions, rewards, memory, time_steps, src_mask=None, tgt_mask=None):
        
        x = self.merge(observations, actions, rewards, time_steps)
        x = self.tgt_embedding(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return x
    
    def make_prediction(self, x):

        # reward, state, action  = self.split(x)
        # state = self.state_pred(state)
        # action = self.action_pred(action)
        # reward = self.reward_pred(reward)

        action = self.action_pred(x)
        
        return action 
    


