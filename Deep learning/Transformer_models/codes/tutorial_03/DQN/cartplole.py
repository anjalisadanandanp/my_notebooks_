import gym
from gym.wrappers.record_video import RecordVideo
import torch
import torch.nn as nn
from torchsummary import summary

# https://gymnasium.farama.org/environments/classic_control/cart_pole/

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = RecordVideo(env, "Deep learning/Transformer_models/codes/tutorial_03/DQN/untrained-cartpole" , episode_trigger = lambda episode_number: True)

# import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"

print("\n Action Space:", env.action_space, "\n")
print("\n Observation Space:", env.observation_space, "\n")

for episode in range(10):

    env.reset()

    for i in range(100):
        env.step(env.action_space.sample())
        # env.render()

env.close()


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.n_observations = n_observations
        self.n_actions = n_actions

        self.layer_1 = nn.Linear(self.n_observations, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, self.n_actions)

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.layer_3(x)
        #x = self.softmax(x)
        return x
    
    def print_summary(self):

        print(summary(self, (self.n_observations,)))

        print("\n Number of parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")

        print("\n Number of layers: ", len(list(self.parameters())), "\n")

# load the model
policy_network = DQN(4, 2)
policy_network.load_state_dict(torch.load("Deep learning/Transformer_models/codes/tutorial_03/DQN/policy_net.pth"))

# test the trained model
env = gym.make('CartPole-v1', render_mode="rgb_array")
env = RecordVideo(env, "Deep learning/Transformer_models/codes/tutorial_03/DQN/trained-cartpole" , episode_trigger = lambda episode_number: True)

state, info = env.reset()
done = False

while not done:
    action = policy_network(torch.tensor(state)).argmax().item()
    next_state, reward, done, terminal, info = env.step(action)
    state = next_state
    
    if terminal:
        break

