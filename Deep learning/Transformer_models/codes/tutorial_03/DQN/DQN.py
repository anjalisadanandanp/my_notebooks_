import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque
import random
from torchviz import make_dot
import gymnasium as gym
import math
from gym.wrappers.record_video import RecordVideo



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



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



class replay_buffer():

    def __init__(self, max_capacity):
        # Initialize the buffer
        self.buffer = deque([], maxlen=max_capacity)

    def push(self, *args):
        # Push a transition to the buffer
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        # Sample a batch of transitions from the buffer
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)



class Learning_model():

    def __init__(self, gamma, epsilon_start, epsilon_end, epsilon_decay, learning_rate, batch_size):

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.env = gym.make('CartPole-v1')
        # self.env = RecordVideo(self.env, "./gym-results" , episode_trigger = lambda episode_number: True)
        self.actions = self.env.action_space.n
        self.observations = self.env.observation_space.shape[0]

        self.policy_net = DQN(self.observations, self.actions)
        self.target_net = DQN(self.observations, self.actions)

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = replay_buffer(max_capacity=10000)

        # print("Policy network weights:", self.policy_net.state_dict())
        # print("Target network weights:", self.target_net.state_dict())

    def reset_env(self):
        return self.env.reset()

    def select_action(self, state, epsilon):

        if random.random() > epsilon:
            #print("Exploitation")
            with torch.no_grad():
                return self.policy_net(torch.tensor(state)).argmax()
        else:
            #print("Exploration")
            return torch.tensor(random.randrange(self.actions), dtype=torch.long)
        
    def update_epsilon(self, step):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * step / self.epsilon_decay)

    def optimize_model(self):

        if len(self.memory.buffer) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)

        batch_data_state = torch.tensor([sample.state for sample in transitions], dtype=torch.float32)
        batch_data_action = torch.tensor([sample.action for sample in transitions], dtype=torch.long)
        batch_data_reward = torch.tensor([sample.reward for sample in transitions], dtype=torch.float32)
        batch_data_next_state = torch.tensor([], dtype=torch.float32)
        
        terminal_mask = []

        for ind, sample in enumerate(transitions):
            # print("\n Sample: ", sample, "\n")
            # check if any value in the sample is None
            try:
                if (sample.next_state != None).any():
                    terminal_mask.append(False)
                    batch_data_next_state = torch.cat((batch_data_next_state, torch.tensor(sample.next_state, dtype=torch.float32).unsqueeze(0)), dim=0)

            except:   #If the next state is None, then the episode has ended
                # Create a terminal mask
                batch_data_next_state = torch.cat((batch_data_next_state, torch.tensor([0,0,0,0], dtype=torch.float32).unsqueeze(0)), dim=0)
                terminal_mask.append(True)


        # print("\n Batch data reward: ", batch_data_reward, "\n")
        # print("\n Batch data terminal state: ", batch_data_reward[terminal_mask], "\n")

        #Use policy network to predict Q(s,a): actual Q values predicted by the policy network
        Q_values_policy_net = self.policy_net(batch_data_state).gather(1, batch_data_action.unsqueeze(1)).view(-1)

        #Use target network to predict Q(s,a)
        Q_values_target_net = self.target_net(batch_data_next_state)
        Q_values_expected = batch_data_reward + torch.mul(Q_values_target_net.max(axis=1)[0], float(self.gamma))
        Q_values_expected[terminal_mask] = batch_data_reward[terminal_mask]
        
        # print("\n Expected Q values: ", Q_values_expected, "\n")
        # print("\n Predicted Q values: ", Q_values_policy_net, "\n")

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_values_expected, Q_values_policy_net)

        #print("\n Loss:", loss.item(), "\n")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        


        




Learning_model = Learning_model(gamma = 0.99, epsilon_start = 0.90, epsilon_end = 0.05, epsilon_decay = 1000, learning_rate = 1e-4, batch_size=128)
TAU = 0.005

# global_steps: calculate the number of steps taken in total across all episodes
# This is used to update the value of the parameter epsilon
global_steps = 0

for episode in range(600):

    print("Episode: ", episode)

    # Reset the environment
    state, info = Learning_model.reset_env()
    done = False

    # Reset the step counter within the episode
    step = 0

    while not done:

        epsilon = Learning_model.update_epsilon(global_steps)
        action = Learning_model.select_action(state, epsilon).item()
        next_state, reward, done, terminal, info = Learning_model.env.step(action)

        if done or terminal:
            next_state = None
            Learning_model.memory.push(state, action, next_state, reward)
            print("Episode finished after {} timesteps".format(step+1))
            break

        Learning_model.memory.push(state, action, next_state, reward)
        Learning_model.optimize_model()

        step += 1
        global_steps += 1

        # update the target network
        target_net_state_dict = Learning_model.target_net.state_dict()
        policy_net_state_dict = Learning_model.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        Learning_model.target_net.load_state_dict(target_net_state_dict)

        # update the state
        state = next_state

# save the model
torch.save(Learning_model.policy_net.state_dict(), "Deep learning/Transformer_models/codes/tutorial_03/DQN/policy_net.pth")
torch.save(Learning_model.target_net.state_dict(), "Deep learning/Transformer_models/codes/tutorial_03/DQN/target_net.pth")


# load the model
policy_network = DQN(4, 2)
policy_network.load_state_dict(torch.load("Deep learning/Transformer_models/codes/tutorial_03/DQN/policy_net.pth"))

# test the trained model
env = gym.make('CartPole-v1', render_mode="rgb_array")
env = RecordVideo(env, "codes/tutorials/05_DQN/cart-pole-trained" , episode_trigger = lambda episode_number: True)

state, info = env.reset()
done = False

while not done:
    action = policy_network(torch.tensor(state)).argmax().item()
    next_state, reward, done, terminal, info = env.step(action)
    state = next_state


