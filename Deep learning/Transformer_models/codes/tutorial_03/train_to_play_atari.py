import gym
import torch
import torch.nn as nn
from torchsummary import summary
import argparse
import sys
import random
import wandb
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np


config = dict()
config["learning_rate"] = 5e-5
config["max_sequence_length"] = 100     #max number of steps in a game
config["truncate"] = True
config["max_steps_truncate"] = 10       #context length
config["d_model"] = 64*3
config["d_in"] = 64*3
config["dropout"] = 0.1
config["validation_steps"] = 100


# start a new wandb run to track this script
run = wandb.init(
    # set the wandb project where this run will be logged
    project="cartpole",
    # track hyperparameters and run metadata
    config=config,
)

sys.path.append('/mnt/data2/Deep learning/Transformer_models/codes/tutorial_03')


from decision_transformer import DecisionTransformer


parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='CartPole-v1')
parser.add_argument('--n_observations', type=int, default=4)
parser.add_argument('--n_actions', type=int, default=2)
parser.add_argument('--epsilon', type=float, default=0)       #exploration rate
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--n_games', type=int, default=100)
parser.add_argument('--n_epoches', type=int, default=100)




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

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.layer_3(x)
        return x
    
    def print_summary(self):

        print(summary(self, (self.n_observations,)))

        print("\n Number of parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")

        print("\n Number of layers: ", len(list(self.parameters())), "\n")



    
class Play_Games():

    def __init__(self, game, n_observations, n_actions, epsilon, gamma):

        self.game = game
        self.actions = n_actions
        self.observations = n_observations
        self.epsilon = epsilon
        self.gamma = gamma

        # load the trained model
        self.policy_network = DQN(4, 2)
        self.policy_network.load_state_dict(torch.load("Transformer_models/codes/tutorial_03/DQN/policy_net.pth"))
        self.policy_network.print_summary()

    def select_action(self, state, epsilon):

        if random.random() > epsilon:
            #print("Exploitation")
            with torch.no_grad():
                return self.policy_network(state.clone().detach()).argmax()
        else:
            #print("Exploration")
            return torch.tensor(random.randrange(self.actions), dtype=torch.long)
        
    def calc_rewards_to_go(self, rewards, gamma):

        rewards_to_go = []
        reward = 0
        for r in reversed(rewards):
            reward = r + gamma*reward
            rewards_to_go.insert(0, reward)

        return rewards_to_go

    def play_game(self, seed, render=False, num_steps=config["max_sequence_length"]):

        # create the environment
        env = gym.make(self.game)
        #env.seed(seed)

        #reset the environment
        observation = env.reset()

        OBSERVATION = []
        REWARD = []
        ACTION = []
        STEPS = []

        step = 0

        while True:

            STEPS.append(step)
            #convert the observation to tensor
            observation = torch.tensor(observation, dtype=torch.float32)    
            OBSERVATION.append(observation)

            #render the environment
            if render:
                env.render()

            #get the action from the trained model
            action = self.select_action(state=observation, epsilon=self.epsilon).item()
            ACTION.append(action)

            #perform the action
            observation, reward, done, info = env.step(action)
            REWARD.append(reward)

            step += 1
    
            #check if the game is over
            if done:
                break

            #check if the number of steps is greater than the max steps
            if step > num_steps - 1:
                break
        
        env.close()

        return OBSERVATION, ACTION, REWARD, STEPS
    
    def play_multiple_games(self, n_games=10):
            
        self.GAME_STATES = []
            
        for i in range(1, n_games+1):
            #print("Playing game number: ", i)
            OBSERVATION, ACTION, REWARD, STEPS = self.play_game(seed=i, render=False)
            REWARD_TO_GO = self.calc_rewards_to_go(REWARD, self.gamma)
            game_states = zip(STEPS, OBSERVATION, ACTION, REWARD_TO_GO)
            self.GAME_STATES.append(game_states)

        return self.GAME_STATES





class embedding_layer(nn.Module):

    def __init__(self, d_in, d_out):
        super(embedding_layer, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=True)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
    



class time_step_encoder(nn.Module):

    def __init__(self, max_sequence_length, d_model):
        super(time_step_encoder, self).__init__()
        self.layer = nn.Embedding(max_sequence_length, d_model)

    def forward(self, time_steps):
        embedd = self.layer(time_steps)
        return embedd




def subsequent_mask(size):
    """ Mask out subsequent positions """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    mask = mask == 0
    return mask.bool()




def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )





class Trainer():

    def __init__(self):

        #parse the arguments
        self.game = parser.parse_args().game
        self.n_observations = parser.parse_args().n_observations
        self.n_actions = parser.parse_args().n_actions
        self.epsilon = parser.parse_args().epsilon
        self.gamma = parser.parse_args().gamma
        self.n_games = parser.parse_args().n_games
        self.n_epoches = parser.parse_args().n_epoches

        self.d_model = config["d_model"] 
        self.d_in = config["d_in"]

        self.truncate = config["truncate"]
        self.max_steps = config["max_sequence_length"]
        self.max_steps_truncate = config["max_steps_truncate"]

        #create the decision transformer model
        self.DT = DecisionTransformer(d_model=self.d_model, 
                                      num_head=4, 
                                      d_ff=512, 
                                      num_layers=2, 
                                      dropout=config["dropout"], 
                                      vocab_size=None, 
                                      src_embed=embedding_layer(self.d_in, d_out=self.d_model), 
                                      tgt_embed=embedding_layer(self.d_in, d_out=self.d_model), 
                                      positional_encoding=time_step_encoder(max_sequence_length=self.max_steps, d_model=self.d_model//3))
        
        #initialize the weights
        # for p in self.DT.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

        #Binary Cross Entropy Loss
        self.loss = nn.BCELoss()

        #Cross Entropy Loss
        #self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.DT.parameters(), lr=config["learning_rate"])

        # self.scheduler = lr_scheduler.LambdaLR(
        #     optimizer=self.optimizer, 
        #     lr_lambda=lambda step: rate(step, model_size=config["d_model"] , factor=1.0, warmup=10)
        #     )

        resume = False
        if resume:
            self.DT.load_state_dict(torch.load("Transformer_models/codes/tutorial_03/trained_model/decision_transformer_200_.pth"))
        
    def train(self):

        self.play_games = Play_Games(game=self.game, n_observations=self.n_observations, n_actions=self.n_actions,
                                epsilon=self.epsilon, gamma=self.gamma)
        
        for epoch in range(1, self.n_epoches+1):

            self.DT.train()
        
            game_states = self.play_games.play_multiple_games(n_games = self.n_games + 1)

            for id, game_state in enumerate(game_states):  

                #unzip the game states
                steps_, observations_, actions_, rewards_to_go_ = zip(*game_state)

                for index in range(0, len(steps_)):

                    if self.truncate:

                        if index < self.max_steps_truncate:

                            steps = steps_[:index + 1]
                            observations = observations_[:index + 1]
                            actions_prev = actions_[:index]
                            actions_to_pred = actions_[:index + 1]
                            rewards_to_go = rewards_to_go_[:index + 1]

                            #pad zeros to the vectors to make them of equal length
                            if index < self.max_steps_truncate:

                                pad = [0 for _ in range(self.max_steps_truncate - len(steps))]
                                steps = pad + list(steps)
                                actions_to_pred = pad + list(actions_to_pred)
                                rewards_to_go = pad + list(rewards_to_go)

                                pad = [0 for _ in range(self.max_steps_truncate - len(actions_prev))]
                                actions_prev = pad + list(actions_prev)

                                pad = [torch.zeros(4) for _ in range(self.max_steps_truncate - len(observations))]
                                observations = pad + list(observations)

                        else:
                            steps = steps_[index - self.max_steps_truncate + 1:index + 1]
                            observations = observations_[index - self.max_steps_truncate + 1:index + 1]
                            actions_prev = actions_[index - self.max_steps_truncate:index]
                            actions_to_pred = actions_[index - self.max_steps_truncate + 1:index + 1]
                            rewards_to_go = rewards_to_go_[index - self.max_steps_truncate + 1:index + 1]

                    #convert the game states to tensors
                    step = torch.tensor(steps).unsqueeze(-1).unsqueeze(0)
                    observations = torch.stack(observations).unsqueeze(0)
                    actions_prev = torch.tensor(actions_prev, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
                    actions_to_pred = torch.tensor(actions_to_pred, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
                    rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)

                    batch_size, seq_len, d_model = observations.shape   
                    mask = subsequent_mask(seq_len)

                    hidden = self.DT.encode(observations, actions_prev, rewards_to_go, step, src_mask=None)   
                    out = self.DT.decode(observations, actions_prev, rewards_to_go, hidden, step, src_mask=None, tgt_mask=mask)      #self attention mask: causal, cross attention mask: None
                    action_pred  = self.DT.make_prediction(out)
                    action_loss = self.loss(action_pred[:, -1, :], actions_to_pred[:, -1, :])
                    
                    # print("Step:", index, "  Actions to predict: ", actions_to_pred[:, -1, :], "  Actions predicted: ", action_pred[:, -1, :])
                    self.optimizer.zero_grad()
                    action_loss.backward()
                    self.optimizer.step()

                    wandb.log({"loss": action_loss})
                    wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr']})

            # self.scheduler.step()

            #print the loss
            print("Game Play:", epoch, "    Action loss: ", action_loss.item(), "\n")

            #save the model
            torch.save(trainer.DT.state_dict(), "Transformer_models/codes/tutorial_03/trained_model/decision_transformer_" + str(epoch) + "_.pth")
            
            # test the model
            self.test()

    def test(self):

        REWARDS_TO_GO = [20, 40, 60, 80, 100]

        for REWARD in REWARDS_TO_GO:

            #print("Validating the model on replay with reward to go: ", REWARD)

            #validate the model
            length_of_game = []

            for game in range(1, config["validation_steps"]+1):

                seq_actions = [0]

                env = gym.make('CartPole-v1')
                #env.seed(0)
                state = env.reset()

                rewards_to_go = REWARD
                action = 0
                steps_count = 0
                context_length = config["max_steps_truncate"]

                steps_ = [steps_count]
                state_ = torch.tensor([state], dtype=torch.float32)
                actions_ = [action]
                rewards_to_go_ = [rewards_to_go]

                for index in range(0, REWARD+1):

                    #pad zeros to the vectors to make them of equal length
                    if index < self.max_steps_truncate:

                        pad = [0 for _ in range(self.max_steps_truncate - len(steps_))]
                        steps = pad + list(steps_)
                        actions = pad + list(actions_)
                        rewards_to_go = pad + list(rewards_to_go_)

                        pad = [torch.zeros(4) for _ in range(self.max_steps_truncate - len(state_))]
                        state = pad + list(state_)

                    else:
                        steps = steps_[index - self.max_steps_truncate:index]
                        state = state_[index - self.max_steps_truncate:index]
                        actions = actions_[index - self.max_steps_truncate:index]
                        rewards_to_go = rewards_to_go_[index - self.max_steps_truncate:index]

                        #convert to tuple
                        state = tuple(state)

                    for i,item in enumerate(state):
                        try:
                            if item.dtype != torch.float32:
                                state[i] = torch.tensor(item, dtype=torch.float32)
                        except:
                            pass

                    #convert the game states to tensors
                    steps = torch.tensor(steps).unsqueeze(-1).unsqueeze(0).int()
                    state = torch.stack(state).unsqueeze(0).float()
                    actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).float()
                    rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).float()

                    mask = subsequent_mask(context_length)

                    with torch.no_grad():
                        self.DT.eval()
                        hidden = self.DT.encode(state, actions, rewards_to_go, steps, src_mask=None)
                        output = self.DT.decode(state, actions, rewards_to_go, hidden, steps, src_mask=None, tgt_mask=mask)
                        action_pred = self.DT.make_prediction(output)

                    action_prob = action_pred[:, -1, :].item()

                    if action_prob > 0.5:
                        action_next = 1
                    else:
                        action_next = 0

                    observation, reward, done, info = env.step(action_next)

                    steps_count += 1

                    if (done == True) or (index == REWARD):
                        #print('Episode finished after {} timesteps'.format(steps_count))
                        length_of_game.append(steps_count)
                        break

                    #add observation to the state
                    state_ = torch.cat((state_, torch.tensor([observation], dtype=torch.float32)), dim=0)
                    # convert state_ to a list
                    # state_ = state_.tolist()

                    rewards_to_go_.append(rewards_to_go_[-1]-reward)
                    actions_.append(action_next)
                    steps_.append(steps_count)
                    seq_actions.append(action_next)

                env.close()

                # print("Sequence of actions predicted: ", seq_actions, "\n")
                # OBSERVATION, ACTION, REWARD, STEPS = self.play_games.play_game(render=True, seed=0, num_steps=length_of_game[-1])
                # print("Sequence of actions (true): ", ACTION, "\n")

            #calculate the average length of the game
            average_length_of_game = sum(length_of_game)/config["validation_steps"]

            #log the average length of the game
            #wandb.log({"average_length_of_game_on_replay_" + str(REWARD) : average_length_of_game})
            print("Average length of game on replay with reward to go: ", REWARD, " is: ", average_length_of_game, "\n")




trainer = Trainer()
trainer.train()




