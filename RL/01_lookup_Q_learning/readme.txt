
We are working with a scenario of Reinforcement Learning where we are applying the letter O as a wanderer.

That wanderer wants to get the treasure T as fast as it can.

The condition looks like this:
O-----T

(O P Q R S T)


The wanderer tries to find the quickest path to reach the treasure. 
During each Episode, the steps the wanderer takes to reach the treasure are counted. 
With each episode, the condition improves and the number of steps declines.



Here are some of the basic steps in terms of Reinforcement Learning:

# The program tries to work with actions, as actions are very important in terms of Reinforcement Learning.
# The available actions for this wanderer is moving left or right: ACTIONS = ['left','right']
# The wanderer can be considered the agent.
# The number of states (also called the number of steps) is limited to 6 in this example: N_States = 6


HYPERPARAMETERS:
# Epsilon is the greedy factor (Exploration versus Exploitation)
# Alpha is the learning rate
# Gamma is the discount factor

The maximum number of episodes in this case is MAX_EPISODES. 
The refresh rate is when the scenario is refreshed.