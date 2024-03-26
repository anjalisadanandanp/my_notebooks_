# Solving the taxi problem using Q learning

To demonstrate the problem let's say our agent is the driver. 
There are four locations and the agent has to pick up a passenger at one location and drop them off at another. 
The agent will receive +20 points as a reward for successful drop off and -1 point for every time step it takes. 
The agent will also lose -10 points for illegal pickups and drops. 
So the goal of our agent is to learn to pick up and drop off passengers at the correct location in a short time without adding illegal passengers.
