import tkinter as tk
import time
import random
import pandas as pd
import numpy as np

MAZE_HEIGHT = 6
MAZE_WIDTH = 6
UNIT = 50

class Maze(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('maze')
        self.actions = ["up", "down", "left", "right"]
        self._built_maze()

    def _built_maze(self):

        #create canvas
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_HEIGHT*UNIT, width=MAZE_WIDTH*UNIT)

        #create grids
        for c in range(0, MAZE_WIDTH+1):
            self.canvas.create_line(UNIT*c, 0, UNIT*c, MAZE_HEIGHT*UNIT)

        for c in range(0, MAZE_HEIGHT+1):
            self.canvas.create_line(0,UNIT*c, MAZE_WIDTH*UNIT, UNIT*c)

        #create blocks
        self.block1 = self.canvas.create_rectangle(UNIT*2, UNIT, UNIT*3, UNIT*2, fill='black')
        self.block2 = self.canvas.create_rectangle(UNIT*1, UNIT*3, UNIT*2, UNIT*4, fill='black')
        self.block2 = self.canvas.create_rectangle(UNIT*2, UNIT*4, UNIT*3, UNIT*5, fill='black')

        self.goal = self.canvas.create_rectangle(UNIT*(MAZE_WIDTH-1), UNIT*(MAZE_HEIGHT-1), UNIT*MAZE_WIDTH, UNIT*MAZE_HEIGHT, fill='green')
        self.player = self.canvas.create_oval(UNIT*0, UNIT*0, UNIT*1, UNIT*1, fill='red')

        self.player_x = 0
        self.player_y = 0

        self.goal_x = MAZE_WIDTH-1
        self.goal_y = MAZE_HEIGHT-1

        self.canvas.pack(padx = 100, pady= 100)

        # print("Player start position: ", self.player_x, self.player_y)
        # print("Goal position: ", self.goal_x, self.goal_y)

        return
    
    def step(self, action):

        old_x, old_y = self.player_x, self.player_y

        if action == "up":
            if self.player_y == 0:
                #print("Can't go up")
                pass
            else:
                self.player_y -= 1

        elif action == "down":
            if self.player_y == (MAZE_HEIGHT-1):
                #print("Can't go down")
                pass
            else:
                self.player_y += 1

        elif action == "left":
            if self.player_x == 0:
                #print("Can't go left")
                pass
            else:
                self.player_x -= 1

        elif action == "right":
            if self.player_x == (MAZE_WIDTH-1):
                #print("Can't go right")
                pass
            else:
                self.player_x += 1

        #print("New player position: ", self.player_x, self.player_y)
 
        self.canvas.move(self.player, (self.player_x-old_x)*UNIT, (self.player_y-old_y)*UNIT)

        if self.player_x == self.goal_x and self.player_y == self.goal_y:
            #print("You win!")
            reward = 1
            done = True
        
        elif self.player_x == 2 and self.player_y == 1:
            reward = -1
            done = True

        elif self.player_x == 1 and self.player_y == 3:
            reward = -1
            done = True

        elif self.player_x == 2 and self.player_y == 4:
            reward = -1
            done = True

        else:
            #print("AT THE EDGE")
            reward = 0
            done = False

        return (self.player_x, self.player_y, reward, done)

    def render(self):
        #time.sleep(0.5)
        self.update()

    def reset(self):
        #time.sleep(0.5)
        self.canvas.delete(self.player)
        self.player = self.canvas.create_oval(UNIT*0, UNIT*0, UNIT*1, UNIT*1, fill='red')
        self.player_x = 0
        self.player_y = 0
        self.update()
        #time.sleep(0.5)
        done = False
        return self.player_x, self.player_y, done








class learning():

    def __init__(self, alpha, gamma, epsilon):

        self.actions = ["up", "down", "left", "right"]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        #initialise Q-table
        self.q_table = pd.DataFrame(np.zeros((MAZE_HEIGHT*MAZE_WIDTH, len(self.actions))), columns=self.actions)

    def choose_action_e_greedy(self, state):

        player_x = state[0]
        player_y = state[1]

        state_id = player_x + player_y*MAZE_WIDTH

        if np.random.rand() < self.epsilon:
            #choose best action
            maxq_action_id = self.q_table.loc[state_id, :].argmax()
            greedy_action = self.actions[maxq_action_id]
            return greedy_action

        else:
            #choose random action
            random_action = np.random.choice(self.actions)
            return random_action
        
    def q_learn(self, state, action, reward, next_state):

        player_x = state[0]
        player_y = state[1]
        state_id = player_x + player_y*MAZE_WIDTH

        player_x = next_state[0]
        player_y = next_state[1]
        next_state_id = player_x + player_y*MAZE_WIDTH

        self.q_table.loc[state_id, action] += self.alpha*(reward + self.gamma * (self.q_table.loc[next_state_id].max()) - self.q_table.loc[state_id, action])

        return
    
    def SARSA(self, state, action, reward, next_state, next_action):

        player_x = state[0]
        player_y = state[1]
        state_id = player_x + player_y*MAZE_WIDTH

        player_x = next_state[0]
        player_y = next_state[1]
        next_state_id = player_x + player_y*MAZE_WIDTH

        self.q_table.loc[state_id, action] += self.alpha*(reward + self.gamma * (self.q_table.loc[next_state_id, next_action]) - self.q_table.loc[state_id, action])

        return



def update_qlearn():

    MAX_EPISODES = 10000

    instance = learning(alpha=0.1, gamma=0.5, epsilon=0.5)

    for episode in range(1, MAX_EPISODES+1):
        print("EPISODE: ", episode)
        x, y, done = env.reset()
        env.render()

        instance.state = (x,y)

        while not done:
            action = instance.choose_action_e_greedy((x,y))
            #print(action)
            x_new, y_new, reward, done = env.step(action)
            env.render()
            instance.q_learn((x,y), action, reward, (x_new, y_new))
            x = x_new
            y = y_new

    env.destroy()

    q_table = instance.q_table
    print(q_table)

    return 


def update_SARSA():

    MAX_EPISODES = 1000

    instance = learning(alpha=0.1, gamma=0.5, epsilon=0.5)

    for episode in range(1, MAX_EPISODES+1):
        print("EPISODE: ", episode)
        x, y, done = env.reset()
        env.render()

        instance.state = (x,y)

        while not done:
            action = instance.choose_action_e_greedy((x,y))
            x_new, y_new, reward, done = env.step(action)
            env.render()
            next_action = instance.choose_action_e_greedy((x_new,y_new))
            instance.SARSA((x,y), action, reward, (x_new, y_new), next_action)
            x = x_new
            y = y_new

    env.destroy()

    q_table = instance.q_table
    print(q_table)

    return 





if __name__ == '__main__':
    env = Maze()
    q_table = env.after(100, update_SARSA)
    env.mainloop()
    del env




