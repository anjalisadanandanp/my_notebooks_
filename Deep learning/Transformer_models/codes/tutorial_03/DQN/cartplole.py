import gymnasium as gym
from gym.wrappers.record_video import RecordVideo

# https://gymnasium.farama.org/environments/classic_control/cart_pole/

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = RecordVideo(env, "./gym-results" , episode_trigger = lambda episode_number: True)

# import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"

print("\n Action Space:", env.action_space, "\n")
print("\n Observation Space:", env.observation_space, "\n")

for episode in range(10):

    env.reset()

    for i in range(100):
        env.step(env.action_space.sample())
        #env.render()

env.close()