import random
import numpy as np
import gymnasium as gym
#import gym
from gymnasium import spaces
from kaggle_environments import make 

class MyEnvironmentGymConnect4(gym.Env):
    def __init__(self, agents2=["random"]):
        super(MyEnvironmentGymConnect4, self).__init__()
        self.ks_env = make("connectx", debug=True)
        self.agents2=agents2
        self.rows = self.ks_env.configuration.rows
        self.columns = self.ks_env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)                                   # one of the gym spaces (Discrete, Box, ...) that describes the action space, so the type of action that can be taken
        self.observation_space = spaces.Box(low=0, high=2,                                  
                                            shape=(1,self.rows,self.columns), dtype=int)    # one of the gym spaces (Discrete, Box, ...) and describe the type and shape of the observation
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)                                                    
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self,*,seed: int | None = None):
        super().reset(seed=seed)
        agent2 = random.choice(self.agents2)
        agents_in_random_order = ([None, agent2], [agent2, None])[random.randint(0,1) == 0]
        #print('agents_in_random_order:', agents_in_random_order)                                           #todo: disable debug message
        self.env =  self.ks_env.train(agents_in_random_order)
        self.obs = self.env.reset()
        info = {}
        observation = np.array(self.obs['board']) \
            .reshape(1,self.rows,self.columns)
        return observation, info
    def change_reward(self, old_reward, done):
        if old_reward == 1: # The agent won the game
            return 1
        elif done: # The opponent won the game
            return -1
        else: # Reward 1/42
            return 1/(self.rows*self.columns)
    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid: # Play the move
            self.obs, old_reward, done, info = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else: # End the game and penalize agent
            reward, done, info = -10, True, {}
        truncated = False  # we do not limit the number of steps here
        if done and reward<-1:
            print("Done. Reward: ", reward)
        return np.array(self.obs['board']).reshape(1,self.rows,self.columns), reward, done, truncated, info
    def render(self):
        if self.render_mode == "console":
            self.render_console()
    def render_console(self):
        for row in np.array(self.obs['board']).reshape(self.rows,self.columns):
            print(row)