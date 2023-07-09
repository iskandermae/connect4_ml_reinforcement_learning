import gymnasium as gym
from stable_baselines3 import PPO

#from ConnectFourGymEnv import ConnectFourGymEnv
from MyEnvironmentGymConnect4 import MyEnvironmentGymConnect4
from kaggle_environments import make, evaluate
import numpy as np

    
def learn(neural_net, policyName, agents2):
    # Create ConnectFour environment 
    env_train = MyEnvironmentGymConnect4(agents2)

    policy_kwargs = dict(
        features_extractor_class=neural_net,
    )

    # Initialize agent
    model = PPO(policyName, env_train, policy_kwargs=policy_kwargs, verbose=0)

    # Train agent
    model.learn(total_timesteps=5)

    return model

def model_to_agent(model):
    def agent_local(obs, config):
        observation = np.array(obs['board']) \
            .reshape(1,6,7)
        action, _ = model.predict(observation, deterministic=True)
        print(action.item())
        return action.item()
    return agent_local

def play_and_draw(model, agents2):
    env_train = MyEnvironmentGymConnect4(agents2)
    from stable_baselines3.common.vec_env import (DummyVecEnv)
    env = DummyVecEnv([lambda: env_train])
    play3times(model_to_agent(model), env, agent2 = "random")
    
def play3times(agent1, env_train, agent2 = "random"):
    # Create the game environment
    env = make("connectx")
    # Two random agents play one game round
    env.run([agent1, agent2])
    # Show the game
    print(env.specification['agents'])
    print([agent1, agent2],env.state[0].reward,env.state[0].status,env.state[1].reward,env.state[1].status)
    print(np.array(env.state[0].observation['board']).reshape(1,6,7))

    env.reset
    env.run([agent2, agent1])
    # Show the game
    print(env.specification['agents'])
    print([agent2, agent1],env.state[0].reward,env.state[0].status,env.state[1].reward,env.state[1].status)
    print(np.array(env.state[0].observation['board']).reshape(1,6,7))

