from MyEnvironmentGymConnect4 import MyEnvironmentGymConnect4
from UtilsModule import play3times, model_to_agent
from CustomCNN import CustomCNN
from stable_baselines3 import PPO
from UtilsModule import play3times, learn, play_and_draw,print_evaluation
from my_heuristic_agent import my_heuristic_agent


def play_and_draw(model, model2):
    agent2=model_to_agent(model2)
    env_train = MyEnvironmentGymConnect4([agent2])
    from stable_baselines3.common.vec_env import (DummyVecEnv)
    env = DummyVecEnv([lambda: env_train])
    play3times(model_to_agent(model), env, agent2)


def play_and_draw2(model):
    agent2=model_to_agent(my_heuristic_agent)
    env_train = MyEnvironmentGymConnect4([agent2])
    from stable_baselines3.common.vec_env import (DummyVecEnv)
    env = DummyVecEnv([lambda: env_train])
    play3times(model_to_agent(model), env, my_heuristic_agent)

load_file = "ppo_v2"
model = PPO.load(load_file)
model.set_env(MyEnvironmentGymConnect4(["random"]))

print_evaluation(model)

play_and_draw2(model)

##model2 = PPO.load("ppo_v2")
#play_and_draw(model, model2)
