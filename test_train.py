
# One-Step Lookahead: https://www.kaggle.com/code/alexisbcook/one-step-lookahead
# Creating a custom Gym environment: https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/5_custom_gym_env.ipynb
# how to submit pretrained model : https://www.kaggle.com/competitions/connectx/discussion/397021
# Reinforcement Learning Implementation : https://github.com/DLR-RM/stable-baselines3 (doc: https://stable-baselines3.readthedocs.io/)
# RL tutorial : https://github.com/araffin/rl-tutorial-jnrr19
# How to save/load model: https://araffin.github.io/post/sb3/

from BigCustomCNN import BigCustomCNN
from CustomCNN import CustomCNN
from CustomCNNSmall import CustomCNNSmall
from stable_baselines3 import PPO
from UtilsModule import play3times, play_and_draw,print_evaluation,create_model,policyName, get_tensorboard_logdir

from MyEnvironmentGymConnect4 import MyEnvironmentGymConnect4
from my_heuristic_agent import my_heuristic_agent

logdir = get_tensorboard_logdir()

load_file = ""
load_file = "ppo_v0"
if load_file != "":
    env_train = MyEnvironmentGymConnect4(["random"])
    model = PPO.load(load_file, env_train)
else:
    model = create_model(CustomCNN, policyName, ["random"], logdir) # to view the logs start: tensorboard --logdir=./logs

#model.set_env(MyEnvironmentGymConnect4([my_heuristic_agent]))

policyName = "CnnPolicy"

learn_mode = True
#learn_mode = False
if learn_mode:
    for i in range(1):
        model.learning_rate = 0.0001
        model.learn(total_timesteps=20000, tb_log_name="PPO")
        model.save(f"ppo_v{i+1}")
        print_evaluation(model)

