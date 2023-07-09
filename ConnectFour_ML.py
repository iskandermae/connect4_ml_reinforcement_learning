
# One-Step Lookahead: https://www.kaggle.com/code/alexisbcook/one-step-lookahead
# Creating a custom Gym environment: https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/5_custom_gym_env.ipynb
# how to submit pretrained model : https://www.kaggle.com/competitions/connectx/discussion/397021
# Reinforcement Learning Implementation : https://github.com/DLR-RM/stable-baselines3 (doc: https://stable-baselines3.readthedocs.io/)
# RL tutorial : https://github.com/araffin/rl-tutorial-jnrr19
# How to save/load model: https://araffin.github.io/post/sb3/


from CustomCNN import CustomCNN
from stable_baselines3 import PPO
from UtilsModule import play3times, learn, play_and_draw

model = PPO.load("ppo_v1")
play_and_draw(model, agents2=["random"])



policyName = "CnnPolicy"

learn_mode = True
learn_mode = False
if learn_mode:
    model = learn(CustomCNN, policyName, agents2=["random"])
    # Save the model
    model.save("ppo_v1")
    # save net (if no need training more)
    policy = model.policy
    policy.save("ppo_policy_v1.pkl")

# Load the trained model
model = PPO.load("ppo_v1")
model.policy.save("ppo_policy_v1.pkl")

#Load net
   #from stable_baselines3.sac.policies import CnnPolicy
   #saved_policy = CnnPolicy.load("ppo_policy_v1.pkl")
# ot another load way
saved_policy = PPO.policy_aliases[policyName].load("ppo_policy_v1.pkl")

from MyEnvironmentGymConnect4 import MyEnvironmentGymConnect4
ev_env_train = MyEnvironmentGymConnect4(["random"])
from stable_baselines3.common.vec_env import (DummyVecEnv)
ev_env = DummyVecEnv([lambda: ev_env_train])

from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, ev_env, n_eval_episodes=10)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

mean_reward, std_reward = evaluate_policy(saved_policy, ev_env, n_eval_episodes=10)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")



play_and_draw(model, agents2=["random"])


# FOR SUBMIT - without learning capabilities
#def agent22(obs, config):
#    action, _ = saved_policy.predict(obs, deterministic=True)
#    return action

