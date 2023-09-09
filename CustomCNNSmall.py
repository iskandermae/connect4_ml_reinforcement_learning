from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gymnasium as gym

# Neural network for predicting action values
class CustomCNNSmall(BaseFeaturesExtractor):
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=7):
        super(CustomCNNSmall, self).__init__(observation_space, features_dim)
        # CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 1024, kernel_size=4, stride=1, padding=0),
            nn.Mish(),
            nn.Conv2d(1024, 8384, kernel_size=3, stride=1, padding=0),
            nn.Mish(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten+7, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        additional_inputs = observations[:,0,0,:]
        x = th.cat((self.cnn(observations), additional_inputs), dim=1)
        return self.linear(x)