from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gym
from gym import spaces
import torch as th
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy



class LOStransformer(BaseFeaturesExtractor):
    # def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, dim_input=64, num_outputs=1, dim_output=3, dim_hidden=64, num_heads=4):
        # super(LOStransformer, self).__init__(observation_space, features_dim)
        # # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        super(LOStransformer, self).__init__(dim_input, num_outputs, dim_output)
        n_input_channels = observation_space.shape[0]
#         self.enc = nn.Sequential(
#                 SAB(dim_input, dim_hidden, num_heads),
#                 SAB(dim_hidden, dim_hidden, num_heads))
        encoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=4, dim_feedforward=2*dim_hidden, dropout=0.0)
        decoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=4, dim_feedforward=2*dim_hidden, dropout=0.0)
        self.feat_in = nn.Sequential(
                        nn.Linear(dim_input, dim_hidden),
                    )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.dec = nn.Sequential(
#                 PMA(dim_hidden, num_heads, num_outputs),
#                 SAB(dim_hidden, dim_hidden, num_heads),
#                 SAB(dim_hidden, dim_hidden, num_heads),
#                 nn.Linear(dim_hidden, dim_output))
        self.pool = PMA(dim_hidden, num_heads, num_outputs)
        self.dec = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.feat_out = nn.Sequential(
                    nn.Linear(dim_hidden, dim_output)
                    )

    def forward(self, x, pad_mask=None):
        x = self.feat_in(x)
        x = self.enc(x, src_key_padding_mask=pad_mask)
        x = self.pool(x, src_key_padding_mask=pad_mask)
        x = self.dec(x)
        out = self.feat_out(x)
        return torch.squeeze(out, dim=0)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(th.Tensor(num_seeds, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = nn.MultiheadAttention(dim, num_heads)

    def forward(self, X, src_key_padding_mask=None):
        Q = self.S.repeat(1, X.size(1), 1)
        out, _ = self.mab(Q, X, X, key_padding_mask=src_key_padding_mask)
        return out

class CustomMLP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        # self.mlp = nn.Sequential(
        #     nn.Linear(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        self.flatten=nn.Sequential(nn.Flatten())
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.flatten(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.mlp = nn.Sequential(nn.Flatten(),nn.Linear(n_flatten, features_dim), nn.ReLU())


        # self.linear = nn.Sequential(nn.Linear(n_input_channels, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations) #self.linear(self.mlp(observations))

class CustomLSTM(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomLSTM, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        tmp=observation_space.sample()[None]

        # # for multi-input e.g., 13*4 LSTM output 1*13*features_dim
        # n_input_channels = observation_space.shape[-1]
        # self.lstmmulti = nn.Sequential(nn.LSTM(n_input_channels, features_dim, 2))
        # with th.no_grad():
        #     out_lstmmulti, hiddenmulti = self.lstmmulti(
        #         th.as_tensor(tmp).float()
        #     )

        # LSTM containing sigmoid and tanh
        # for single input LSTM output 1*features_dim
        self.flatten=nn.Sequential(nn.Flatten())
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.flatten(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # self.lstm = nn.Sequential(nn.LSTM(n_flatten, features_dim))
        self.lstm = nn.Sequential(nn.Flatten(),nn.LSTM(n_flatten, features_dim, 2))
        with th.no_grad():
            out_lstm, hidden = self.lstm(
                th.as_tensor(tmp).float()
            )

        # self.linear = nn.Sequential(nn.Linear(n_input_channels, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.lstm(observations)[0] #self.linear(self.mlp(observations))

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations)) #self.mlp(observations) #

from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 64):
        super().__init__(observation_space, features_dim=features_dim)

        # if is_image_space(observation_space):
        #     raise ValueError("LSTMFeatureExtractor only works with non-image spaces")

        self.lstm_hidden_size = features_dim

        tmp=observation_space.sample()[None]
        self.flatten=nn.Sequential(nn.Flatten())
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.flatten(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # self.lstm = nn.Sequential(nn.LSTM(n_flatten, features_dim))
        # self.lstm = nn.Sequential(nn.Flatten(),nn.LSTM(n_flatten, features_dim, 2))
        self.lstm = nn.Sequential(nn.Flatten(),
                                  nn.LSTM(n_flatten, features_dim, batch_first=True))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(observations)
        features = x#[-1]#[:, -1, :]

        return features
