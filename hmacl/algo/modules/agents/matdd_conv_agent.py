"""Convolutional Pursuit student described in the MATDD manuscript."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MATDDConvAgent(nn.Module):
    """Two 3x3 convolutions followed by the paper's recurrent policy stack."""

    def __init__(self, input_shape, args):
        super().__init__()
        self.obs_shape = tuple(args.obs_shape)
        if len(self.obs_shape) != 3:
            raise ValueError("MATDDConvAgent requires a 3D spatial observation")
        self.observation_features = math.prod(self.obs_shape)
        self.auxiliary_features = int(input_shape) - self.observation_features
        if self.auxiliary_features < 0:
            raise ValueError("controller input is smaller than the observation")

        channels = int(getattr(args, "matdd_conv_channels", 16))
        self.recurrent_dim = int(getattr(args, "matdd_recurrent_dim", 256))
        projection_dim = int(getattr(args, "matdd_projection_dim", 128))
        input_channels = self.obs_shape[-1]
        self.convolution = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        conv_height = self.obs_shape[0] - 4
        conv_width = self.obs_shape[1] - 4
        if conv_height <= 0 or conv_width <= 0:
            raise ValueError("observation is too small for two 3x3 convolutions")
        conv_features = channels * conv_height * conv_width
        encoder_dim = int(getattr(args, "matdd_encoder_dim", 256))
        self.fc1 = nn.Linear(conv_features + self.auxiliary_features, encoder_dim)
        self.rnn = nn.GRUCell(encoder_dim, self.recurrent_dim)
        self.fc2 = nn.Linear(self.recurrent_dim, projection_dim)
        self.output = nn.Linear(projection_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.recurrent_dim)

    def forward(self, inputs, hidden_state):
        observation = inputs[:, : self.observation_features]
        observation = observation.reshape(-1, *self.obs_shape).permute(0, 3, 1, 2)
        spatial = self.convolution(observation).reshape(observation.size(0), -1)
        if self.auxiliary_features:
            spatial = torch.cat(
                [spatial, inputs[:, self.observation_features :]], dim=-1
            )
        encoded = F.relu(self.fc1(spatial))
        hidden = self.rnn(encoded, hidden_state.reshape(-1, self.recurrent_dim))
        projected = F.relu(self.fc2(hidden))
        return self.output(projected), hidden
