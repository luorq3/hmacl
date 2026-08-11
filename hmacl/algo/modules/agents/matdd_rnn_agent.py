"""Student network described for Football in the MATDD manuscript."""

import torch.nn as nn
import torch.nn.functional as F


class MATDDRNNAgent(nn.Module):
    """FC(256) -> GRU(256) -> FC(128) -> action outputs."""

    def __init__(self, input_shape, args):
        super().__init__()
        self.encoder_dim = int(getattr(args, "matdd_encoder_dim", 256))
        self.recurrent_dim = int(getattr(args, "matdd_recurrent_dim", 256))
        self.projection_dim = int(getattr(args, "matdd_projection_dim", 128))
        self.fc1 = nn.Linear(input_shape, self.encoder_dim)
        self.rnn = nn.GRUCell(self.encoder_dim, self.recurrent_dim)
        self.fc2 = nn.Linear(self.recurrent_dim, self.projection_dim)
        self.output = nn.Linear(self.projection_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.recurrent_dim)

    def forward(self, inputs, hidden_state):
        encoded = F.relu(self.fc1(inputs))
        hidden = self.rnn(encoded, hidden_state.reshape(-1, self.recurrent_dim))
        projected = F.relu(self.fc2(hidden))
        return self.output(projected), hidden
