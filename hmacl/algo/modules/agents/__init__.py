from .matdd_conv_agent import MATDDConvAgent
from .matdd_rnn_agent import MATDDRNNAgent
from .rnn_agent import RNNAgent
from .rnn_cs_agent import RNNCSAgent
from .rnn_ns_agent import RNNNSAgent

REGISTRY = {
    "matdd_conv": MATDDConvAgent,
    "matdd_rnn": MATDDRNNAgent,
    "rnn": RNNAgent,
    "rnn_cs": RNNCSAgent,
    "rnn_ns": RNNNSAgent,
}
