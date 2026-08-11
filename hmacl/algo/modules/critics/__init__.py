from .ac import ACCritic
from .ac_ns import ACCriticNS
from .centralV import CentralVCritic
from .centralV_ns import CentralVCriticNS
from .coma import COMACritic
from .coma_ns import COMACriticNS
from .dyna_central_v import DynaCentralVCritic
from .maddpg import MADDPGCritic
from .maddpg_ns import MADDPGCriticNS

REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["dyna_cv_critic"] = DynaCentralVCritic
REGISTRY["coma_critic_ns"] = COMACriticNS
REGISTRY["cv_critic_ns"] = CentralVCriticNS
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["maddpg_critic_ns"] = MADDPGCriticNS
REGISTRY["ac_critic"] = ACCritic
REGISTRY["ac_critic_ns"] = ACCriticNS
