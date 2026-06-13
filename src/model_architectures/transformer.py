
import torch
import torch.nn as nn
from torch.distributions import Categorical

"""
Logic behind using attention:
every board square can attend to one another
kinda makes sense
"""

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim) -> None:
        super().__init__()
        pass
    
    def forward(self, x, action_mask):
        pass
    
    def select_action(self, state):
        pass
    
    def evaluate_actions(self, states, actions):
        pass
    
def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    pass