
import torch
import torch.nn as nn
from torch.distributions import Categorical

"""
Maximize benefits of both cnn and a transformer
conv -> transformer -> a/c heads
"""

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim) -> None:
        super().__init__()
        
        pass
    
    def forward(self, x, action_mask):
        pass
    
    def select_action(self, state, action_mask):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        state_t = state_t.unsqueeze(0) # add batch dim
        logits, value = self(state_t, action_mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return (action.item(), log_prob.detach(), value.detach())
    
    def evaluate_actions(self, states, actions, old_action_masks):
        states_t = torch.as_tensor(states, dtype=torch.float32)
        logits, values = self(states_t, old_action_masks)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(torch.as_tensor(actions, dtype=torch.long))
        entropy = dist.entropy()
        return (new_log_probs, entropy, values)
    
def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    advantages = []
    values = values + [last_value]
    gae = 0.0
    for t in reversed(range(len(rewards))): # this goes backwards thru time
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    values_t = torch.as_tensor(values, dtype=torch.float32)
    returns = advantages + values_t
    advantages = (advantages - advantages.mean())/(advantages.std() + 1e-9) # normalize
    return (advantages, returns)