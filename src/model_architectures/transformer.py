
import torch
import torch.nn as nn
from torch.distributions import Categorical

"""
Logic behind using attention:
every board square can attend to one another
kinda makes sense
"""

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256) -> None:
        super().__init__()
        
        # convert 111 features to the 64 dmodel (hardcoded)
        input_dim = state_dim[-1]
        self.projection = nn.Linear(input_dim, 64)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, 
            nhead=8, 
            dim_feedforward=hidden_dim,
            activation='gelu',
            batch_first=True
        )
        self.positional_embedding = nn.Parameter(torch.zeros(1, 64, 64))

        # just an encoder for move prediction/value estimation
        # no decoder bc we're not generating stuff
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                         num_layers=2)
        
        # heads
        self.actor_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, action_dim)
        )

        self.critic_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 1)
        )
    
    def forward(self, x, action_mask):
        x = x.flatten(1, 2) # (b, 8, 8, 111) to (b, 64, 111)
        x = self.projection(x) # (b, 64, 111) to (b, 64, 64)
        x = x + self.positional_embedding # the b and 1 dims get casted to b
        x = self.transformer_encoder(x)
        logits = self.actor_head(x)
        logits = logits.masked_fill(torch.as_tensor(action_mask, device=logits.device) == 0, -1e9)
        return (logits, self.critic_head(x).squeeze(-1))
    
    def select_action(self, state, action_mask):
        device = next(self.parameters()).device
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        state_t = state_t.unsqueeze(0) # add batch dim
        logits, value = self(state_t, action_mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return (action.item(), log_prob.detach(), value.detach())
    
    def evaluate_actions(self, states, actions, old_action_masks):
        states_t = torch.as_tensor(states, dtype=torch.float32, device=next(self.parameters()).device)
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
        delta = rewards[t] + gamma * -values[t + 1] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * -gae
        advantages.insert(0, gae)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    values_t = torch.as_tensor(values[:-1], dtype=torch.float32)
    returns = advantages + values_t
    advantages = (advantages - advantages.mean())/(advantages.std() + 1e-9) # normalize
    return (advantages, returns)