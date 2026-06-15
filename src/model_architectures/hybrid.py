
import torch
import torch.nn as nn
from torch.distributions import Categorical

"""
Maximize benefits of both cnn and a transformer
conv -> transformer -> a/c heads
"""

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256) -> None:
        super().__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(state_dim[-1], # features = 111
                      64,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.SiLU()
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, 
            nhead=8, 
            dim_feedforward=hidden_dim,
            activation='gelu',
            batch_first=True
        )
        self.positional_embedding = nn.Parameter(torch.zeros(1, 64, 64))
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                         num_layers=2)
        
        self.actor_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, action_dim)
        )

        self.critic_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 1)
        )

        """
        Here the tensors flow like:
        (b, 8, 8, 111)
        (b, 8, 8, 64)
        (b, 64, 64) -- this is a flatten in forward for (1, 2)
        (b, 64 * 64)
        (b, action_dim OR 1)
        """
    
    def forward(self, x, action_mask):
        x = x.permute(0, 3, 1, 2) # turns into (b, 111, 8, 8)
        x = self.conv_layer(x) # (b, 111, 8, 8) -> (b, 64, 8, 8)
        x = x.flatten(2, 3).permute(0, 2, 1) # (b, 111, 8, 8) -> (b, 64 (seq), 64 (features)) -> (b, 64 (features), 64 (seq))
        x = x + self.positional_embedding
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