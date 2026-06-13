
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim) -> None:
        super().__init__()

        """
        https://pettingzoo.farama.org/environments/classic/chess/
        "Channel 6: All ones to help neural networks find board edges in padded convolutions"
        implies use a conv network
        """
        self.conv_layer = nn.Sequential(
            nn.Conv2d(state_dim,
                      hidden_dim,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim,
                      hidden_dim,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim,
                      hidden_dim,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.SiLU(),
        )

        self.actor_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 8 * 8, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, action_dim),
        )

        self.critic_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 8 * 8, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, 1), # maps down to scalar value
        )
    
    def forward(self, x, action_mask):
        # action_mask comes from observations dict

        x = self.conv_layer(x)
        logits = self.actor_head(x)
        logits[action_mask] = -1e9 # very small value, softmax makes this be functionally zero

        # squeeze to turn (batch_size, 1) to (batch_size)
        return (logits, self.critic_head(x).squeeze(-1))
    
    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32) # state features are 8x8x111
        logits, value = self(state_t)

        dist = Categorical(logits=logits) # softmaxes logits internally

        action = dist.sample() # get action based on probs above
        log_prob = dist.log_prob(action)

        return (action.item(), log_prob, value)
    
    def evaluate_actions(self, states, actions):
        # Re-score the picks from select action

        states_t = torch.as_tensor(states, dtype=torch.float32)

        logits, values = self(states_t)

        dist = Categorical(logits=logits)

        # we don't need to find actions because we are just rescoring the past action
        new_log_probs = dist.log_prob(torch.as_tensor(actions, dtype=torch.long))

        entropy = dist.entropy()

        return (new_log_probs, entropy, values)
    
def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    advantages = []
    values = values + [last_value] # so V(t+1) exists even at the end

    for t in reversed(range(len(rewards))): # this goes backwards thru time
        mask = 1.0 - float(dones[t]) # if dones[t] is true, mask zeros delta & gae
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae # gae recursion
        advantages.insert(0, gae) # prepend bc iterating backwards
    
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    values_t = torch.as_tensor(values, dtype=torch.float32)

    returns = advantages + values_t
    
    advantages = (advantages - advantages.mean())/(advantages.std() + 1e-9) # normalize

    return (advantages, returns)