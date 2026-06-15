
from pettingzoo.classic import chess_v6
from gymnasium.spaces import Discrete

import torch
import numpy as np
from model_architectures.cnn import ActorCritic, compute_gae

import yaml
import itertools
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from typing import cast



DATE_FORMAT = "%y-%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set

        self.learning_rate = hyperparameters['learning_rate']
        self.gamma = hyperparameters['gamma']
        self.hidden_dim = hyperparameters['hidden_dim']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.window = hyperparameters['window']
        self.gae_lambda = hyperparameters['gae_lambda']
        self.clip_ratio = hyperparameters['clip_ratio']
        self.steps_per_update = hyperparameters['steps_per_update']
        self.epochs_per_rollout = hyperparameters['epochs_per_rollout']
        self.minibatch_size = hyperparameters['minibatch_size']
        self.value_coef = hyperparameters['value_coef']
        self.entropy_coef = hyperparameters['entropy_coef']
        self.max_grad_norm = hyperparameters['max_grad_norm']
        self.episodes_per_elo_update = hyperparameters['episodes_per_elo_update']

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def collect_rollout(self, env, network):
        """
        interesting things:
        current_ep_reward is prolly 0 bc white +1 == black -1
        """

        """
        This has 10 parts: (i struggled a lot to understand this)
        1. init all the buffers
        2. loops & get obs, reward, term, and trunc 
        3. update rewards and dones bc those are past things (see pending logic)
        4. update current ep reward
        5. get action (and log prob and value)
        6. append everything to buffers if not already done so
        7. env step by action
        8. check agents still existing and length actions logic
        9. outside all that loop: bootstrap last val in case cut off
        10. convert necessary buffers -> tensors, return them

        Note that there is no global observation 
        """
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        masks = []
        completed_ep_rewards = []
        current_ep_reward = 0
        pending = {} # key: each agent. value: id of their PAST reward
        env.reset() # in MARL, doesn't return anything

        while len(actions) < self.steps_per_update:
            for agent in env.agent_iter():
                observation, reward, terminated, truncated, _ = env.last()
                """
                In AEC, env.last() returns data for the current agent (env.agent_selection)
                so all of this is correct for this agent
                """

                done = terminated | truncated
                """
                while coding this, misc idea came up:
                if we make done = term OR trunc, we essentially incentivize the agent to finish the game as fast as possible
                bc there is no future value for trunc
                """

                # the reward from env.last() is what the agent earned ever since its previous turn
                # so attach that reward to the previous transition because that reward
                # is the reward AS A RESULT of the transition in the past!
                if agent in pending:
                    rewards[pending[agent]] += reward
                    dones[pending[agent]] = done
                
                current_ep_reward += reward

                if done:
                    action = None
                else:
                    mask = observation["action_mask"]
                    action, log_prob, value = network.select_action(observation["observation"], mask)

                    # append them to their respective buffers
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value.item()) # .item() to make value go from 0D tensor to a plain float
                    masks.append(mask)
                    observations.append(observation["observation"])
                    rewards.append(0.0) # this is a placeholder
                    dones.append(False) # ditto ^
                    pending[agent] = len(actions) - 1 # so it points to the last transition
                
                env.step(action)

                # check conditions
                if not env.agents:
                    completed_ep_rewards.append(current_ep_reward)
                    current_ep_reward = 0
                    pending = {}
                    env.reset()
                
                if len(actions) >= self.steps_per_update:
                    break
        
        # bootstrap last value if we ended midway thru OR at the end of an episode
        # network(torch.tensor(env.last()[0]))[1] == network(torch.tensor(next state))[1] == value of next state
        if not env.agents:
            last_value = 0
        else:
            """
            because this runs AFTER env.step(action)
            step advances to the next agent
            so we need to negative this last value because the value of the next agent
            is the opposite of our value
            and vice versa


            BUT this issue is also in GAE
            bc our buffers alternate white black white black...
            so in GAE, we negative the v[t + 1] for delta
            and negative GAE for recursion
            so the fix for this is in GAE calculation.
            """
            state = env.last()[0]
            _, last_value = network(torch.as_tensor(state["observation"], dtype=torch.float32), state["action_mask"])
            last_value = last_value.item()
        
        # make these a tensor bc needed for optimizer, the rest gets normal math done on by GAE calc
        observations = torch.as_tensor(np.array(observations), dtype=torch.float32) # np array bc each state is a list of ndarrays (i think)
        actions = torch.as_tensor(actions, dtype=torch.long)
        old_log_probs = torch.stack(log_probs) # just stack into a shape of the number of transitions — stacks 0D log probs in to 1D tensor

        # do np.array bc masks is a list of np.ndarray 1D masks
        masks = torch.as_tensor(np.array(masks), dtype=torch.int8) # note: could update architecture forwards bc they already torch.tensorify mask anyway

        return (observations, actions, old_log_probs, masks, rewards, values, dones,
        last_value, completed_ep_rewards)

    def train(self):
        """
        Can summarize this in two steps:
        1. pre-train loop, initializing everything
        2. train loop:
        - Collect a rollout
        - Update all the metrics
        - Optimize/
        """

        # 1. Pre-Train loop
        """
        Before training need to
        - create env
        - create the networks and optimizers (weight sharing here)
        - Collect any metrics we need
        - Begin logging
        - start the loop
        """

        # Create env
        env = chess_v6.env(render_mode=None)

        # Weight sharing -- create networks & optimizer
        agent = env.agent_selection # gets the agent we're starting with — whichever plays white

        num_states = env.observation_space(agent).shape

        action_space = cast(Discrete, env.action_space(agent))
        num_actions = action_space.n

        network = ActorCritic(num_states, num_actions, hidden_dim=self.hidden_dim)
        optimizer = torch.optim.Adam(params=network.parameters(),
                                     lr=self.learning_rate)

        # also create the loss function
        loss_fn = torch.nn.MSELoss()

        # start collecting metrics
        episodes = 0
        rewards_per_episodes_in_rollout = []
        mean_rewards = []
        best_reward = float('-inf')

        # begin logging
        start_time = datetime.now()
        last_graph_update_time = start_time
        message = f"{start_time.strftime(DATE_FORMAT)}: Training starting. Reward: N/A; ELO: 0"
        self._log(message)

        # Begin the loop
        for _ in itertools.count():
            # 2. Train loop
            
            # Collect rollout

            # Compute GAE

            # Optimize

            # Metrics
            pass
        pass

    def optimize(self):
        pass

    def run(self):
        # remember render = true here
        pass

    def save_graph(self):
        # do this if its possible to graph elo over time
        pass
    
    def save_models(self):
        pass

    def load_models(self):
        pass
    
    # ideally we log both reward AND elo tho
    def _log(self, message):
        print(message)
        with open(self.LOG_FILE, 'a') as f:
            f.write(message + '\n')

def main():
    pass

if __name__ == '__main__':
    pass