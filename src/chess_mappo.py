
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

    def collect_rollout(self):
        pass

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
            pass
        """
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                mask = observation["action_mask"]
                # this is where you would insert your policy
                action = env.action_space(agent).sample(mask)

            env.step(action)
        env.close()
        """

        # 2. Train loop
        """
        During the train loop, I need to:
        - collect a rollout
        - update all metrics, including for saving a graph, saving the model, and stop on reward
        - Optimize the models
        """
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