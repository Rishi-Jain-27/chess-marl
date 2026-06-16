#need to create colab notebook for training this ofc

from pettingzoo.classic import chess_v6
from pettingzoo.classic.chess import chess_utils
from gymnasium.spaces import Discrete

import chess
import chess.engine

import torch
import numpy as np
import math
from model_architectures.cnn import ActorCritic, compute_gae # change this to change what architecture we test!

import yaml
import itertools
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from typing import cast

# before stockfish path, run brew install stockfish
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")

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
        self.epochs_per_optimize = hyperparameters['epochs_per_optimize']
        self.minibatch_size = hyperparameters['minibatch_size']
        self.value_coef = hyperparameters['value_coef']
        self.entropy_coef = hyperparameters['entropy_coef']
        self.max_grad_norm = hyperparameters['max_grad_norm']
        self.steps_per_save = hyperparameters['steps_per_save']
        self.num_elo_loops = hyperparameters['num_elo_loops']

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
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
                if not env.agents: # could also move this outside of the for loop
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
            observation = env.last()[0]
            with torch.no_grad():
                # unsqueeze to add the batch dim of 1
                _, last_value = network(torch.as_tensor(np.ascontiguousarray(observation["observation"]), dtype=torch.float32, device=device).unsqueeze(0), observation["action_mask"])
            last_value = last_value.item()
        
        # make these a tensor bc needed for optimizer, the rest gets normal math done on by GAE calc
        observations = torch.as_tensor(np.ascontiguousarray(np.array(observations)), dtype=torch.float32) # np array bc each state is a list of ndarrays (i think)
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
        env.reset()

        # Weight sharing -- create networks & optimizer
        agent = env.agent_selection # gets the agent we're starting with — whichever plays white

        num_states = env.observation_space(agent)["observation"].shape # type: ignore

        action_space = cast(Discrete, env.action_space(agent))
        num_actions = action_space.n

        network = ActorCritic(num_states, num_actions, hidden_dim=self.hidden_dim).to(device)
        optimizer = torch.optim.Adam(params=network.parameters(),
                                     lr=self.learning_rate)

        # start collecting metrics
        num_steps = 0
        total_steps = 0
        elo_steps = []
        best_elo = float('-inf')
        network_elos = []

        # begin logging
        last_graph_update_time = datetime.now()
        message = f"{datetime.now().strftime(DATE_FORMAT)}: Training starting. ELO: 0"
        self._log(message)

        # Begin the loop
        for _ in itertools.count():
            # 2. Train loop

            # Collect rollout
            observations, actions, old_log_probs, masks, rewards, values, dones, last_value, completed_ep_rewards = self.collect_rollout(env, network)

            # Compute GAE
            advantages, returns = compute_gae(rewards, values, dones, last_value, self.gamma, self.gae_lambda)

            # Optimize
            self.optimize(network, optimizer, observations, actions, masks, old_log_probs, advantages, returns)

            # Metrics
            num_steps += len(actions)
            total_steps += len(actions)

            if num_steps >= self.steps_per_save:
                # Find network elo (logging happens in compute elo)
                network_elo, wins, draws, losses = self.compute_elo(network) # type: ignore

                # Log
                message = f"New ELO: {network_elo} at step: {num_steps} ({wins}W/{draws}D/{losses}L)"
                self._log(message)

                # Add that to list
                network_elos.append(network_elo)
                elo_steps.append(total_steps)

                # Graph it
                if datetime.now() - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(network_elos, elo_steps)
                    last_graph_update_time = datetime.now()

                # Save permanently if elo is 100 greater than previous best elo
                assert network_elo is not None # these asserts are so pylance doesn't go insane
                assert best_elo is not None
                if network_elo >= best_elo:
                    self.save_model(network, network_elo)
                
                # Update best elo
                if network_elo > best_elo:
                    best_elo = network_elo
                
                num_steps = 0

    def optimize(self, network, optimizer, observations, actions, masks, old_log_probs, advantages, returns):
        """
        Implement's PPO surrogate loss calculation stuff

        Basically:
        - each rollout we loop epochs times and then however many times to fill the minibatches
        - and create indexes for the minibatches
        - Get minibatches
        - Evaluate the past minibatched states and minibatched actions (remember the mask)
        - Find the ratio
        - Find surrogate one and two
        - Get the actor and critic loss
        - calculate total loss, optimize.
        """

        observations = observations.to(device)
        actions = actions.to(device)
        masks = masks.to(device)
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)

        for epoch in range(self.epochs_per_optimize):
            # note that self.steps_per_update may not be 100% equal to the batch length
            perm = torch.randperm(observations.shape[0], device=device)
            for start in range(observations.shape[0] // self.minibatch_size):
                minibatch_idx = perm[start * self.minibatch_size : (start + 1) * self.minibatch_size]

                observations_minibatch = observations[minibatch_idx]
                actions_minibatch = actions[minibatch_idx]
                old_log_probs_minibatch = old_log_probs[minibatch_idx]
                advantages_minibatch = advantages[minibatch_idx]
                returns_minibatch = returns[minibatch_idx]
                masks_minibatch = masks[minibatch_idx]

                # Evaluate past states and actions
                new_log_probs, entropy, values = network.evaluate_actions(observations_minibatch, actions_minibatch, masks_minibatch)
                ratio = torch.exp(new_log_probs - old_log_probs_minibatch.squeeze(-1)) # importance sampling ratio
                surrogate_one = ratio * advantages_minibatch
                surrogate_two = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_minibatch # surrogate 2 is 1 but clipped
                actor_loss = -torch.min(surrogate_one, surrogate_two).mean()
                critic_loss = torch.mean((values - returns_minibatch)**2) # what we predicted to get - what we actually got **2
                loss = actor_loss + (self.value_coef * critic_loss) - (self.entropy_coef * entropy.mean())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
                optimizer.step()

    # this computes ELO scores for the network passed in
    def compute_elo(self, network):
        """
        This just evaluates the model against stockfish
        and computes ELO!
        """

        network.eval()

        # Creating stockfish engine
        # Stockfish depth 3 == 1000 elo baseline
        try:
            # Initialize the engine
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            opponent_elo = 1000
        except FileNotFoundError:
            print(f"Stockfish not at {STOCKFISH_PATH}. Run 'brew install stockfish'")
            network.train() # just set it back to train before going back
            return
        
        reward_per_game = []

        try:
            for i in range(self.num_elo_loops):
                network_agent = "player_0" if i % 2 == 0 else "player_1"
                env = chess_v6.env(render_mode=None)
                unwrapped_env = env.unwrapped

                env.reset()

                game_reward = 0

                # game loop
                for agent in env.agent_iter():
                    observation, reward, terminated, truncated, _ = env.last()

                    if agent == network_agent: game_reward += reward # accum reward if its our turn!
                    # don't put ^ below or will miss reward when terminated or truncated

                    # Get action
                    if terminated or truncated:
                        action = None
                    elif agent == network_agent: # network turn
                        with torch.inference_mode():
                            assert observation is not None
                            logits, _ = network(torch.as_tensor(np.ascontiguousarray(observation["observation"]), dtype=torch.float32, device=device).unsqueeze(0), observation["action_mask"])
                            action = int(torch.argmax(logits, dim=-1).item()) # get the best move
                    else: # stockfish turn
                        
                        # We ask stockfish for a move

                        board = unwrapped_env.board # type: ignore[attr-defined]
                        player = 0 if board.turn == chess.WHITE else 1

                        result = engine.play(board, chess.engine.Limit(depth=3))

                        # then find what legal action decodes to it
                        """
                        Stockfish outputs a chess.Move, like e2e4 (pawn from e2 to e4)
                        This is just decoding that to the 0-4671 integers that PettingZoo uses
                        and PettingZoo encodes that to the action
                        """
                        assert observation is not None
                        for act in np.flatnonzero(observation["action_mask"]): # loop thru legal moves
                            if chess_utils.action_to_move(board, int(act), player) == result.move:
                                action = int(act)
                                break

                    env.step(action)
                
                # Add the game's reward to the overall list
                reward_per_game.append(game_reward)
            
            # outside of looping all the games:
            wins = 0
            draws = 0
            num_games = len(reward_per_game)
            for i in reward_per_game:
                if i == 1: wins += 1
                if i == 0: draws += 1
            losses = num_games - wins - draws
            
            # Calculated inverse of the ELO formula
            # E = 1/(1 + 10^(elo difference/400))
            # thus elo difference = -400 * logbase10of (1/Elo - 1)
            score_rate = min(max(((wins + 0.5 * draws) / num_games), 1e-4), 1 - 1e-4) # clamp
            elo_diff = -400 * math.log10(1/(score_rate) - 1)

            network.train()
            return (round(opponent_elo + elo_diff), wins, draws, losses) # represents our bot elo as an integer
        
        finally:
            engine.quit()
            network.train()

    def run(self, elo):
        # Create env
        env = chess_v6.env(render_mode="human")
        env.reset()

        # ditto network
        agent = env.agent_selection
        num_states = env.observation_space(agent)["observation"].shape # type: ignore
        action_space = cast(Discrete, env.action_space(agent))
        num_actions = action_space.n
        network = ActorCritic(num_states, num_actions, hidden_dim=self.hidden_dim).to(device)

        self.load_model(network, elo)

        # Activate inference settings
        network.eval()
        with torch.inference_mode():
            # play!
            for agent in env.agent_iter():
                observation, reward, terminated, truncated, _ = env.last()

                # get action
                if terminated or truncated:
                    action = None
                else:
                    assert observation is not None
                    mask = observation["action_mask"]

                    logits, _ = network(torch.as_tensor(np.ascontiguousarray(observation["observation"]), dtype=torch.float32, device=device).unsqueeze(0), mask)
                    action = int(torch.argmax(logits, dim=-1).item())

                env.step(action)
            
            env.close()

    def save_graph(self, network_elos, network_elo_step_computed):
        fig = plt.figure(1)
        plt.xlabel('Steps')
        plt.ylabel('Model ELO')
        plt.plot(network_elo_step_computed, network_elos)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig) # so figures don't pile up 
    
    def save_model(self, network, elo):
        model_path = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_ELO_{elo}.pt')
        torch.save(network.state_dict(), model_path)

    def load_model(self, network, elo):
        model_path = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_ELO_{elo}.pt')
        try:
            network.load_state_dict(torch.load(model_path, weights_only=True))
        except:
            raise ValueError("Elo value must correspond to possible model elos.")
    
    # log elo
    def _log(self, message):
        print(message)
        with open(self.LOG_FILE, 'a') as f:
            f.write(message + '\n')

def main():
    parser = argparse.ArgumentParser(description="Train or run the chess bot?")
    parser.add_argument("hyperparameters", help="Enter the name of the set of hyperparameters to test/train")
    parser.add_argument("--train", help="Training mode", action="store_true")
    parser.add_argument("--elo", help="ELO value of the model to run.", type=int)
    args = parser.parse_args()

    chess_bot = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        chess_bot.train()
    else:
        chess_bot.run(elo=args.elo)

if __name__ == '__main__':
    main()