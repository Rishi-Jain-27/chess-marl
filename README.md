<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Rishi-Jain-27/chess-marl">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">chess-marl</h3>

  <p align="center">
    Teaching chess agents to play through multi-agent self-play with MAPPO — comparing CNN, Transformer, and hybrid policies, scored against Stockfish.
    <br />
    <a href="https://github.com/Rishi-Jain-27/chess-marl"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#demos">View Demo</a>
    &middot;
    <a href="https://github.com/Rishi-Jain-27/chess-marl/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/Rishi-Jain-27/chess-marl/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#how-it-works">How It Works</a>
      <ul>
        <li><a href="#the-environment">The Environment</a></li>
        <li><a href="#the-algorithm">The Algorithm</a></li>
        <li><a href="#the-architectures">The Architectures</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#demos">Demos</a></li>
    <li><a href="#results--reflections">Results & Reflections</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

**chess-marl** trains a chess-playing agent from scratch using **multi-agent reinforcement learning**. Two agents (white and black) share a single neural network and learn entirely through self-play with **MAPPO** (Multi-Agent Proximal Policy Optimization). No human games, no opening books — just two copies of the same network playing each other and improving.

The project was built to answer a concrete question: **how does the policy/value network architecture affect playing strength?** To find out, the same MAPPO training loop is run against three interchangeable architectures — a CNN, a Transformer, and a hybrid of the two — and each is scored by playing rated games against [Stockfish](https://stockfishchess.org/).

Key ideas explored here:

* **Weight sharing** — a single `ActorCritic` network plays both colors. The environment's `observe()` always orients the board towards the side to move, so one network can learn to play both white and black.
* **The zero-sum value trick** — in chess, white's value is exactly the negative of black's. Because the rollout buffer alternates `white, black, white, black, ...`, the GAE computation negates `V(t+1)` and the running advantage on every step to keep the math consistent.
* **Sparse-reward credit assignment** — rewards in chess only arrive at checkmate/draw. Each agent's reward is attached to its *previous* transition (the move that caused it) via a `pending` bookkeeping dict.
* **ELO benchmarking** — the network is periodically evaluated against Stockfish (depth 3 ≈ 1000 ELO) and its ELO is back-solved from the win/draw/loss rate.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* [![PyTorch][PyTorch]][PyTorch-url]
* [![PettingZoo][PettingZoo]][PettingZoo-url]
* [![Gymnasium][Gymnasium]][Gymnasium-url]
* [![python-chess][chess]][chess-url]
* [![Stockfish][Stockfish]][Stockfish-url]
* [![NumPy][NumPy]][NumPy-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- HOW IT WORKS -->
## How It Works

### The Environment

Training uses the [PettingZoo `chess_v6`](https://pettingzoo.farama.org/environments/classic/chess/) AEC environment, which mirrors AlphaZero's representation:

* **Observation** — an `8 × 8 × 111` tensor (the board plus history and metadata planes).
* **Action space** — a flattened `8 × 8 × 73 = 4672` discrete action space encoding queen-like moves, knight moves, and underpromotions. Illegal moves are removed with the environment's `action_mask`, which is applied as a `-1e9` fill on the policy logits.

### The Algorithm

The core MAPPO loop lives in [src/chess_mappo.py](src/chess_mappo.py):

1. **Collect a rollout** — step through self-play games until `steps_per_update` transitions are buffered, handling delayed rewards and bootstrapping the final value.
2. **Compute GAE** — generalized advantage estimation with the white/black value-negation fix, plus advantage normalization.
3. **Optimize** — the clipped PPO surrogate objective over several epochs of minibatches, with a value loss and an entropy bonus.
4. **Benchmark & checkpoint** — every `steps_per_save` steps, evaluate ELO against Stockfish and save the model whenever it matches or beats its best ELO.

All hyperparameters are defined per-experiment in [src/hyperparameters.yml](src/hyperparameters.yml).

### The Architectures

Each architecture exposes the same interface (`ActorCritic`, `select_action`, `evaluate_actions`, `compute_gae`) so it can be dropped into the training loop by changing a single import in [src/chess_mappo.py](src/chess_mappo.py):

| Architecture | File | Idea |
|---|---|---|
| **CNN** | [cnn.py](src/model_architectures/cnn.py) | 3 conv layers + MLP actor/critic heads — treats the board as a spatial image. |
| **Transformer** | [transformer.py](src/model_architectures/transformer.py) | Projects each square to a token and lets all 64 squares attend to each other. |
| **Hybrid** | [hybrid.py](src/model_architectures/hybrid.py) | A conv stem feeds a 2-layer Transformer encoder — local features *then* global attention. |

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* **Python 3.11**
* **Stockfish** — used as the benchmark opponent. On macOS:
  ```sh
  brew install stockfish
  ```
  By default the code looks for the binary at `/opt/homebrew/bin/stockfish`. Override with the `STOCKFISH_PATH` environment variable if it lives elsewhere.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Rishi-Jain-27/chess-marl.git
   cd chess-marl
   ```
2. Create a virtual environment and install the dependencies
   ```sh
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install torch pettingzoo[classic] gymnasium chess numpy matplotlib pyyaml
   ```
3. (Optional) Install Stockfish for ELO evaluation and head-to-head games — see [Prerequisites](#prerequisites).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

All commands are run from the `src/` directory (the script loads `hyperparameters.yml` and the model architectures relative to it):

```sh
cd src
```

**Pick an architecture.** Edit the import at the top of [src/chess_mappo.py](src/chess_mappo.py) to choose which network to use:

```python
from model_architectures.hybrid import ActorCritic, compute_gae   # or .cnn / .transformer
```

**Train** a model against one of the hyperparameter sets defined in [hyperparameters.yml](src/hyperparameters.yml) (`cnn_chess`, `transformer_chess`, `hybrid_chess`, …). Checkpoints, logs, and an ELO-vs-steps graph are written to `src/runs/`:

```sh
python chess_mappo.py hybrid_chess --train
```

**Watch a trained model play itself** in a rendered window (load a saved checkpoint by its ELO):

```sh
python chess_mappo.py hybrid_chess --elo 915
```

**Play a trained model against Stockfish** (depth 3 ≈ 1000 ELO):

```sh
python chess_mappo.py hybrid_chess --stockfish --elo 915
```

You can also drive the whole pipeline interactively from [src/training.ipynb](src/training.ipynb).

> **Note:** model checkpoints are saved as `runs/<hyperparameter_set>_ELO_<elo>.pt`. The included [hybrid_chess_ELO_915.pt](src/runs/hybrid_chess_ELO_915.pt) is the strongest trained model in this repo.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- DEMOS -->
## Demos

Sample games are in the [demos/](demos/) folder:

* `random_action.mov` — an untrained agent making random legal moves (the starting point).
* `hybrid_elo_915_versus_stockfish.mov` — the hybrid model (ELO ≈ 915) playing against Stockfish.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- RESULTS & REFLECTIONS -->
## Results & Reflections

The best model reached an **ELO of ~915** against the Stockfish depth-3 baseline — competitive but short of beating it consistently.

The bigger takeaway is a lesson about *when* PPO is the right tool. Chess has **extremely sparse rewards** (a single +1/−1/0 signal at the end of a long game), which makes the credit-assignment problem brutal for a policy-gradient method like PPO. Search-based approaches such as **Monte Carlo Tree Search** (à la AlphaZero) are a far better fit for this setting because they can plan through the sparse-reward horizon rather than waiting for gradients to propagate it. This project was as much about *learning where RL methods break down* as it was about building a strong chess engine.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] CNN, Transformer, and hybrid architectures
- [x] MAPPO self-play training loop with weight sharing
- [x] ELO benchmarking against Stockfish
- [ ] Batched rollouts (remove the per-step `unsqueeze(0)`)
- [ ] Vectorized / parallel environments for faster data collection
- [ ] Explore MCTS-based training (AlphaZero-style) to handle sparse rewards

See the [open issues](https://github.com/Rishi-Jain-27/chess-marl/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Rishi Jain — [@Rishi-Jain-27](https://github.com/Rishi-Jain-27)

Project Link: [https://github.com/Rishi-Jain-27/chess-marl](https://github.com/Rishi-Jain-27/chess-marl)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [PettingZoo — Chess (chess_v6)](https://pettingzoo.farama.org/environments/classic/chess/)
* [AlphaZero / *Mastering Chess and Shogi by Self-Play* (Silver et al.)](https://arxiv.org/abs/1712.01815)
* [The MAPPO paper — *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (Yu et al.)](https://arxiv.org/abs/2103.01955)
* [Stockfish](https://stockfishchess.org/)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/Rishi-Jain-27/chess-marl.svg?style=for-the-badge
[contributors-url]: https://github.com/Rishi-Jain-27/chess-marl/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Rishi-Jain-27/chess-marl.svg?style=for-the-badge
[forks-url]: https://github.com/Rishi-Jain-27/chess-marl/network/members
[stars-shield]: https://img.shields.io/github/stars/Rishi-Jain-27/chess-marl.svg?style=for-the-badge
[stars-url]: https://github.com/Rishi-Jain-27/chess-marl/stargazers
[issues-shield]: https://img.shields.io/github/issues/Rishi-Jain-27/chess-marl.svg?style=for-the-badge
[issues-url]: https://github.com/Rishi-Jain-27/chess-marl/issues
[license-shield]: https://img.shields.io/github/license/Rishi-Jain-27/chess-marl.svg?style=for-the-badge
[license-url]: https://github.com/Rishi-Jain-27/chess-marl/blob/main/LICENSE
[PyTorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[PettingZoo]: https://img.shields.io/badge/PettingZoo-0B5394?style=for-the-badge&logo=farama&logoColor=white
[PettingZoo-url]: https://pettingzoo.farama.org/
[Gymnasium]: https://img.shields.io/badge/Gymnasium-007ACC?style=for-the-badge&logo=openai&logoColor=white
[Gymnasium-url]: https://gymnasium.farama.org/
[chess]: https://img.shields.io/badge/python--chess-000000?style=for-the-badge&logo=lichess&logoColor=white
[chess-url]: https://python-chess.readthedocs.io/
[Stockfish]: https://img.shields.io/badge/Stockfish-769656?style=for-the-badge&logo=chessdotcom&logoColor=white
[Stockfish-url]: https://stockfishchess.org/
[NumPy]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
