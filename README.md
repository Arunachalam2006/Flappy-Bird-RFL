# Flappy Bird – Deep Q-Learning (DQN)

A Deep Q-Network agent that learns to play Flappy Bird using PyTorch and the
[`flappy-bird-gymnasium`](https://pypi.org/project/flappy-bird-gymnasium/) environment.

## Project structure

| File | What it does |
|---|---|
| `agent.py` | Training and testing loop: epsilon-greedy actions, replay memory, target network, model saving |
| `dqn.py` | The neural network (input → 256 hidden units → one Q-value per action) |
| `experience_replay.py` | Replay memory (a fixed-size queue of past experiences, sampled randomly) |
| `parameters.yaml` | Hyperparameters (set `flappybirdv3`) |
| `runs/flappybirdv3.pt` | Trained model |
| `runs/flappybirdv3.log` | Training log (best rewards and best 100-episode averages) |

## How it works

- **Observation:** 12 numbers from the game (bird height, vertical speed, rotation and the positions of the next 3 pipes), using `use_lidar=False`.
- **Actions:** 2 (do nothing / flap).
- **Learning:** after every step, a random mini-batch of 32 past experiences updates the policy network using the target
  `reward + gamma × max Q_target(next_state)`.
- **Target network:** copied from the policy network every 100 steps.
- **Exploration:** epsilon starts at 1 and decays by 0.999 per episode down to 0.05.
- **Saving:** the model is saved whenever the average reward of the last 100 episodes improves.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# train
python agent.py flappybirdv3 --train

# watch the trained agent play
python agent.py flappybirdv3
```

Training runs until you stop it with `Ctrl + C`. Note: training again with the same set name starts from new random
weights and will overwrite `runs/flappybirdv3.pt`, so copy the model first if you want to keep it.

## Results

Trained for about 32,600 episodes:

| Episode | Best 100-episode average reward |
|---|---|
| 1,000 | 0.35 |
| 5,000 | 9.0 |
| 20,000 | 11.8 |
| 28,072 | **13.2** |

Best single episode: **69.6** (episode 30,533).
Reward per step: +0.1 for staying alive, +1 for passing a pipe, −1 for dying.

## What I learned

The first version used the environment's default **lidar** observation (180 distance rays). Even after more than
30,000 episodes it only passed 1–2 pipes. Switching to the 12-feature observation, which includes the bird's
vertical speed, made the same DQN learn the game and it improved drastically when compared to the previous method
