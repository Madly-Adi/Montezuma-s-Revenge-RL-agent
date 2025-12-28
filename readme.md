# Reinforcement Learning Course Project  
## Montezuma’s Revenge (Atari)

**Authors :** Aditya Sharma and Sivaveerkrishna S.
 

---

## Overview

This project was carried out as part of a Reinforcement Learning course and focuses on **Track 2: Montezuma’s Revenge**, one of the most challenging Atari benchmarks due to its sparse rewards and long-horizon exploration requirements.

Unlike standard benchmarks, Montezuma’s Revenge requires structured exploration and careful algorithmic design. The project progresses systematically from theoretical modeling to simulator setup and finally to learning-based agents.

---

## Project Objectives

The project is divided into the following major components:

1. **Task 1 — MDP Formulation and Exact Solution**
   - Formulate a simplified version of Montezuma’s Revenge as a discrete-state, discrete-action MDP.
   - Solve the MDP using Value Iteration and Policy Iteration.

2. **Task 2 — Simulator Setup**
   - Set up a complete simulator for Montezuma’s Revenge using the Atari Learning Environment.
   - Generate sample trajectories under a random policy.

3. **Learning and Evaluation**
   - Train reinforcement learning agents using PPO and PPO + Random Network Distillation (RND).
   - Evaluate trained policies and analyze learning behavior under sparse rewards.

---

## Directory Structure
```
├── task-1/
│ └── main.py
│
├── task-2/
│ └── main.py
│
├── testing/
│ ├── evaluate.py
│ ├── learn.py
│ ├── main.py
│ └── train_montezuma_rnd.py
│
├── evaluate.py
├── train_montezuma_ppo.py
├── train_montezuma_ppo_rnd.py
├── requirements.txt
└── README.md
```

---

## Task Descriptions

### Task 1: Simplified MDP (Exact Planning)

- A grid-based abstraction of Montezuma’s Revenge is constructed.
- State space includes agent position and key possession.
- Reward structure captures delayed rewards and sparse feedback.
- Solved using **Value Iteration** and **Policy Iteration** via `mdptoolbox`.

**Location:** `task-1/main.py`

---

### Task 2: Atari Simulator Setup

- Montezuma’s Revenge is instantiated using `gymnasium` and `ale-py`.
- The simulator generates sample trajectories using a **random policy**.
- Visual rendering confirms correct environment interaction.

**Location:** `task-2/main.py`

---

### Learning-Based Experiments

- PPO and PPO + RND agents are trained on the Atari environment.
- RND is used to encourage exploration in the sparse-reward setting.
- Multiple scripts are used for training, testing, and evaluation.

**Key scripts:**
- `train_montezuma_ppo.py`
- `train_montezuma_ppo_rnd.py`
- `testing/train_montezuma_rnd.py`
- `evaluate.py`
- `testing/evaluate.py`

---

## Installation and Setup

It is recommended to use a Python virtual environment.

```bash
python -m venv rl-env
source rl-env/bin/activate
pip install -r requirements.txt
```
Ensure that the Atari ROMs required by ALE are properly installed as per ale-py guidelines.

## Running the Code
- Task 1 (MDP Solution)
```bash
python task-1/main.py
```

- Task 2 (Simulator with Random Policy)
```bash
python task-2/main.py
```

- Training PPO Agent
```bash
python train_montezuma_ppo.py
```

- Training PPO + RND Agent
```bash
python train_montezuma_ppo_rnd.py
```

- Evaluating a Trained Agent
```bash
python evaluate.py
```

## Key Observations

- Random policies almost never achieve positive reward, highlighting the difficulty of Montezuma’s Revenge hence exact planning methods work only in heavily simplified environments.

- PPO struggles with exploration when used alone.

- PPO combined with intrinsic motivation (RND) exhibits improved exploratory behavior, though the task remains challenging.

- Further advanced methods (Go explore or demonstration based RL) must be used to complete the game or in hopes to clear any room.  







