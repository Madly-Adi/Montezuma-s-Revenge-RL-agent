import numpy as np
import pandas as pd
import mdptoolbox    # pymdptoolbox
import mdptoolbox.example
from collections import defaultdict


W, H = 5, 5
actions = ['U','D','L','R']
action_dirs = {'U':(0,-1),'D':(0,1),'L':(-1,0),'R':(1,0)}
n_actions = len(actions)


key_pos = (1,1)
treasure_pos = (4,3)
step_reward = -0.01
key_reward = 1.0
treasure_reward = 10.0
gamma = 0.99

# build state space
states = []
for y in range(H):
    for x in range(W):
        for has_key in (0,1):
            states.append(((x,y), has_key))
state_to_idx = {s:i for i,s in enumerate(states)}
idx_to_state = {i:s for s,i in state_to_idx.items()}
n_states = len(states)

def in_bounds(x,y):
    return 0 <= x < W and 0 <= y < H

def is_terminal(state):
    (x,y), has_key = state
    return (x,y) == treasure_pos and has_key == 1

def transition_index_and_reward(state_idx, action):
    state = idx_to_state[state_idx]
    (x,y), has_key = state
    dx,dy = action_dirs[action]
    nx, ny = x + dx, y + dy
    if not in_bounds(nx,ny):
        nx, ny = x, y  # bump -> stay
    n_has_key = has_key
    reward = step_reward
    if (nx,ny) == key_pos and has_key == 0:
        n_has_key = 1
        reward += key_reward
    if (nx,ny) == treasure_pos and n_has_key == 1:
        reward += treasure_reward
    next_state = ((nx,ny), n_has_key)
    return state_to_idx[next_state], reward, is_terminal(next_state)

# Build P (A x S x S) and R (S x A)
P = np.zeros((n_actions, n_states, n_states), dtype=float)
R = np.zeros((n_states, n_actions), dtype=float)

for s in range(n_states):
    if is_terminal(idx_to_state[s]):
        # terminal: self-loop with zero reward for all actions
        for a_idx in range(n_actions):
            P[a_idx, s, s] = 1.0
            R[s, a_idx] = 0.0
        continue
    for a_idx, a in enumerate(actions):
        ns, r, term = transition_index_and_reward(s, a)
        P[a_idx, s, ns] = 1.0   # deterministic
        R[s, a_idx] = r



