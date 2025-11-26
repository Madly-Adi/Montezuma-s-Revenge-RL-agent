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


vi = mdptoolbox.mdp.ValueIteration(P, R, discount=gamma, epsilon=1e-6, max_iter=10000)
vi.run()

pi = mdptoolbox.mdp.PolicyIteration(P, R, discount=gamma, max_iter=10000)
pi.run()

# Extract results
policy_vi = np.array(vi.policy)      # length S, action indices 0..A-1; -1 not used here
V_vi = np.array(vi.V)
policy_pi = np.array(pi.policy)
V_pi = np.array(pi.V)

# Helper visualization (arrow map)
arrow_map = {0:'↑',1:'↓',2:'←',3:'→', -1:'·'}
def policy_grid(policy, has_key_flag):
    grid = [['' for _ in range(W)] for __ in range(H)]
    for y in range(H):
        for x in range(W):
            s_idx = state_to_idx[((x,y), has_key_flag)]
            grid[y][x] = arrow_map[int(policy[s_idx])]
    return grid

df_vi_no_key = pd.DataFrame(policy_grid(policy_vi, 0), index=[f'y={y}' for y in range(H)], columns=[f'x={x}' for x in range(W)])
df_vi_with_key = pd.DataFrame(policy_grid(policy_vi, 1), index=[f'y={y}' for y in range(H)], columns=[f'x={x}' for x in range(W)])
df_pi_no_key = pd.DataFrame(policy_grid(policy_pi, 0), index=[f'y={y}' for y in range(H)], columns=[f'x={x}' for x in range(W)])
df_pi_with_key = pd.DataFrame(policy_grid(policy_pi, 1), index=[f'y={y}' for y in range(H)], columns=[f'x={x}' for x in range(W)])


print("MDP summary: grid {}x{}, states={}, actions={}".format(W,H,n_states,n_actions))
print(f"Key at {key_pos}, Treasure at {treasure_pos}. step={step_reward}, key={key_reward}, treasure={treasure_reward}")
print("\nValue Iteration policy (no key):")
print(df_vi_no_key)
print("\nValue Iteration policy (with key):")
print(df_vi_with_key)
print("\nPolicy Iteration policy (no key):")
print(df_pi_no_key)
print("\nPolicy Iteration policy (with key):")
print(df_pi_with_key)


print("\nPolicy arrays (action indices: 0=U,1=D,2=L,3=R):")
print("policy_vi:", policy_vi)
print("policy_pi:", policy_pi)

