import ale_py
import gymnasium as gym


gym.register_envs(ale_py)  

env = gym.make("ALE/MontezumaRevenge-v5", render_mode="human")
obs, info = env.reset()

for _ in range(50000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()

