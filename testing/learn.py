from stable_baselines3 import PPO
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)  

env = gym.make("ALE/MontezumaRevenge-v5", render_mode="human")
obs, info = env.reset()

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1e3)

model.save("ppo_rnd_montezuma_0")

for i in range(5):
    model = PPO.load(f"ppo_rnd_montezuma_{i}", env=env)
    model.learn(total_timesteps=1e3, reset_num_timesteps=False)
    model.save(f"ppo_rnd_montezuma_{i+1}")

