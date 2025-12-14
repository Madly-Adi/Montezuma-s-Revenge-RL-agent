import os
import gymnasium as gym
import ale_py
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import collections
import argparse
import time

gym.register_envs(ale_py)  

class FrameStackFallback(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 2:
            h, w = obs_shape
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(h, w, num_stack), dtype=np.uint8
            )
        elif len(obs_shape) == 3:
            h, w, c = obs_shape
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(h, w, c * num_stack), dtype=np.uint8
            )
        else:
            self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        stacked = self._get_observation()
        return stacked, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        arrs = list(self.frames)
        if arrs[0].ndim == 2:
            return np.stack(arrs, axis=-1)
        else:
            return np.concatenate(arrs, axis=-1)

FrameStack = FrameStackFallback

def make_montezuma_env(render_mode=None, seed=None):
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    env = FrameStack(env, num_stack=4)
    env = Monitor(env)
    return env

def make_vec_env(n_envs=1, render_mode=None, seed=0):
    def _make():
        return make_montezuma_env(render_mode=render_mode, seed=seed)
    vec_env = DummyVecEnv([_make for _ in range(n_envs)])
    vec_env = VecTransposeImage(vec_env)
    return vec_env

def train(total_chunks=10, timesteps_per_chunk=2000, out_folder="models", n_envs=1, n_steps=512):
    os.makedirs(out_folder, exist_ok=True)
    tb_log_dir = os.path.join(out_folder, "tensorboard")
    env = make_vec_env(n_envs=n_envs, render_mode=None, seed=0)

    ModelClass = PPO
    model_name = "PPO_noRND"
    model_path0 = os.path.join(out_folder, "checkpoint_0.zip")
    if os.path.exists(model_path0):
        print(f"Loading checkpoint {model_path0} ...")
        model = ModelClass.load(model_path0, env=env)
        print("Loaded existing model. Continuing training.")
    else:
        print(f"Creating new model: {model_name}")
        model = ModelClass(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=tb_log_dir,
            n_steps=n_steps
        )

    
    total_so_far = 0
    for chunk in range(total_chunks):
        print(f"\n=== Chunk {chunk+1}/{total_chunks} — training {timesteps_per_chunk} timesteps ===")
        start = time.time()
        model.learn(total_timesteps=timesteps_per_chunk, reset_num_timesteps=False)
        elapsed = time.time() - start
        total_so_far += timesteps_per_chunk
        ckpt_name = os.path.join(out_folder, f"ppo_montezuma_{chunk+1}.zip")
        model.save(ckpt_name)
        print(f"Saved checkpoint: {ckpt_name} (chunk training time: {elapsed:.1f}s). Total timesteps (approx): {total_so_far}")
    env.close()
    print("Training finished. Final model saved at:", ckpt_name)
    return model, ckpt_name

from stable_baselines3 import PPO as SB3_PPO
def evaluate(model_path, episodes=3, render=True):
    render_mode = "human" if render else None
    env = make_montezuma_env(render_mode=render_mode, seed=42)
    model = SB3_PPO.load(model_path, env=env)
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
        print(f"Eval episode {ep+1} reward: {ep_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=6, help="How many chunks to train")
    parser.add_argument("--chunk_steps", type=int, default=4000, help="Timesteps per chunk")
    parser.add_argument("--out", type=str, default="models", help="Output folder for checkpoints")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs")
    parser.add_argument("--n-steps", type=int, default=512, help="PPO n_steps per update (smaller for quick tests)")
    args = parser.parse_args()

    print("use_rnd flag:", args.use_rnd)
    model, final_ckpt = train(total_chunks=args.chunks,
                              timesteps_per_chunk=args.chunk_steps,
                              out_folder=args.out,
                              n_envs=args.n_envs,
                              n_steps=args.n_steps)

    print("\nTo evaluate the saved model run:")
    print(f"python {__file__} --eval {final_ckpt}")

