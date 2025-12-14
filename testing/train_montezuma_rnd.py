# train_montezuma_rnd.py
"""
Train PPO + RND on ALE/MontezumaRevenge-v5 with common Atari preprocessing.
- Uses gymnasium wrappers: ResizeObservation, GrayScaleObservation, FrameStack
- Vectorized with DummyVecEnv and VecTransposeImage for SB3 CNN policy compatibility
- Trains in small chunks and saves checkpoints
- Demonstrates loading & continuing training
"""

import os
# os.environ['SDL_AUDIODRIVER'] = 'dummy'   # reduce ALSA audio warnings if headless
# (optional) Uncomment if you have trouble with X displays:
# os.environ['DISPLAY'] = ':0'

import gymnasium as gym
import ale_py
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import collections
import argparse
import time

gym.register_envs(ale_py)  

class FrameStackFallback(gym.Wrapper):
    """
    Simple FrameStack fallback compatible with Gymnasium's API:
    - stacks last `num_stack` observations along a new channel dimension.
    - expects observations to be numpy arrays (H,W) or (H,W,C).
    """
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)
        obs_shape = env.observation_space.shape
        # if grayscale obs (H,W) -> we make (H,W,num_stack)
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
            # fallback; keep same space (best-effort)
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
        # arrays could be (H,W) or (H,W,C)
        if arrs[0].ndim == 2:
            # stack along last axis -> (H,W,num_stack)
            return np.stack(arrs, axis=-1)
        else:
            # flatten channel axis: (H,W,C*num_stack)
            return np.concatenate(arrs, axis=-1)

FrameStack = FrameStackFallback

# Try to import PPO from sb3_contrib (RND implementation)
try:
    from sb3_contrib import PPO as PPO_RND
    has_sb3_contrib = True
except Exception:
    # fallback to stable-baselines3 PPO (no RND)
    from stable_baselines3 import PPO
    PPO_RND = PPO
    has_sb3_contrib = False

def make_montezuma_env(render_mode=None, seed=None):
    """
    Create a single wrapped Montezuma env:
    - Resize to 84x84
    - Convert to grayscale
    - FrameStack 4
    """
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    # Resize then gray then stack
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    env = FrameStack(env, num_stack=4)
    # Monitor to log episode reward/length
    env = Monitor(env)
    return env

def make_vec_env(n_envs=1, render_mode=None, seed=0):
    # Build a DummyVecEnv from our wrapped env factory
    def _make():
        return make_montezuma_env(render_mode=render_mode, seed=seed)
    vec_env = DummyVecEnv([_make for _ in range(n_envs)])
    # VecTransposeImage changes channel ordering for SB3 CNN (HWC -> CHW)
    vec_env = VecTransposeImage(vec_env)
    return vec_env

def train(total_chunks=10, timesteps_per_chunk=2000, out_folder="models", use_rnd=True, n_envs=1, n_steps=512):
    """
    Train loop:
      total_chunks * timesteps_per_chunk total timesteps (approx).
    Use smaller n_steps for quicker iterations when testing.
    """
    os.makedirs(out_folder, exist_ok=True)
    tb_log_dir = os.path.join(out_folder, "tensorboard")
    env = make_vec_env(n_envs=n_envs, render_mode=None, seed=0)

    # Choose policy and model class
    if use_rnd and has_sb3_contrib:
        ModelClass = PPO_RND
        model_name = "PPO_RND"
        model_kwargs = {"use_rnd": True}  # sb3_contrib's PPO accepts this kwarg
    else:
        ModelClass = PPO_RND  # either stable-baselines3.PPO or sb3_contrib.PPO if contrib missing
        model_name = "PPO_noRND" if not has_sb3_contrib else "PPO_contrib_no_rndflag"
        model_kwargs = {}

    # Create (or load) model
    model_path0 = os.path.join(out_folder, "checkpoint_0.zip")
    if os.path.exists(model_path0):
        print(f"Loading checkpoint {model_path0} ...")
        model = ModelClass.load(model_path0, env=env)
        # If using stable-baselines3.PPO, ensure 'n_steps' matches or it's okay to continue
        print("Loaded existing model. Continuing training.")
    else:
        print(f"Creating new model: {model_name}")
        model = ModelClass(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=tb_log_dir,
            n_steps=n_steps,
            **model_kwargs
        )

    # Checkpoint callback to save every chunk
    # ckpt_callback = CheckpointCallback(save_freq=1, save_path=out_folder,
    #                                    name_prefix="ppo_montezuma_chunk")

    total_so_far = 0
    for chunk in range(total_chunks):
        print(f"\n=== Chunk {chunk+1}/{total_chunks} — training {timesteps_per_chunk} timesteps ===")
        start = time.time()
        # model.learn respects full PPO rollouts. If timesteps_per_chunk < n_steps, SB3 will complete at least one rollout.
        model.learn(total_timesteps=timesteps_per_chunk, reset_num_timesteps=False) #Include callback in case it's uncommented above in future
        elapsed = time.time() - start
        total_so_far += timesteps_per_chunk
        # Save a checkpoint
        ckpt_name = os.path.join(out_folder, f"ppo_montezuma_{chunk+1}.zip")
        model.save(ckpt_name)
        print(f"Saved checkpoint: {ckpt_name} (chunk training time: {elapsed:.1f}s). Total timesteps (approx): {total_so_far}")
    env.close()
    print("Training finished. Final model saved at:", ckpt_name)
    return model, ckpt_name

def evaluate(model_path, episodes=3, render=True):
    """Load a saved model and run a few episodes with rendering."""
    render_mode = "human" if render else None
    env = make_montezuma_env(render_mode=render_mode, seed=42)
    # For SB3, load with env
    try:
        model = PPO_RND.load(model_path, env=env)
    except Exception:
        # fallback load using stable-baselines3 API
        from stable_baselines3 import PPO as SB3_PPO
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
    parser.add_argument("--no-rnd", dest="use_rnd", action="store_false", help="Disable RND (if sb3-contrib missing)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs")
    parser.add_argument("--n-steps", type=int, default=512, help="PPO n_steps per update (smaller for quick tests)")
    args = parser.parse_args()

    print("sb3-contrib available:", has_sb3_contrib)
    print("use_rnd flag:", args.use_rnd)
    model, final_ckpt = train(total_chunks=args.chunks,
                              timesteps_per_chunk=args.chunk_steps,
                              out_folder=args.out,
                              use_rnd=args.use_rnd and has_sb3_contrib,
                              n_envs=args.n_envs,
                              n_steps=args.n_steps)

    print("\nTo evaluate the saved model run:")
    print(f"python {__file__} --eval {final_ckpt}")

