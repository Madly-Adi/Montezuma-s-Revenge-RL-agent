import os
import time
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
import ale_py

from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import VecEnvWrapper

gym.register_envs(ale_py)

class FrameStackFallback(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.frames = collections.deque(maxlen=num_stack)
        h, w = env.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, num_stack), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.frames.maxlen):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=-1)

class RNDNetwork(nn.Module):
    def __init__(self, obs_shape, feature_dim=512):
        super().__init__()
        c, h, w = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, feature_dim)
        )

    def forward(self, x):
        return self.net(x / 255.0)


class RNDVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, intrinsic_coef=1.0, lr=1e-4):
        super().__init__(venv)

        obs_shape = self.observation_space.shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target = RNDNetwork(obs_shape).to(self.device)
        self.predictor = RNDNetwork(obs_shape).to(self.device)

        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.intrinsic_coef = intrinsic_coef

    def reset(self):
        obs = self.venv.reset()
        return obs

    def set_logger(self, logger):
        self.logger = logger
        
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            target_feat = self.target(obs_t)

        pred_feat = self.predictor(obs_t)

        loss = (pred_feat - target_feat).pow(2).mean(dim=1)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        intrinsic_reward = loss.detach().cpu().numpy()

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.record(
                "rnd/intrinsic_reward_mean",
                float(intrinsic_reward.mean())
            )
            self.logger.record(
                "rnd/intrinsic_reward_max",
                float(intrinsic_reward.max())
            )

        rewards += self.intrinsic_coef * intrinsic_reward

        return obs, rewards, dones, infos



def make_env(seed=None, render_mode=None):
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)

    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    env = FrameStackFallback(env, 4)
    env = Monitor(env)
    return env


def make_vec_env(n_envs=8):
    vec_env = DummyVecEnv([lambda: make_env() for _ in range(n_envs)])
    vec_env = VecTransposeImage(vec_env)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )

    vec_env = RNDVecEnvWrapper(vec_env, intrinsic_coef=1.0)


    return vec_env

def train(chunks, steps_per_chunk, n_envs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "ppo_rnd_latest.zip")
    vecnorm_path = os.path.join(out_dir, "vecnormalize.pkl")

    env = make_vec_env(n_envs)

    if os.path.exists(model_path):
        env = VecNormalize.load(vecnorm_path, env)
        model = PPO.load(model_path, env=env)
        env.training = True
    else:
        model = PPO(
            "CnnPolicy",
            env,
            n_steps=2048,
            batch_size=256,
            n_epochs=3,
            gamma=0.999,
            learning_rate=2.5e-4,
            verbose=1,
            tensorboard_log=os.path.join(out_dir, "tensorboard"),
            policy_kwargs=dict(normalize_images=False)
        )


    logger_attached = False

    for c in range(chunks):
        print(f"\n=== Chunk {c+1}/{chunks} ===")
        start = time.time()

        model.learn(
            total_timesteps=steps_per_chunk,
            reset_num_timesteps=False,
            tb_log_name="PPO_RND"
        )

        if not logger_attached and hasattr(env, "set_logger"):
            env.set_logger(model.logger)
            logger_attached = True
            print("✓ Logger attached to RND wrapper")

        model.logger.dump(model.num_timesteps)
        model.save(model_path)
        env.save(vecnorm_path)

        print(f"Chunk time: {(time.time()-start)/60:.1f} min")


    env.close()

def evaluate(out_dir, episodes=3):
    env = DummyVecEnv([lambda: make_env(render_mode="human")])
    env = VecTransposeImage(env)
    env = VecNormalize.load(os.path.join(out_dir, "vecnormalize.pkl"), env)

    env.training = False
    env.norm_reward = False

    model = PPO.load(os.path.join(out_dir, "ppo_rnd_latest.zip"), env=env)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0]

        print(f"Episode {ep+1} reward: {ep_reward}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--out", type=str, default="models")
    args = parser.parse_args()

    if args.train:
        train(args.chunks, args.steps, args.envs, args.out)

    if args.eval:
        evaluate(args.out)
