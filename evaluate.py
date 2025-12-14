import gymnasium as gym
import ale_py
import numpy as np
import collections
import argparse

from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
    VecNormalize
)
from stable_baselines3.common.monitor import Monitor

gym.register_envs(ale_py)

class FrameStackFallback(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        h, w = env.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, num_stack),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=-1)

def make_eval_env(render=False, seed=42):
    env = gym.make(
        "ALE/MontezumaRevenge-v5",
        render_mode="human" if render else None
    )
    env.reset(seed=seed)

    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    env = FrameStackFallback(env, num_stack=4)
    env = Monitor(env)

    return env

def evaluate(model_path, vecnorm_path=None, episodes=5, render=False):
    env = DummyVecEnv([lambda: make_eval_env(render=render)])
    env = VecTransposeImage(env)

    if vecnorm_path is not None:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False        
        env.norm_reward = False    

    model = PPO.load(model_path, env=env)

    episode_lengths = []
    episode_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_len = 0
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

            ep_len += 1
            ep_reward += reward[0]

        episode_lengths.append(ep_len)
        episode_rewards.append(ep_reward)

        print(
            f"Episode {ep+1}: "
            f"length = {ep_len}, reward = {ep_reward}"
        )

    env.close()

    print("\n=== Evaluation Summary ===")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    print(f"Std episode length : {np.std(episode_lengths):.1f}")
    print(f"Mean reward        : {np.mean(episode_rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PPO model checkpoint (.zip)"
    )
    parser.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        help="Path to VecNormalize stats (.pkl). Required for PPO+RND."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment"
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        episodes=args.episodes,
        render=args.render
    )
