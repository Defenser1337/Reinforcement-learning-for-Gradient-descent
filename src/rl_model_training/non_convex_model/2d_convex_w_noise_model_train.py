import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import src.gymnasium_envs.non_convex_optimization_env
from torch import nn

log_dir = "logs"
model_dir = "models"
dim = 2

config = {
    2 : {
        "timesteps": 100_000, 
        "n_envs": 32,
        "n_steps" : 1024
    }
}

vec_env = make_vec_env(
    "non_convex_optimization_env/NonConvexOptimization-v0", 
    n_envs=config[dim]["n_envs"],
    env_kwargs={
        "in_features": dim,
        "max_iterations" : 5000,
    }
)

vec_env = VecNormalize(
    vec_env, 
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0
)

model = PPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    n_steps=config[dim]["n_steps"],
    tensorboard_log=f"{log_dir}/noisy/{dim}d/",
    device="cuda"
)

model.learn(total_timesteps=config[dim]["timesteps"])

model.save(f"{model_dir}/{dim}d_noise_convex_optimization")
vec_env.save(f"{model_dir}/{dim}d_noise_convex_optimization_vec_normalize_stats.pkl")