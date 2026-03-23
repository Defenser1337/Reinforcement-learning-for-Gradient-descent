import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import src.gymnasium_envs.convex_optimization_env

log_dir = "logs"
model_dir = "models"
dim = 2

config = {
    2 : {"timesteps": 100_000, "n_envs": 14, "batch_size": 1024, "policy_kwargs": {}}
}

vec_env = make_vec_env(
    "convex_optimization_env/ConvexOptimization-v0", 
    n_envs=config[dim]["n_envs"],
    env_kwargs={
        "in_features": dim,
        "add_noise" : True
    })

vec_env = VecNormalize(
    vec_env, 
    clip_obs=10.0
)

model = PPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,  
    tensorboard_log=f"{log_dir}/{dim}d/",
    policy_kwargs=config[dim]["policy_kwargs"]
)

model.learn(total_timesteps=config[dim]["timesteps"])

model.save(f"{model_dir}/{dim}d_convex_w_noise_optimization")
vec_env.save(f"{model_dir}/{dim}d_convex_w_noise_optimization_vec_normalize_stats.pkl")