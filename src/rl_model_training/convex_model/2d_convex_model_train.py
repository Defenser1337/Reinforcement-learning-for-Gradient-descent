import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import src.gymnasium_envs.convex_optimization_env

log_dir = "logs"
model_dir = "models"
dim = 2

config = {
    "timesteps": 10_000, 
    "n_envs": 14,
}

vec_env = make_vec_env(
    "convex_optimization_env/ConvexOptimization-v1", 
    n_envs=config["n_envs"],
    env_kwargs={
        "in_features": dim
    })

vec_env = VecNormalize(
    vec_env, 
    clip_obs=10.0
)

model = PPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,  
    tensorboard_log=f"{log_dir}/{dim}d/"
)

model.learn(total_timesteps=config["timesteps"])

model.save(f"{model_dir}/{dim}d_convex_optimization")
vec_env.save(f"{model_dir}/{dim}d_convex_optimization_vec_normalize_stats.pkl")