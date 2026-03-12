import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import src.gymnasium_envs.convex_optimization_env

vec_env = make_vec_env(
    "convex_optimization_env/ConvexOptimization-v0", 
    n_envs=14,
    env_kwargs={
        "in_features": 2
    })

vec_env = VecNormalize(
    vec_env, 
    clip_obs=10.0
)

model = PPO(
    "MultiInputPolicy", 
    vec_env, 
    verbose=1
)

model.learn(total_timesteps=250000)

model.save("2d_convex_optimization")
vec_env.save("2d_convex_optimization_vec_normalize_stats.pkl")