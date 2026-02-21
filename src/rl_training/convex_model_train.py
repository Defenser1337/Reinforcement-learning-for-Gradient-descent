import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import src.gymnasium_envs.convex_optimization_env

vec_env = make_vec_env("convex_optimization_env/ConvexOptimization-v0", n_envs=14)

vec_env = VecNormalize(
    vec_env, 
    clip_obs=10.0
)

model = PPO(
    "MultiInputPolicy", 
    vec_env, 
    verbose=1
)

model.learn(total_timesteps=100000)

model.save("convex_optimization")
vec_env.save("vec_normalize_stats.pkl")