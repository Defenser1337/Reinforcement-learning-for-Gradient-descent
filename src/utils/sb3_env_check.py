import gymnasium
import src.gymnasium_envs.convex_optimization_env

from stable_baselines3.common.env_checker import check_env

env = gymnasium.make("convex_optimization_env/ConvexOptimization-v0")

check_env(env)