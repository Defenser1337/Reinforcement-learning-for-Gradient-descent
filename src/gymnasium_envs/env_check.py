import gymnasium
from stable_baselines3.common.env_checker import check_env

import src.gymnasium_envs.convex_optimization_env
import src.gymnasium_envs.nn_optimization_env

env1 = gymnasium.make("convex_optimization_env/ConvexOptimization-v0")
env2 = gymnasium.make("convex_optimization_env/ConvexOptimization-v1")
env3 = gymnasium.make("nn_optimization_env/NeuralNetworkOptimization-v0", dataset_name="MNIST")

check_env(env1)
check_env(env2)
check_env(env3)