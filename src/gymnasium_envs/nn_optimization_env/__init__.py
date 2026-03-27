from gymnasium.envs.registration import register

register(
    id="nn_optimization_env/NeuralNetworkOptimization-v0",
    entry_point="src.gymnasium_envs.nn_optimization_env.envs.nn_optimization:NeuralNetworkOptimization",
)

from src.gymnasium_envs.nn_optimization_env.envs.nn_optimization import NeuralNetworkOptimization

__all__ = ["NeuralNetworkOptimization"]