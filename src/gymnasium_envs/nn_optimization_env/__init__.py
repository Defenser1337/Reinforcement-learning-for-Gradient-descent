from gymnasium.envs.registration import register

register(
    id="nn_optimization_env/NeuralNetworkOptimization-v1",
    entry_point="src.gymnasium_envs.nn_optimization_env.envs.nn_optimization_v1:NeuralNetworkOptimizationV1",
)

from src.gymnasium_envs.nn_optimization_env.envs.nn_optimization_v1 import NeuralNetworkOptimizationV1


__all__ = ["NeuralNetworkOptimizationV1"]