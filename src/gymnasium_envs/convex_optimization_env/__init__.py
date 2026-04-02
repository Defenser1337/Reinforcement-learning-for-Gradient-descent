from gymnasium.envs.registration import register

register(
    id="convex_optimization_env/ConvexOptimization-v0",
    entry_point="src.gymnasium_envs.convex_optimization_env.envs.convex_optimization_v0:ConvexOptimizationV0",
)

from src.gymnasium_envs.convex_optimization_env.envs.convex_optimization_v0 import ConvexOptimizationV0

__all__ = ["ConvexOptimizationV0"]
