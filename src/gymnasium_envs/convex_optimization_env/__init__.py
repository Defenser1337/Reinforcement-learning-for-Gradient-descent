from gymnasium.envs.registration import register

register(
    id="convex_optimization_env/ConvexOptimization-v0",
    entry_point="src.gymnasium_envs.convex_optimization_env.envs.convex_optimization_v0:ConvexOptimizationV0",
)

register(
    id="convex_optimization_env/ConvexOptimization-v1",
    entry_point="src.gymnasium_envs.convex_optimization_env.envs.convex_optimization_v1:ConvexOptimizationV1",
)

from src.gymnasium_envs.convex_optimization_env.envs.convex_optimization_v0 import ConvexOptimizationV0
from src.gymnasium_envs.convex_optimization_env.envs.convex_optimization_v1 import ConvexOptimizationV1

__all__ = ["ConvexOptimizationV0", "ConvexOptimizationV1"]
