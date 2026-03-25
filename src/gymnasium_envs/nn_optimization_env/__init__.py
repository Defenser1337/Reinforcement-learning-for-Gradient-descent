from gymnasium.envs.registration import register

register(
    id="convex_optimization_env/ConvexOptimization-v0",
    entry_point="src.gymnasium_envs.convex_optimization_env.envs.convex_optimization:ConvexOptimization",
)

from src.gymnasium_envs.convex_optimization_env.envs.convex_optimization import ConvexOptimization

__all__ = ["ConvexOptimization"]
