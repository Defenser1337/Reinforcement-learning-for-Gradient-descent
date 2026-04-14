from gymnasium.envs.registration import register

register(
    id="non_convex_optimization_env/NonConvexOptimization-v0",
    entry_point="src.gymnasium_envs.non_convex_optimization_env.envs.non_convex_optimization_v0:NonConvexOptimizationV0",
)

from src.gymnasium_envs.non_convex_optimization_env.envs.non_convex_optimization_v0 import NonConvexOptimizationV0

