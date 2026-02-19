from gymnasium.envs.registration import register

register(
    id="convex_optimization_env/GridWorld-v0",
    entry_point="convex_optimization_env.envs:GridWorldEnv",
)
