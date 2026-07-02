def get_env_config(seed, in_features, max_iterations, env_id, env_kwargs=None) -> dict:
    base_kwargs = {
        "render_mode": "ansi",
        "in_features": in_features,
        "max_iterations": max_iterations,
    }
    if env_kwargs:
        base_kwargs.update(env_kwargs)

    return {
        "env_id": env_id,
        "n_envs": 1,
        "seed": seed,
        "env_kwargs": base_kwargs,
    }


def get_model_dir(stats, model) -> dict:
    return {"stats": stats, "model": model}