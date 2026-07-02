from collections import defaultdict

import numpy as np
import pandas as pd

from src.utils.prng import get_rng
from .experiments import (
    optimize_exp_rl,
    optimize_exp_standart,
    make_standard_method_exp,
    make_rl_method_exp_batch,
)


def _compute_stats_table(data: dict) -> pd.DataFrame:
    """Shared entry point for computing statistics over accumulated samples."""
    return pd.DataFrame({
        algo: {
            'mean': np.mean(vals),
            'variance': np.var(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
            'range': np.ptp(vals),
        }
        for algo, vals in data.items()
    }).T.round(2)


def plot_iterations_distribution_vs_standart(sample_count: int, env_config: dict, model_dir: dict) -> pd.DataFrame:
    env_config = dict(env_config)
    data = defaultdict(list)
    rng = get_rng(env_config["seed"], location_name="plot_iterations_distribution_vs_standart_function")

    dim = env_config["env_kwargs"]["in_features"]
    rl_name = f"Adaptive LR (dim={dim})"

    for _ in range(sample_count):
        env_config["seed"] = int(rng.integers(low=0, high=np.iinfo(np.uint32).max))

        result_rl, x0, function = optimize_exp_rl(method=rl_name, env_config=env_config, model_dir=model_dir)
        result_gdesc = optimize_exp_standart(method="GD", x0=x0, function=function, env_config=env_config)
        result_adam = optimize_exp_standart(method="ADAM", x0=x0, function=function, env_config=env_config)

        for name, values in (result_rl | result_gdesc | result_adam).items():
            data[name].append(len(values[0]))

    return _compute_stats_table(data)


def plot_iterations_distribution(sample_count: int, env_config: dict, models: dict) -> pd.DataFrame:
    base_config = dict(env_config)
    data = defaultdict(list)
    rng = get_rng(base_config["seed"], location_name="plot_iterations_distribution_vs_standart_function")

    for _ in range(sample_count):
        current_config = dict(base_config)
        current_config["seed"] = int(rng.integers(low=0, high=np.iinfo(np.uint32).max))

        for model_name, model_dir in models.items():
            # FIX: previously the original env_config was always passed here,
            # so the seed change inside the loop never took effect
            result_rl, _, _ = optimize_exp_rl(method=model_name, env_config=current_config, model_dir=model_dir)
            for name, values in result_rl.items():
                data[name].append(len(values[0]))

    return _compute_stats_table(data)

def plot_iterations_distribution_batched(sample_count: int, env_config: dict, models: dict) -> pd.DataFrame:
    """Same as plot_iterations_distribution, but each model is evaluated on
    `sample_count` environments in parallel via a single DummyVecEnv batch,
    instead of sequential single-env rollouts. Much faster for large sample_count."""
    env_config = dict(env_config)
    data = {}

    rng = get_rng(env_config["seed"], location_name="plot_iterations_distribution_batched")

    for model_name, model_dir in models.items():
        _, iter_counts, _ = make_rl_method_exp_batch(env_config, model_dir, sample_count, rng)
        data[model_name] = iter_counts.tolist()

    return _compute_stats_table(data)


def compute_best_method_rates_batched(sample_count: int, env_config: dict, model_dir: dict, tol=1e-6) -> dict:
    env_config = dict(env_config)
    rng = get_rng(env_config["seed"], location_name="compute_best_method_rates")

    final_rl, _, env = make_rl_method_exp_batch(env_config, model_dir, sample_count, rng)

    counts = {"AdaRL": 0, "GD": 0, "ADAM": 0}
    max_iterations = env_config["env_kwargs"]["max_iterations"]

    for i in range(sample_count):
        x0 = env.envs[i].unwrapped.get_x_start()
        function = env.envs[i].unwrapped.get_function()

        gd_info = make_standard_method_exp(function=function, x0=x0, max_iterations=max_iterations,
                                            name="GD", add_noise=True)
        adam_info = make_standard_method_exp(function=function, x0=x0, max_iterations=max_iterations,
                                              name="ADAM", add_noise=True)

        values = {
            "AdaRL": final_rl[i],
            "GD": gd_info[-1]['function_value'],
            "ADAM": adam_info[-1]['function_value'],
        }
        best_val = min(values.values())
        for name, val in values.items():
            if np.isclose(val, best_val, rtol=tol, atol=tol):
                counts[name] += 1

    return {name: 100.0 * c / sample_count for name, c in counts.items()}