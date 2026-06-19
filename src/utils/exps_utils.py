import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import gymnasium as gym
import src.gymnasium_envs.convex_optimization_env

from src.optimization.optimization_functions.convex_function_w_noise import ConvexFunctionWithNoise
from src.optimization.optimization_methods import gradient_descent_optimizer, adam_optimizer
from src.optimization.optimization_functions.convex_function import ConvexFunction

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.utils.prng import get_rng


def get_env_config(seed, in_features, max_iterations, env_id, env_kwargs):
    base_kwargs = {
        "render_mode": "ansi", 
        "in_features": in_features,
        "max_iterations": max_iterations
    }

    if env_kwargs:
        base_kwargs.update(env_kwargs)

    return {
        "env_id": env_id,
        "n_envs": 1,
        "seed": seed,
        "env_kwargs": base_kwargs
    }

def get_model_dir(stats, model):
    return {
        "stats" : stats,
        "model" : model
    }


def make_standard_method_exp(function, x0, max_iterations, name = "GD", add_noise = False) -> list:
    gd_info = []

    if name == "GD":
        gradient_descent_optimizer(function, x0=x0, opt_info=gd_info, max_iteration_count = max_iterations, add_noise = add_noise)
    elif name == "ADAM":
        adam_optimizer(function, x0=x0, opt_info=gd_info, max_iteration_count= max_iterations, add_noise = add_noise)
    else:
        raise ValueError("Only ADAM and GD optimizers are supported.")

    return gd_info

def make_rl_method_exp(env_config, model_dir) -> tuple[list, object, np.array]: 
    env = make_vec_env(
        env_id=env_config["env_id"],
        n_envs=env_config["n_envs"],
        seed=env_config["seed"],
        env_kwargs=env_config["env_kwargs"]
    )

    env = VecNormalize.load(model_dir["stats"], env)
    env.training = False
    env.norm_reward = False 

    model = PPO.load(model_dir["model"], env=env, seed=env_config["seed"])

    obs = env.reset()
    x0 = env.envs[0].unwrapped.get_x_start()

    function = env.envs[0].unwrapped.get_function()

    opt_info = [[{
        'iteration': 0, 
        'loss': function(x0), 
        'x': x0.copy()
    }]]

    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, info = env.step(action)
        
        opt_info.append(info)

        done = terminated

    return opt_info, function, x0 

def optimize_exp_standart(method = "GD", x0 = None, function = None, env_config = None, add_noise = False):
    if method not in ["GD", "ADAM"]:
        raise ValueError("Only ADAM and GD optimizers are supported.")

    method_name = "Градиентный спуск" if method == "GD" else "ADAM"
    gd_info = make_standard_method_exp(function=function,
                                        x0 = x0,
                                        max_iterations=env_config["env_kwargs"]["max_iterations"],
                                        name=method,
                                        add_noise = add_noise)
    
    gd_it, gd_val = zip(*[(item['iteration'], item['function_value']) for item in gd_info])

    
    return {method_name : (gd_it, gd_val)}


def optimize_exp_rl(method, env_config = None, model_dir = None):
    if env_config is None or model_dir is None:
        raise ValueError("When using the RL model, all attributes must be specified.")
    
    method_name = method
    gd_info, function, x0 = make_rl_method_exp(env_config, model_dir)
    gd_it, gd_val = zip(*[(item[0]['iteration'], item[0]['loss']) for item in gd_info[:-2]])

    return {method_name : (gd_it, gd_val)}, x0, function

def plot_converging_comparasion(result : dict, dim : int, title = "_blank_name_"):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    line_styles = [
        '-',        
        '--',      
        '-.',       
        ':',        
        (0, (3, 5, 1, 5)) 
    ]

    for i, (name, values) in enumerate(result.items()):
        sns.lineplot(
            x=values[0], 
            y=values[1], 
            label=name,
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2.5  # Делаем линию чуть толще, чтобы штрихи ЧБ печати читались четко
        )

    plt.title(title, fontsize=18)
    plt.xlabel('Шаг оптимизации (log)', fontsize=16)
    plt.ylabel('Значение функции (log)', fontsize=16)
    plt.legend(fontsize=14)    
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def plot_iterations_distribution_vs_standart(sample_count, env_config, model_dir):
    env_config = dict(env_config)

    data = {}

    rng = get_rng(env_config["seed"], location_name="plot_iterations_distribution_vs_standart_function")

    for i in range(sample_count):
        sub_seed = int(rng.integers(low=0, high=np.iinfo(np.uint32).max))
        env_config["seed"] = sub_seed

        result_rl, x0, function = optimize_exp_rl(method=f"Adaptive LR (dim={env_config["env_kwargs"]["in_features"]})", env_config=env_config, model_dir=model_dir)
        result_gdesc = optimize_exp_standart(method="GD", x0=x0, function=function, env_config=env_config)
        result_adam = optimize_exp_standart(method="ADAM", x0=x0, function=function, env_config=env_config)

        result = result_rl | result_gdesc | result_adam

        if i == 0:
            for name, values in result.items():
                data.setdefault(name, [len(values[0])])
            continue
        
        for name, values in result.items():
            data[name].append(len(values[0]))

    stats = pd.DataFrame({
        algo: {
            'mean': np.mean(vals),
            'variance': np.var(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
            'range': np.ptp(vals)
        }
        for algo, vals in data.items()
    }).T.round(2)

    return stats


def plot_iterations_distribution(sample_count, env_config, models):
    data = {}

    base_config = env_config.copy()

    rng = get_rng(base_config["seed"], location_name="plot_iterations_distribution_vs_standart_function")

    for i in range(sample_count):
        sub_seed = int(rng.integers(low=0, high=np.iinfo(np.uint32).max))

        current_config = base_config.copy()
        current_config["seed"] = sub_seed

        result = {}

        for model_name, model_dir in models.items():
            result_rl, _, _ = optimize_exp_rl(method=model_name, env_config=env_config, model_dir=model_dir)
            result.update(result_rl)

        
        if i == 0:
            for name, values in result.items():
                data.setdefault(name, [len(values[0])])
        else:
            for name, values in result.items():
                data[name].append(len(values[0]))

    stats = pd.DataFrame({
        algo: {
            'mean': np.mean(vals),
            'variance': np.var(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
            'range': np.ptp(vals)
        }
        for algo, vals in data.items()
    }).T.round(2)

    return stats


def make_rl_method_exp_batch(env_config, model_dir, n_samples, base_rng):

    sub_seeds = [int(base_rng.integers(low=0, high=np.iinfo(np.uint32).max)) for _ in range(n_samples)]

    env_kwargs_list = []
    for s in sub_seeds:
        cfg = dict(env_config["env_kwargs"])
        env_kwargs_list.append(cfg)

    def make_env(rank):
        def _init():
            e = gym.make(env_config["env_id"], **env_config["env_kwargs"])
            e.reset(seed=sub_seeds[rank])
            return e
        return _init

    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([make_env(i) for i in range(n_samples)])
    env = VecNormalize.load(model_dir["stats"], env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_dir["model"], env=env)

    obs = env.reset()  # повторный reset — ок, см. предыдущий разбор про "двойной reset"

    final_loss = np.array([e.unwrapped._curr_loss for e in env.envs], dtype=float)
    done_mask = np.zeros(n_samples, dtype=bool)

    while not done_mask.all():
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        for i in range(n_samples):
            if not done_mask[i]:
                final_loss[i] = infos[i]["loss"]
                if dones[i]:
                    done_mask[i] = True

    return final_loss, env

def compute_best_method_rates_batched(sample_count, env_config, model_dir, tol=1e-6):
    env_config = dict(env_config)
    rng = get_rng(env_config["seed"], location_name="compute_best_method_rates")

    final_rl, env = make_rl_method_exp_batch(env_config, model_dir, sample_count, rng)

    counts = {"AdaRL": 0, "GD": 0, "ADAM": 0}
    max_iterations = env_config["env_kwargs"]["max_iterations"]

    for i in range(sample_count):
        x0 = env.envs[i].unwrapped.get_x_start()
        function = env.envs[i].unwrapped.get_function()

        gd_info = make_standard_method_exp(function=function, x0=x0,
                                            max_iterations=max_iterations, name="GD", add_noise=True)
        adam_info = make_standard_method_exp(function=function, x0=x0,
                                              max_iterations=max_iterations, name="ADAM", add_noise=True)

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