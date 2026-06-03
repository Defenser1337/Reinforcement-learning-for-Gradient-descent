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

def optimize_exp_standart(method = "GD", x0 = None, function = None, env_config = None):
    if method not in ["GD", "ADAM"]:
        raise ValueError("Only ADAM and GD optimizers are supported.")

    method_name = "Gradient Descent" if method == "GD" else "ADAM"
    gd_info = make_standard_method_exp(function=function,
                                        x0 = x0,
                                        max_iterations=env_config["env_kwargs"]["max_iterations"],
                                        name=method)
    
    gd_it, gd_val = zip(*[(item['iteration'], item['function_value']) for item in gd_info])

    
    return {method_name : (gd_it, gd_val)}

def optimize_exp_rl(method, env_config = None, model_dir = None):
    if env_config is None or model_dir is None:
        raise ValueError("When using the RL model, all attributes must be specified.")
    
    method_name = method
    gd_info, function, x0 = make_rl_method_exp(env_config, model_dir)
    gd_it, gd_val = zip(*[(item[0]['iteration'], item[0]['loss']) for item in gd_info[:-2]])

    return {method_name : (gd_it, gd_val)}, x0, function

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

def make_standard_method_exp(function, x0, max_iterations, name = "GD") -> list:
    gd_info = []

    if name == "GD":
        gradient_descent_optimizer(function, x0=x0, opt_info=gd_info, max_iteration_count = max_iterations)
    elif name == "ADAM":
        adam_optimizer(function, x0=x0, opt_info=gd_info, max_iteration_count= max_iterations)
    else:
        raise ValueError("Only ADAM and GD optimizers are supported.")

    return gd_info

def plot_converging_comparasion(result : dict, dim : int, title = "_blank_name_"):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    for name, values in result.items():
        sns.lineplot(x=values[0], y=values[1], label=name)

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()    
    plt.yscale('log')
    plt.show()

def plot_comparasion_table(result : dict):
    table_data = [{
        "Name" : name,
        "Iteration count" : len(values[0]),
        "Loss" : values[1][-1]
        } 
    for name, values in result.items()]
    
    df = pd.DataFrame(table_data)

    return df

def plot_iterations_distribution_vs_standart(sample_count, env_config, model_dir):
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