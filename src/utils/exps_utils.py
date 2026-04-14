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


def get_env_config(seed, in_features, max_iterations, add_noise):
    return {
        "env_id" : "convex_optimization_env/ConvexOptimization-v1",
        "n_envs" : 1,
        "seed" : seed,
        "env_kwargs": {
            "render_mode": "ansi", 
            "in_features": in_features,
            "add_noise" : add_noise,
            "max_iterations" : max_iterations
        }
    }

def get_model_dir(stats, model):
    return {
        "stats" : "../models/2d_convex_optimization_vec_normalize_stats.pkl",
        "model" : "../models/2d_convex_optimization"
    }

def optimize_exp(env_config, model_dir):
    rl_gd_info, function, x0 = make_rl_method_exp(env_config, model_dir)

    gd_info, adam_info = make_standard_method_exp(function=function,
                                                  x0 = x0,
                                                  max_iterations=env_config["env_kwargs"]["max_iterations"],)
    
    gd_it, gd_val = zip(*[(item['iteration'], item['function_value']) for item in gd_info])
    adam_it, adam_val = zip(*[(item['iteration'], item['function_value']) for item in adam_info])
    rl_gd_it, rl_gd_val = zip(*[(item[0]['iteration'], item[0]['loss']) for item in rl_gd_info[:-2]])

    return {
        "Gradient Descent" : (gd_it, gd_val),
        "ADAM" : (adam_it, adam_val),
        "GD with adaptive LR" : (rl_gd_it, rl_gd_val)
    }

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

def make_standard_method_exp(function, x0, max_iterations) -> tuple[list, list]:
    gd_info = []
    adam_info = []

    gradient_descent_optimizer(function, x0=x0, opt_info=gd_info, max_iteration_count = max_iterations)
    adam_optimizer(function, x0=x0, opt_info=adam_info, max_iteration_count= max_iterations)

    return gd_info, adam_info

def plot_converging_comparasion(result : dict, dim : int):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    for name, values in result.items():
        sns.lineplot(x=values[0], y=values[1], label=name)

    plt.title(f'Convex optimization task (n_features={dim}): ADAM vs GD vs GD with adaptive LR (log-scale)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log-scale)')
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

def plot_iterations_distribution(sample_count, env_config, model_dir):
    data = {}

    rng = np.random.default_rng(env_config["seed"])

    for i in range(sample_count):
        sub_seed = int(rng.integers(low=0, high=np.iinfo(np.uint32).max))
        env_config["seed"] = sub_seed

        result = optimize_exp(env_config, model_dir)

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