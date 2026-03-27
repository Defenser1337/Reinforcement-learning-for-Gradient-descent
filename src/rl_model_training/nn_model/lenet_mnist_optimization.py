import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import src.gymnasium_envs.nn_optimization_env
import torch

log_dir = "logs"
model_dir = "models"

config = {
    "timesteps": 1_000_000, 
    "n_envs": 14,
    "n_steps" : 512,
    "batch_size": 256,
    "ent_coef" : 0.01, 
    "policy_kwargs": {
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        "activation_fn": torch.nn.Tanh
    }
}

vec_env = make_vec_env(
    "nn_optimization_env/NeuralNetworkOptimization-v0", 
    n_envs=config["n_envs"],
    env_kwargs={
        "dataset_name": "MNIST"
    })

vec_env = VecNormalize(
    vec_env, 
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0
)

model = PPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    n_steps=config["n_steps"],
    batch_size=config["batch_size"],
    tensorboard_log=f"{log_dir}/lenet_mnist/",
    policy_kwargs=config["policy_kwargs"]
)

model.learn(total_timesteps=config["timesteps"])

model.save(f"{model_dir}/lenet_mnist_optimization")
vec_env.save(f"{model_dir}/lenet_mnist_convex_optimization_vec_normalize_stats.pkl")