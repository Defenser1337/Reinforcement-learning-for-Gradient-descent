import logging
import time
from pathlib import Path
from typing import Dict, Any

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import src.gymnasium_envs.convex_optimization_env

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")

TRAINING_CONFIGS = {
    5: {"timesteps": 1_000_000, 
        "n_envs": 32, 
        "batch_size": 256, 
        "policy_kwargs": {
            "net_arch": dict(pi=[512, 512], vf=[512, 512])
        }},
    10: {"timesteps": 1_000_000, 
         "n_envs": 32, 
         "batch_size": 256, 
         "policy_kwargs": {
            "net_arch": dict(pi=[512, 512], vf=[512, 512])
        }},
    100: {"timesteps": 1_000_000, 
          "n_envs": 32, 
          "batch_size": 256, 
          "policy_kwargs": {
            "net_arch": dict(pi=[512, 512], vf=[512, 512])
        }},
    800: {
        "timesteps": 1_000_000,
        "n_envs": 32,
        "batch_size": 256,
        "policy_kwargs": {
            "net_arch": dict(pi=[512, 512], vf=[512, 512])
        }
    }
}

def train_dimension_model(dim: int, config: Dict[str, Any], log_dir: str = "logs"):
    """
    Выполняет цикл обучения модели PPO для заданной размерности.
    """
    logger.info(f"Starting training for {dim}D task. Timesteps: {config['timesteps']}, Envs: {config['n_envs']}")
    
    start_time = time.time()
    model_path = Path(f"models/{dim}d_convex_optimization")
    stats_path = Path(f"models/{dim}d_convex_optimization_vec_normalize_stats.pkl")
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    vec_env = None
    try:
        vec_env = make_vec_env(
            "convex_optimization_env/ConvexOptimization-v0",
            n_envs=config["n_envs"],
            env_kwargs={"in_features": dim}
        )

        vec_env = VecNormalize(
            vec_env,
            clip_obs=10.0,
            norm_obs=True,
            norm_reward=True
        )
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=1,  
            tensorboard_log=f"{log_dir}/{dim}d/",
            policy_kwargs=config["policy_kwargs"]
        )

        model.learn(
            total_timesteps=config["timesteps"],
            tb_log_name=f"PPO_{dim}d"
        )

        model.save(str(model_path))
        vec_env.save(str(stats_path))

        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"Training for {dim}D completed successfully. Duration: {elapsed_time:.2f} min.")
        logger.info(f"Model saved to: {model_path}")

    except Exception as e:
        logger.error(f"Error during training for {dim}D: {str(e)}", exc_info=True)
    
    finally:
        if vec_env is not None:
            vec_env.close()
            logger.debug(f"Environment for {dim}D closed.")

def main():
    logger.info("Initializing production training pipeline")
    
    for dim, config in TRAINING_CONFIGS.items():
        train_dimension_model(dim, config)
        
    logger.info("All training tasks finished.")

if __name__ == "__main__":
    main()