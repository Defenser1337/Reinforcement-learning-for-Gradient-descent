import numpy as np
import gymnasium as gym
import src.gymnasium_envs.convex_optimization_env  # noqa: F401 — registers the env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.optimization.optimization_methods import gradient_descent_optimizer, adam_optimizer


def _load_ppo_env(env_config: dict, model_dir: dict):
    """Shared loading of VecEnv + VecNormalize + PPO model for a single run."""
    env = make_vec_env(
        env_id=env_config["env_id"],
        n_envs=env_config["n_envs"],
        seed=env_config["seed"],
        env_kwargs=env_config["env_kwargs"],
    )
    env = VecNormalize.load(model_dir["stats"], env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_dir["model"], env=env, seed=env_config["seed"])
    return env, model


def make_standard_method_exp(function, x0, max_iterations, name="GD", add_noise=False) -> list:
    if name not in ("GD", "ADAM"):
        raise ValueError("Only ADAM and GD optimizers are supported.")

    gd_info = []
    optimizer = gradient_descent_optimizer if name == "GD" else adam_optimizer
    optimizer(function, x0=x0, opt_info=gd_info, max_iteration_count=max_iterations, add_noise=add_noise)
    return gd_info


def make_rl_method_exp(env_config: dict, model_dir: dict) -> tuple[list, object, np.ndarray]:
    env, model = _load_ppo_env(env_config, model_dir)

    obs = env.reset()
    x0 = env.envs[0].unwrapped.get_x_start()
    function = env.envs[0].unwrapped.get_function()

    opt_info = [[{'iteration': 0, 'loss': function(x0), 'x': x0.copy()}]]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        opt_info.append(info)
        done = terminated

    return opt_info, function, x0


def optimize_exp_standart(method="GD", x0=None, function=None, env_config=None, add_noise=False) -> dict:
    if method not in ("GD", "ADAM"):
        raise ValueError("Only ADAM and GD optimizers are supported.")

    method_name = "Gradient Descent" if method == "GD" else "ADAM"
    gd_info = make_standard_method_exp(
        function=function,
        x0=x0,
        max_iterations=env_config["env_kwargs"]["max_iterations"],
        name=method,
        add_noise=add_noise,
    )
    gd_it, gd_val = zip(*[(item['iteration'], item['function_value']) for item in gd_info])
    return {method_name: (gd_it, gd_val)}


def optimize_exp_rl(method: str, env_config=None, model_dir=None):
    if env_config is None or model_dir is None:
        raise ValueError("When using the RL model, all attributes must be specified.")

    gd_info, function, x0 = make_rl_method_exp(env_config, model_dir)
    # the last 2 elements of opt_info are internal VecEnv bookkeeping records
    # produced after "done" (e.g. terminal_observation / auto-reset), skip them for plotting
    gd_it, gd_val = zip(*[(item[0]['iteration'], item[0]['loss']) for item in gd_info[:-2]])

    return {method: (gd_it, gd_val)}, x0, function


def make_rl_method_exp_batch(env_config: dict, model_dir: dict, n_samples: int, base_rng):
    sub_seeds = [int(base_rng.integers(low=0, high=np.iinfo(np.uint32).max)) for _ in range(n_samples)]

    def make_env(rank):
        def _init():
            e = gym.make(env_config["env_id"], **env_config["env_kwargs"])
            e.reset(seed=sub_seeds[rank])
            return e
        return _init

    env = DummyVecEnv([make_env(i) for i in range(n_samples)])
    env = VecNormalize.load(model_dir["stats"], env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_dir["model"], env=env)

    obs = env.reset()  # duplicate reset — intentional, see the earlier discussion on "double reset"

    # NOTE: _curr_loss is a private attribute on ConvexOptimizationV1 — there is
    # no public accessor for it, so we access it directly here.
    final_loss = np.array([e.unwrapped._curr_loss for e in env.envs], dtype=float)
    iter_counts = np.zeros(n_samples, dtype=int)
    done_mask = np.zeros(n_samples, dtype=bool)

    while not done_mask.all():
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        for i in range(n_samples):
            if not done_mask[i]:
                final_loss[i] = infos[i]["loss"]
                iter_counts[i] += 1
                if dones[i]:
                    done_mask[i] = True

    return final_loss, iter_counts, env