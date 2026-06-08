import numpy as np
from typing import Optional
import itertools

import gymnasium as gym
from gymnasium import spaces

import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch

from src.nn_models.lenet import LeNet
from src.gymnasium_envs.nn_optimization_env.envs.custom_lr_v1 import CustomLRV1

EPS = 1e-8

BETA1_LOWER_BOUND = 0.5
BETA1_UPPER_BOUND = 0.999
BETA2_LOWER_BOUND = 0.9
BETA2_UPPER_BOUND = 0.9999

class NeuralNetworkOptimizationV1(gym.Env):
    """
    Represents neural network optimization task process.

    Gymnasium custom environment: 
    https://gymnasium.farama.org/introduction/create_custom_env/

    Version 1:  
        Reworked observation values:
            Deleted raw norm features, added EMA normalization (mirrors ConvexOptimizationV1).
            Observation building and state tracking delegated to CustomLR.

    Observation values:
        grad_norm_scaled_log: 
            Logarithm of normalized gradient L2 norm log(1 + ||∇F||₂ / (EMA(||∇F||₂) + ϵ)),
        grad_delta_norm_scaled_log: 
            Logarithm of normalized norm of difference between gradient log(1 + ||∇F(Xt)-∇F(Xt-1)||₂ / (EMA(||∇F(Xt)-∇F(Xt-1)||₂) + ϵ)),
        cos_sim: 
            Cosine similarity between gradients Cos_sim(∇F(Xt), ∇F(Xt-1)),
        loss_scaled_log:
            Logarithm of normalized loss log(1 + F(X) / (EMA(F) + ϵ))
        loss_delta_scaled_log:
            Logarithm of normalized difference between losses sign(ΔF) * log(1 + |ΔF| / (|EMA(ΔF)| + ϵ))
        prev_action:
            Previous action

    Action value:
        lr: normalized learning rate in logarithmic scale in [-1, 1]

    Parameters:
        render_mode : Only "ansi" or None
        dataset_name (str) : Name of the dataset to use
        max_iterations (int) : Maximum number of gradient steps
        batch_size (int) : Batch size for dataloader
        add_time_penalty (bool) : Adds a time penalty to the reward
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        render_mode=None,
        dataset_name: Optional[str] = None,
        max_iterations: int = 3000,
        batch_size: int = 256,
        learn_betas: bool = False,
        add_time_penalty: bool = False,
    ):
        self.render_mode = render_mode
        self.dataset_name = dataset_name
        self._max_iterations = max_iterations
        self._batch_size = batch_size
        self._learn_betas = learn_betas
        self._add_time_penalty = add_time_penalty

        obs = {
            "grad_norm_scaled_log":       spaces.Box(low=0.0,    high=100.0, shape=(1,), dtype=np.float32),
            "grad_delta_norm_scaled_log": spaces.Box(low=0.0,    high=100.0, shape=(1,), dtype=np.float32),
            "cos_sim":                    spaces.Box(low=-1.0,   high=1.0,   shape=(1,), dtype=np.float32),
            "loss_scaled_log":            spaces.Box(low=0.0,    high=100.0, shape=(1,), dtype=np.float32),
            "loss_delta_scaled_log":      spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32),
            "prev_lr":                    spaces.Box(low=-1.0,   high=1.0,   shape=(1,), dtype=np.float32),
        }

        if self._learn_betas is True:
            obs["prev_beta1"] = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            obs["prev_beta2"] = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict(obs)

        if self._learn_betas is True:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        

        # Dataset loading (done once at construction)
        if dataset_name is None:
            raise ValueError("Missing dataset_name")
        elif dataset_name == "MNIST":
            train_dataset = datasets.MNIST(root="./data", train=True,  download=True)
            test_dataset  = datasets.MNIST(root="./data", train=False, download=True)

            X_train = train_dataset.data.unsqueeze(1).float() / 255.0
            y_train = train_dataset.targets.long()

            X_test = test_dataset.data.unsqueeze(1).float() / 255.0
            y_test = test_dataset.targets.long()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self._train_ds    = TensorDataset(X_train, y_train)
        self._test_ds     = TensorDataset(X_test,  y_test)
        self._test_loader = DataLoader(self._test_ds, batch_size=self._batch_size, shuffle=False)

    # Gymnasium API
    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        train_loader  = DataLoader(self._train_ds, batch_size=self._batch_size, shuffle=True, generator=g)
        self._train_loader = itertools.cycle(train_loader)

        self._iteration = 0

        self._prev_beta1 = 0.0
        self._prev_beta2 = 0.0
        self._prev_lr = 0.0

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.dataset_name == "MNIST":
            self._model = LeNet(seed=seed).to(self._device)
            self._criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self._optimizer = CustomLRV1(self._model.parameters(), learn_betas=self._learn_betas)
        self._optimizer.zero_grad()

        # First forward-backward pass to populate gradients
        self._curr_loss = self._compute_gradients()

        observation = self._optimizer.get_obs(loss=self._curr_loss, 
                                              prev_lr=self._prev_lr, 
                                              prev_beta1=self._prev_beta1, 
                                              prev_beta2=self._prev_beta2)
        info = self._optimizer.get_info(self._iteration)

        return observation, info

    def step(self, action: np.ndarray):
        self._iteration += 1

        if self._learn_betas is True:
            lr = 10 ** (action[0] * 3 - 2) # -> [0.00001, 10]
            beta1 = 0.5 + 0.499 * (action[1] + 1) / 2   # -> [0.5, 0.999]
            beta2 = 0.9 + 0.0999 * (action[2] + 1) / 2  # -> [0.9, 0.9999]

            self._prev_lr = float(action[0])
            self._prev_beta1 = float(action[1])
            self._prev_beta2 = float(action[2])

            self._optimizer.step(lr=lr, beta1=beta1, beta2=beta2)
        else:
            lr = 10 ** (action[0] * 3 - 2)# -> [0.00001, 10]

            self._prev_lr = float(action[0])

            self._optimizer.step(lr=lr)


        prev_loss = self._curr_loss
        self._curr_loss = self._compute_gradients()

        observation = self._optimizer.get_obs(loss=self._curr_loss, 
                                              prev_lr=self._prev_lr, 
                                              prev_beta1=self._prev_beta1, 
                                              prev_beta2=self._prev_beta2)
        info = self._optimizer.get_info(self._iteration)

        reward = float(np.log2(prev_loss + EPS) - np.log2(self._curr_loss + EPS))
        reward = np.clip(reward, -5.0, 5.0)

        if self._add_time_penalty:
            time_penalty = self._iteration / self._max_iterations
            reward -= 0.1 * time_penalty

        diverged = self._optimizer.is_diverged()
        truncated = bool(self._iteration >= self._max_iterations)

        if diverged:
            reward += -10.0
            terminated = True
            info["status"] = "diverged"
        else:
            terminated = False

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            info = self._optimizer.get_info(self._iteration)
            return (
                f"\n--- Iteration {self._iteration} ---\n"
                f"Loss:          {info['loss']:.6f}\n"
                f"Gradient Norm: {info['grad_norm']:.6f}\n"
            )

    def close(self):
        pass


    def _compute_gradients(self) -> float:
        """One mini-batch forward + backward pass. Returns scalar loss."""
        inputs, labels = next(self._train_loader)
        inputs, labels = inputs.to(self._device), labels.to(self._device)

        self._optimizer.zero_grad()
        loss = self._criterion(self._model(inputs), labels)
        loss.backward()

        return float(loss.item())