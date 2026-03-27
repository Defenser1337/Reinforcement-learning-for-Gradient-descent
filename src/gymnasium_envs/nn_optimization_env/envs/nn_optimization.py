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
from src.optimization.custom_lr_optimizer import CustomLR

class NeuralNetworkOptimization(gym.Env):
    """
    Gymnasium custom environment: 
    https://gymnasium.farama.org/introduction/create_custom_env/

    Represents neural network optimization task process where:
        Observation values:
            l1_grad_norm: Gradient L1 norm ||∇F(Xt)||₁,
            l2_grad_norm: Gradient L2 norm ||∇F(Xt)||₂,
            l2_grad_norm_log: Logarithm of gradient L2 norm log(1 + ||∇F(Xt)||₂)
            cos_sim: Cosine similarity between gradient Cos(||∇F(Xt)||₂, ||∇F(Xt-1)||₂),
            grad_delta_norm: Norm of difference between gradient ||∇F(Xt)-∇F(Xt-1)||₂,
            function_value_delta_log: Logarithm of difference between function value SIGN * log(1 + f(Xt) - f(Xt-1))
            ...
            _TODO MORE_
            ...
        Action value:
            lr: normalized learning rate in logarithmic scale in the bound [-1, 1]
    Parameters:
        render_mode: Only accept "ansi" and None type
    """

    metadata = {"render_modes": [None]}

    def __init__(self, render_mode = None, dataset_name : Optional[str] = None):
        self.render_mode = render_mode

        self.tol = 0.001
        self.max_iterations = 10000
        self.dataset_name = dataset_name
        self.eps = 1e-8

        self.observation_space = spaces.Dict({
            "l1_grad_norm" : spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),  
            "l2_grad_norm" : spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "l2_grad_norm_log" : spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),  
            "cos_sim" : spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), 
            "grad_delta_norm" : spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "function_value_delta_log" : spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)
        })

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        if dataset_name is None:
            raise ValueError("Missing dataset")
        elif dataset_name == "MNIST":
            train_dataset = datasets.MNIST(root='./data', train=True, download=True)
            test_dataset = datasets.MNIST(root='./data', train=False, download=True)

            X_train = train_dataset.data.unsqueeze(1).float() / 255.0
            y_train = train_dataset.targets.long()

            X_test = test_dataset.data.unsqueeze(1).float() / 255.0
            y_test = test_dataset.targets.long()
        else:
            raise ValueError("Incorrect dataset name.")

        self._train_ds = TensorDataset(X_train, y_train)
        self._test_ds = TensorDataset(X_test, y_test)

        test_loader = DataLoader(self._test_ds, batch_size=256, shuffle=True)

        self._test_loader = test_loader

    def reset(self, seed: Optional[int] = None, options = None):
        super().reset(seed=seed)

        g = torch.Generator()

        if seed is not None:
            g.manual_seed(seed)

        train_loader = DataLoader(self._train_ds, batch_size=256, shuffle=True, generator=g)
        self._train_loader = itertools.cycle(train_loader)

        self._iteration = 0

        # Initializing objects for neural network optimization
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.dataset_name == "MNIST":
            self._model = LeNet(seed=seed).to(self._device)
            self._criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("Incorrect dataset name.")

        self._optimizer = CustomLR(self._model.parameters())
        self._optimizer.zero_grad()

        self._prev_loss = self._compute_gradients()
        self._curr_loss = self._prev_loss

        # Initializing observation values

        observation = self._optimizer.get_obs(loss=self._curr_loss)
        info = self._optimizer.get_info(self._iteration)

        return observation, info

    def step(self, action : np.ndarray):
        self._iteration += 1

        lr = 10**(action[0] * 3 - 4)

        self._optimizer.step(lr=lr)
        self._curr_loss = self._compute_gradients()

        observation = self._optimizer.get_obs(loss=self._curr_loss)
        info = self._optimizer.get_info(self._iteration)

        reward = float(np.log2(self._prev_loss + self.eps) - np.log2(self._curr_loss + self.eps))
        reward = np.clip(reward, -5.0, 5.0)

        diverged = self._optimizer.is_diverged()
        truncated = bool(self._iteration >= self.max_iterations)

        if diverged:
            reward += -100.0
            terminated = True
            info["status"] = "diverged"
        else:
            terminated = False

        self._prev_loss = self._curr_loss
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        pass

    def close(self):
        pass
    
    def _compute_gradients(self):
        inputs, labels = next(self._train_loader)
        inputs, labels = inputs.to(self._device), labels.to(self._device)

        self._optimizer.zero_grad()

        outputs = self._model(inputs)
        loss = self._criterion(outputs, labels)
        loss.backward()

        loss_value = loss.item()

        return loss_value