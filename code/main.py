from convex_function import ConvexFunction
import numpy as np
from optimization_methods import gradient_descent_optimizer, adam_optimizer

if __name__ == "__main__":
    in_features = 1000
    max_absolute_value = 10.0

    function1 = ConvexFunction(in_features, random_state=42, max_absolute_value=max_absolute_value)

    gd_result = gradient_descent_optimizer(function1, random_state=42)
    adam_result = adam_optimizer(function1, lr = 2 ,random_state=42)