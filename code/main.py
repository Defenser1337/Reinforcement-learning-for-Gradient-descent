from convex_function import ConvexFunction
import numpy as np
from optimization_methods import gradient_descent

if __name__ == "__main__":
    in_features = 1000
    max_absolute_value = 1.0
    iteration_count = 10000
    alpha = 0.0003
    gamma = 0.001

    function1 = ConvexFunction(in_features, random_state=42, max_absolute_value=max_absolute_value)

    gd_result = gradient_descent(function1, random_state=42)