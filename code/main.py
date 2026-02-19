from convex_function import ConvexFunction
from optimization_methods import gradient_descent_optimizer, adam_optimizer

if __name__ == "__main__":
    in_features = 10
    max_absolute_value = 1.0

    function1 = ConvexFunction(in_features, random_state=42, max_absolute_value=max_absolute_value)

    gd_result = gradient_descent_optimizer(function1, lr = 0.0009, random_state=1337)
    adam_result = adam_optimizer(function1, lr = 2 ,random_state=1337)