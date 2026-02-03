from convex_function import ConvexFunction
import numpy as np

if __name__ == "__main__":
    print("Class testing on basic Gradient descent")

    in_features = 1000
    max_absolute_value = 1.0
    iteration_count = 10000
    alpha = 0.0003
    gamma = 0.001

    function1 = ConvexFunction(in_features, random_seed=42, max_absolute_value=max_absolute_value)

    x = max_absolute_value * (2 * np.random.rand(in_features) - 1)

    prev_gradient = function1.get_gradient(x)

    for i in range(1, iteration_count):
        x = x - alpha * prev_gradient

        value = function1(x)
        gradient = function1.get_gradient(x)
        gradient_delta_norm = np.linalg.norm(prev_gradient - gradient)

        if i % 100 == 0:
            print(f"Iteration #{i}:")
            print(f"Function value : {value} | Gradient delta norm : {gradient_delta_norm}") 

        if gradient_delta_norm < gamma:
            print("------------------------------------")
            print(f"Gradient descent converged on {i}th iteration")
            print("Result")
            print(f"Function value : {value} | Gradient delta norm : {gradient_delta_norm}")
            break

        prev_gradient = gradient

   

        
    