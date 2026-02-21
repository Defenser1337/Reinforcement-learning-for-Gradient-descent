import numpy as np
from typing import NamedTuple
from convex_function import ConvexFunction

class OptimizeResult(NamedTuple):
    x: np.ndarray
    function_value: float  
    iteration_count: int  
    status: int # -1 - max iteration reached, 0 - gradient exploded, 1 - gradient converged

def gradient_descent_optimizer(function : ConvexFunction, 
                            lr : float = 0.001, 
                            tol : float = 0.001, 
                            max_iteration_count : int = 10000,
                            random_state : int = None,
                            display_result : bool = False) -> OptimizeResult:
    rng = np.random.RandomState(random_state)

    in_features = function.in_features
    max_absolute_value = function.max_absolute_value

    prev_x = max_absolute_value * rng.uniform(-1, 1, size = in_features)
    prev_grad = function.get_gradient(prev_x)
    prev_norm = np.linalg.norm(prev_grad)

    best_x = prev_x
    best_function_value = function(prev_x)
    best_iteration_count = 0
    best_norm = prev_norm
    best_delta_norm = float("inf")

    x = prev_x
    iteration_count = -1
    # Initial state: max iteration reached
    status = -1

    print("Gradient Descent Optimization")

    for i in range(1, max_iteration_count + 1):
        x = prev_x - lr * prev_grad
        grad = function.get_gradient(x)

        function_value = function(x)
        norm = np.linalg.norm(grad)
        delta_norm = np.linalg.norm(grad - prev_grad)

        if norm/prev_norm > 1e6:
            iteration_count = i
            status = 0
            break

        if norm < tol:
            iteration_count = i 
            status = 1
            break

        if display_result == True and i % 100 == 0:
            print(f"Iteration #{i}:")
            print(f"Function value : {function_value:.4f} | Gradient norm : {norm:.4f} | Gradient delta norm : {delta_norm:.4f}") 

        if function_value < best_function_value:
            best_x = x
            best_function_value = function_value
            best_iteration_count = i
            best_norm = norm
            best_delta_norm = delta_norm

        prev_x = x
        prev_grad = grad     
        prev_norm = norm
            
    print(100 * "-")        

    if status == -1:
        print(f"Maximum iterations reached due Gradient Descent optimization.")
    elif status == 1:
        print(f"Gradient Descent converged on {iteration_count}th iteration.")
    else:
        print(f"Gradient exploded on {iteration_count}th iteration.")

    print(f"Best result on {best_iteration_count}th iteration")
    print(f"Function value : {best_function_value:.4f} | Gradient norm : {best_norm:.4f} | Gradient delta norm : {best_delta_norm:.4f}")
    print(100 * "-")

    return OptimizeResult(
        x=best_x,
        function_value=best_function_value,
        iteration_count=best_iteration_count,
        status=status
    )

def adam_optimizer(function : ConvexFunction, 
                    lr : float = 0.001, 
                    tol : float = 0.001, 
                    max_iteration_count : int = 10000,
                    beta1 : float = 0.99,
                    beta2 : float = 0.999,
                    eps : float = 1e-8,
                    random_state : int = None,
                    display_result : bool = False) -> OptimizeResult:
    rng = np.random.RandomState(random_state)
    in_features = function.in_features
    max_absolute_value = function.max_absolute_value

    prev_x = max_absolute_value * rng.uniform(-1, 1, size = in_features)
    prev_grad = function.get_gradient(x=prev_x)
    prev_norm = np.linalg.norm(prev_grad)
    prev_mean = np.zeros(in_features)
    prev_var = np.zeros(in_features)

    best_x = prev_x
    best_function_value = function(prev_x)
    best_iteration_count = 0
    best_norm = prev_norm
    best_delta_norm = float("inf")

    x = prev_x
    iteration_count = -1
    # Initial state: max iteration reached
    status = -1 

    print("Adam Optimization")

    for i in range(1, max_iteration_count + 1):
        mean = beta1 * prev_mean + (1 - beta1) * prev_grad
        var = beta2 * prev_var + (1 - beta2) * (prev_grad ** 2)

        mean_corr = mean / (1 - beta1 ** i)
        var_corr = var / (1 - beta2 ** i)

        x = prev_x - lr * mean_corr / (np.sqrt(var_corr) + eps)
        grad = function.get_gradient(x)

        function_value = function(x)
        norm = np.linalg.norm(grad)
        delta_norm = np.linalg.norm(grad - prev_grad)

        if norm/prev_norm > 1e6:
            iteration_count = i 
            status = 0
            break

        if delta_norm < tol:
            iteration_count = i 
            status = 1
            break

        if display_result == True and i % 100 == 0:
            print(f"Iteration #{i}:")
            print(f"Function value : {function_value:.4f} | Gradient norm : {norm:.4f} | Gradient delta norm : {delta_norm:.4f}") 

        if function_value < best_function_value:
            best_x = x
            best_function_value = function_value
            best_iteration_count = i
            best_norm = norm
            best_delta_norm = delta_norm
            
        prev_x = x
        prev_norm = norm
        prev_grad = grad
        prev_mean = mean
        prev_var = var         

    print(100 * "-")

    if status == -1:
        print(f"Maximum iterations reached due ADAM optimization.")
    elif status == 1:
        print(f"ADAM converged on {iteration_count}th iteration")
    else:
        print(f"Gradient exploded on {iteration_count}th iteration!")

    print(f"Best result on {best_iteration_count}th iteration")
    print(f"Function value : {best_function_value:.4f} | Gradient norm : {best_norm:.4f} | Gradient delta norm : {best_delta_norm:.4f}")
    print(100 * "-")

    return OptimizeResult(
        x=best_x,
        function_value=best_function_value,
        iteration_count=best_iteration_count,
        status=status
    )