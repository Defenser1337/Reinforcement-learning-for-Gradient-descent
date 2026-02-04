import numpy as np
from typing import NamedTuple
from convex_function import ConvexFunction

class OptimizeResult(NamedTuple):
    x: np.ndarray
    function_value: float  
    iteration_count: int  
    status: int # -1 - not enough iterations, 0 - gradient exploded, 1 - succes

def gradient_descent(function : ConvexFunction, 
                     lr : float = 0.0004, 
                     tol : float = 0.001, 
                     max_iteration_count : int = 10000,
                     random_state : int = None,
                     display_result : bool = False) -> OptimizeResult:
    print("Gradient Descent Optimization")

    rng = np.random.RandomState(random_state)
    in_features = function.in_features
    max_absolute_value = function.max_absolute_value

    prev_x = max_absolute_value * rng.uniform(-1, 1, size = in_features)
    prev_grad = function.get_gradient(x=prev_x)
    prev_norm = np.linalg.norm(prev_grad)

    x = prev_x
    function_value = function(prev_x)
    iteration_count = -1
    status = -1

    for i in range(1, max_iteration_count + 1):
        x = prev_x - lr * prev_grad
        grad = function.get_gradient(x)

        function_value = function(x)
        norm = np.linalg.norm(grad)
        delta_norm = np.linalg.norm(grad - prev_grad)

        if norm/prev_norm > 1e4:
            status = 0
            print(f"Gradient exploded on {i}th iteration!")
            break

        if delta_norm < tol:
            iteration_count = i 
            status = 1
            break

        if i % 100 == 0 and display_result == True:
            print(f"Iteration #{i}:")
            print(f"Function value : {function_value:.4f} | Gradient norm : {norm:.4f} | Gradient delta norm : {delta_norm:.4f}") 

        prev_x = x
        prev_norm = norm
        prev_grad = grad         


    if status == -1:
        print(f"Maximum iterations reached.")
    elif status == 1:
        print(100 * "-")
        print(f"Gradient Descent converged on {i}th iteration")
        print("Result")
        print(f"Function value : {function_value:.4f} | Gradient norm : {norm:.4f} | Gradient delta norm : {delta_norm:.4f}")

    return OptimizeResult(
        x=x,
        function_value=function_value,
        iteration_count=iteration_count,
        status=status
    )

     


