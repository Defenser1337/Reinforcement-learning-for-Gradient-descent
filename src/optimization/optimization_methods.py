import numpy as np
from typing import NamedTuple, Optional, List
from src.optimization.optimization_functions.function import Function
from src.utils.prng import get_rng

class OptimizeResult(NamedTuple):
    iteration_count: int
    x_start : np.ndarray
    x_best : np.ndarray
    function_value: float
    grad_norm : float
    grad_delta_norm : float  
    status: int # -1 - max iteration reached, 0 - gradient exploded, 1 - gradient converged

def gradient_descent_optimizer(function : Function,
                               x0 : Optional[np.ndarray] = None, 
                               lr : float = 0.001, 
                               tol : float = 0.001, 
                               max_iteration_count : int = 10000,
                               random_state : Optional[int] = None,
                               verbose : int = 0,
                               opt_info : Optional[List] = None) -> OptimizeResult:
    """
    Minimizes a convex function using the Gradient Descent algorithm:
        x_{k+1} = x_k - lr * ∇f(x_k)

    Parameters:
        function (ConvexFunction): convex function to minimize
        x0 (np.ndarray, optional): initial point; mutually exclusive with random_state
        lr (float): learning rate (step size)
        tol (float): convergence tolerance for the gradient norm
        max_iteration_count (int): maximum number of iterations
        random_state (int, optional): random seed for initial point generation; mutually exclusive with x0
        verbose (int): verbosity level — 0: silent, 1: every 100 iterations, 2: every iteration
        opt_info (List, optional): empty list to collect per-iteration optimization data

    Returns:
        OptimizeResult: named tuple containing the best iteration, starting point,
        best point, function value, gradient norm, gradient delta norm and status code
            -1 : maximum iterations reached
            0 : gradient exploded
            1 : gradient converged
    """
    in_features = function.in_features
    max_absolute_value = function.max_absolute_value

    if x0 is None and random_state is not None:
        rng = get_rng(seed=random_state, location_name="gd_optimizer_function")
        x0 = max_absolute_value * rng.uniform(-1, 1, size = in_features)
        prev_x = x0.copy()
    elif x0 is not None and random_state is None:
        if x0.ndim != 1 or x0.shape[0] != in_features:
            raise ValueError("Vector dimension is incorrect")
        prev_x = x0.copy()
    else:
        raise ValueError("Arguments are incompatible")
    
    if (opt_info != None) and (len(opt_info) != 0):
        raise ValueError("List opt_info should be empty")

    prev_grad = function.get_gradient(prev_x)
    prev_norm = np.linalg.norm(prev_grad)

    best_iteration_count = 0
    best_x = prev_x
    best_function_value = function(prev_x)
    best_norm = prev_norm
    best_delta_norm = 0.0

    x = prev_x
    iteration_count = -1
    status = -1 # Initial state: max iteration reached

    # Storing optimization data
    if opt_info != None:
        info = {
            "iteration" : best_iteration_count,
            "x" : best_x,
            "function_value" : best_function_value,
            "grad_norm" : best_norm,
            "grad_delta_norm" : best_delta_norm
        }
        opt_info.append(info)

    if verbose != 0:
        print("Gradient Descent Optimization")

    for i in range(1, max_iteration_count + 1):
        x = prev_x - lr * prev_grad
        grad = function.get_gradient(x)
        function_value = function(x)
        norm = np.linalg.norm(grad)
        delta_norm = np.linalg.norm(grad - prev_grad)

        # Storing optimization data
        if opt_info != None:
            info = {
                "iteration" : i,
                "x" : x,
                "function_value" : function_value,
                "grad_norm" : norm,
                "grad_delta_norm" : delta_norm
            }
            opt_info.append(info)

        if norm/prev_norm > 1e6:
            iteration_count = i
            status = 0
            break

        if norm < tol:
            iteration_count = i 
            status = 1
            break

        if (verbose == 1) and (i % 100 == 0):
            print(f"Iteration #{i}:")
            print(f"Function value : {function_value:.4f} | Gradient norm : {norm:.4f} | Gradient delta norm : {delta_norm:.4f}")
        elif (verbose == 2):
            print(f"Iteration #{i}:")
            print(f"Function value : {function_value:.4f} | Gradient norm : {norm:.4f} | Gradient delta norm : {delta_norm:.4f}") 
            print(f"X : {x}")

        if function_value < best_function_value:
            best_iteration_count = i
            best_x = x
            best_function_value = function_value
            best_norm = norm
            best_delta_norm = delta_norm

        prev_x = x
        prev_grad = grad     
        prev_norm = norm 

    if verbose != 0:     
        if status == -1:
            print(f"Maximum iterations reached due Gradient Descent optimization.")
        elif status == 1:
            print(f"Gradient Descent converged on {iteration_count}th iteration.")
        else:
            print(f"Gradient exploded on {iteration_count}th iteration.")

    return OptimizeResult(
        iteration_count=best_iteration_count,
        x_start=x0,
        x_best=best_x,
        function_value=best_function_value,
        grad_norm=best_norm,
        grad_delta_norm=best_delta_norm,
        status=status
    )

def adam_optimizer(function : Function,
                    x0 : Optional[np.ndarray] = None, 
                    lr : float = 0.001, 
                    tol : float = 0.001, 
                    max_iteration_count : int = 10000,
                    beta1 : float = 0.99,
                    beta2 : float = 0.999,
                    eps : float = 1e-8,
                    random_state : Optional[int] = None,
                    verbose : int = 0,
                    opt_info : Optional[List] = None) -> OptimizeResult:
    """
    Minimizes a convex function using the Adam (Adaptive Moment Estimation) algorithm:
        m_k = β1 * m_{k-1} + (1 - β1) * ∇f(x_k)
        v_k = β2 * v_{k-1} + (1 - β2) * ∇f(x_k)²
        x_{k+1} = x_k - lr * m̂_k / (√v̂_k + ε)

    Where m̂_k and v̂_k are bias-corrected estimates of the first and second moments.

    Parameters:
        function (ConvexFunction): convex function to minimize
        x0 (np.ndarray, optional): initial point; mutually exclusive with random_state
        lr (float): learning rate (step size)
        tol (float): convergence tolerance for the gradient delta norm
        max_iteration_count (int): maximum number of iterations
        beta1 (float): exponential decay rate for the first moment estimate
        beta2 (float): exponential decay rate for the second moment estimate
        eps (float): small constant for numerical stability
        random_state (int, optional): random seed for initial point generation; mutually exclusive with x0
        verbose (int): verbosity level — 0: silent, 1: every 100 iterations, 2: every iteration
        opt_info (List, optional): empty list to collect per-iteration optimization data

    Returns:
        OptimizeResult: named tuple containing the best iteration, starting point,
        best point, function value, gradient norm, gradient delta norm and status code
            -1 : maximum iterations reached
            0 : gradient exploded
            1 : gradient converged
    """
    in_features = function.in_features
    max_absolute_value = function.max_absolute_value

    if x0 is None and random_state is not None:
        rng = get_rng(seed=random_state, location_name="adam_optimizer_function")
        x0 = max_absolute_value * rng.uniform(-1, 1, size = in_features)
        prev_x = x0.copy()
    elif x0 is not None and random_state is None:
        if x0.ndim != 1 or x0.shape[0] != in_features:
            raise ValueError("Vector dimension is incorrect")
        prev_x = x0.copy()
    else:
        raise ValueError("Arguments are incompatible")
    
    if (opt_info != None) and (len(opt_info) != 0):
        raise ValueError("List opt_info should be empty")
    
    prev_grad = function.get_gradient(x=prev_x)
    prev_norm = np.linalg.norm(prev_grad)
    prev_mean = np.zeros(in_features)
    prev_var = np.zeros(in_features)

    best_x = prev_x
    best_function_value = function(prev_x)
    best_iteration_count = 0
    best_norm = prev_norm
    best_delta_norm = 0.0

    x = prev_x
    iteration_count = -1
    status = -1 # Initial state: max iteration reached

    # Storing optimization data
    if opt_info != None:
        info = {
            "iteration" : best_iteration_count,
            "x" : best_x,
            "function_value" : best_function_value,
            "grad_norm" : best_norm,
            "grad_delta_norm" : best_delta_norm
        }
        opt_info.append(info)

    if verbose != 0:
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

        # Storing optimization data
        if opt_info != None:
            info = {
                "iteration" : i,
                "x" : x,
                "function_value" : function_value,
                "grad_norm" : norm,
                "grad_delta_norm" : delta_norm
            }
            opt_info.append(info)

        if norm/prev_norm > 1e6:
            iteration_count = i 
            status = 0
            break

        if norm < tol:
            iteration_count = i 
            status = 1
            break

        if (verbose == 1) and (i % 100 == 0):
            print(f"Iteration #{i}:")
            print(f"Function value : {function_value:.4f} | Gradient norm : {norm:.4f} | Gradient delta norm : {delta_norm:.4f}")
        elif (verbose == 2):
            print(f"Iteration #{i}:")
            print(f"Function value : {function_value:.4f} | Gradient norm : {norm:.4f} | Gradient delta norm : {delta_norm:.4f}") 
            print(f"X : {x}")

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

    if verbose != 0:   
        if status == -1:
            print(f"Maximum iterations reached due ADAM optimization.")
        elif status == 1:
            print(f"ADAM converged on {iteration_count}th iteration")
        else:
            print(f"Gradient exploded on {iteration_count}th iteration!")

    return OptimizeResult(
        iteration_count=best_iteration_count,
        x_start=x0,
        x_best=best_x,
        function_value=best_function_value,
        grad_norm=best_norm,
        grad_delta_norm=best_delta_norm,
        status=status
    )

def show_result(result : OptimizeResult):
    print(100 * "-")
    print(f"Best result on {result.iteration_count}th iteration")
    print(f"Function value : {result.function_value:.4f} | Gradient norm : {result.grad_norm:.4f} | Gradient delta norm : {result.grad_delta_norm:.4f}")
    print(f"X start : {result.x_start} | X best : {result.x_best}")
    print(100 * "-")