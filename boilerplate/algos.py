from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os

def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking-Armijo", "Backtracking-Goldstein", "Bisection"],
) -> npt.NDArray[np.float64]:
    x = inital_point.copy()
    f_vals = [f(x)]
    gradient = d_f(x)
    grad_norms = [np.linalg.norm(gradient)]
    x_history = [x.copy()]
    max_iter = 10000
    epsilon = 1e-6
    k = 0
    
    #grad_norms[-1] last element of the list
    while k < max_iter and grad_norms[-1] > epsilon:
        
        d = -gradient
    
        if condition == "Backtracking-Armijo":
            c1 = 0.001
            alpha = 10.0
            rho = 0.75
            for _ in range(100):
                if f(x + alpha * d) <= f(x) + c1 * alpha * gradient.dot(d):
                    break
                alpha *=rho 

            

        elif condition == "Backtracking-Goldstein":
            c1 = 0.001
            alpha = 10.0
            rho = 0.75
            for _ in range(100):
                f_new = f(x + alpha * d)
                upper = f(x) + c1 * alpha * gradient.dot(d)
                lower = f(x) + (1 - c1) * alpha * gradient.dot(d)
                if f_new <= upper and f_new >= lower:
                    break
                alpha *= rho
          

        elif condition == "Bisection":
            c1, c2 = 0.001, 0.1
            alpha, beta = 0.0, 1e6
            t = 1.0
            bisection_iter = 0
            max_bisection_iter = 500  # Prevent infinite loops
        
            while bisection_iter < max_bisection_iter:
                f_new = f(x + t * d)
                grad_new = d_f(x + t * d)
                # Check Armijo condition (sufficient decrease)
                if f_new > f(x) + c1 * t * gradient.dot(d):
                    beta = t
                # Check curvature condition (strong Wolfe condition /sufficient progress)
                elif grad_new.dot(d) < c2 * gradient.dot(d):
                    alpha = t
                else:
                    break
                t = (alpha + beta) / 2
                bisection_iter += 1
            alpha = t #since new x should be updated with t value
            
        x_new = x + alpha * d
        x = x_new
        gradient = d_f(x)
        f_vals.append(f(x))
        grad_norms.append(np.linalg.norm(gradient))
        x_history.append(x.copy())
        # Divergence check
        if np.linalg.norm(x) > 1e10:
            break
        k += 1

    plot_results(f, inital_point, condition, f_vals, grad_norms, x_history)

    

    return x


def newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Pure", "Damped", "Levenberg-Marquardt", "Combined"],
) -> npt.NDArray[np.float64]:
    x = inital_point.copy()
    f_vals = [f(x)]  
    x_history = [x.copy()]
    max_iter = 10000
    epsilon = 1e-6
    k = 0
    gradient = d_f(x)
    grad_norms = [np.linalg.norm(gradient)]

    while k < max_iter and grad_norms[-1] > epsilon:
        try:
            H = d2_f(x)
            
            # Add numerical stability to Hessian modification
            H = modify_hessian(x, H) if condition in ["Levenberg-Marquardt", "Combined"] else H
            # Prevent overflow by clipping extreme values
            H = np.clip(H, -1e10, 1e10)

            # Ensure Hessian is well-conditioned
            if np.linalg.cond(H) > 1e12:  
                H += np.eye(len(x)) * 1e-5  # Regularization to prevent singularity
            
            # Newton direction calculation
            try:
                d = np.linalg.solve(H, -gradient)
            except np.linalg.LinAlgError:
                d = -gradient   # Fallback to gradient direction 
            step = d
            if condition in ["Combined" ,"Damped"]:
                t, alpha_param , rho = 1.0, 0.001, 0.75
                for _ in range(100):
                    if f(x + t * d) <= f(x) + alpha_param * t * gradient.dot(d):
                        break
                    t *= rho
                step = t * d
            # Update position and gradients
            x+= step
            gradient = d_f(x)

            # Store results
            f_vals.append(f(x))
            grad_norms.append(np.linalg.norm(gradient))
            x_history.append(x.copy())
            # Divergence safeguard
            if np.linalg.norm(x) > 1e10:
                break
            k += 1

        except np.linalg.LinAlgError:
            break

    plot_results(f, inital_point, condition, f_vals, grad_norms, x_history)
    return x
#Helper Function to modify the Hessian matrix
def modify_hessian(x: npt.NDArray[np.float64], H: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    eigvals = np.linalg.eigvals(H)
    lambda_min = np.min(eigvals)
    if lambda_min <= 0:
        mu = -lambda_min + 0.1
        return H + mu * np.eye(len(x))
    return H
def plot_results(f, inital_point, condition, f_vals, grad_norms, x_history):
    os.makedirs("plots", exist_ok=True)
    # Function value plot
    plt.figure()
    plt.plot(f_vals)
    plt.xlabel('Iterations')
    plt.ylabel('Function Value')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.png")
    plt.close()
    # Gradient norm plot
    plt.figure()
    plt.plot(grad_norms)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.png")
    plt.close()

    # contour plot for 2D cases
    if inital_point.ndim == 1 and len(inital_point) == 2:
        x_hist = np.array(x_history)
       
        #setting up the plot boundaries.
        x_min, x_max = x_hist[:, 0].min(), x_hist[:, 0].max() #x_hist[:, 0] Stores all x-coordinates.
        y_min, y_max = x_hist[:, 1].min(), x_hist[:, 1].max() #x_hist[:, 1] #Stores all y-coordinates.
        #Padding=max(10^−5 ,0.5×(max value−min value)) to Improve visualization clarity
        x_pad = max(1e-5, 0.5 * (x_max - x_min))
        y_pad = max(1e-5, 0.5 * (y_max - y_min))
        #Creating final grid of 100x100 points with padding
        #np.linspace(start, stop, num) creates evenly spaced values between start and stop
        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 100)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.array([f(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
        plt.figure()
        plt.contour(X, Y, Z, levels=50)
       
        plt.plot(x_hist[:, 0], x_hist[:, 1], 'r--o')

        plt.quiver(x_hist[:-1, 0], x_hist[:-1, 1],#Start points
                    x_hist[1:, 0] - x_hist[:-1, 0],#X-direction change
                      x_hist[1:, 1] - x_hist[:-1, 1], # Y-direction change
                   angles='xy', scale_units='xy', scale=1, color='blue', width=0.003,headwidth=8, headlength=10, headaxislength=7)
        
        
        # Highlight the start point
        #plt.scatter(x_hist[0, 0], x_hist[0, 1], color='green', s=150, marker='o', label='Start point')
        plt.scatter(x_hist[0, 0], x_hist[0, 1], s=150, 
                   edgecolors='black', facecolors='lime', 
                   marker='o', label='Start Point')
        # Highlight the end point
        #plt.scatter(x_hist[-1, 0], x_hist[-1, 1], color='red', s=150, marker='x', label='End point')
        plt.scatter(x_hist[-1, 0], x_hist[-1, 1], s=200,
                   edgecolors='black', facecolors='red', 
                   marker='X', label='Final Point')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Contour Plot: {condition}')
        plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.png")
        plt.close()

