from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking-Armijo", "Backtracking-Goldstein", "Bisection"],
) -> npt.NDArray[np.float64]:
    x = inital_point.copy()
    f_vals = [f(x)]
    grad_norms = [np.linalg.norm(d_f(x))]
    x_history = [x.copy()]
    max_iter = 10000
    epsilon = 1e-6
    k = 0

    #grad_norms[-1] last element of the list
    while k < max_iter and grad_norms[-1] > epsilon:
        gradient = d_f(x)
        d = -gradient

        if condition == "Backtracking-Armijo":
            c1 = 0.001
            alpha = 10.0
            beta = 0.75
            for _ in range(100):
                if f(x + alpha * d) <= f(x) + c1 * alpha * gradient.dot(d):
                    break
                alpha *= beta
            

        elif condition == "Backtracking-Goldstein":
            c1 = 0.001
            alpha = 10.0
            beta = 0.75
            for _ in range(100):
                f_new = f(x + alpha * d)
                upper = f(x) + c1 * alpha * gradient.dot(d)
                lower = f(x) + (1 - c1) * alpha * gradient.dot(d)
                if f_new <= upper and f_new >= lower:
                    break
                alpha *= beta
          

        elif condition == "Bisection":
            c1, c2 = 0.001, 0.1
            alpha, beta = 0.0, 1e6
            t = 1.0
            
        
            for _ in range(100):
                f_new = f(x + t * d)
                grad_new = d_f(x + t * d)
                # Check Armijo condition (sufficient decrease)
                if f_new > f(x) + c1 * t * gradient.dot(d):
                    beta = t
                # Check curvature condition (strong Wolfe condition)
                elif grad_new.dot(d) < c2 * gradient.dot(d):
                    alpha = t
                else:
                    break
                t = (alpha + beta) / 2
            alpha = t #since new x should be updated with t value

        x_new = x + alpha * d
        x = x_new
        f_vals.append(f(x))
        grad_norms.append(np.linalg.norm(d_f(x)))
        x_history.append(x.copy())
        k += 1

    plt.figure()
    plt.plot(f_vals)
    plt.xlabel('Iterations')
    plt.ylabel('Function Value')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.png")
    plt.close()

    plt.figure()
    plt.plot(grad_norms)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.png")
    plt.close()

    #1D NumPy array of length 2 .Handling 2D case only
    if inital_point.ndim == 1 and len(inital_point) == 2:
        x_hist = np.array(x_history)
        #x_hist[:, 0] Stores all x-coordinates.
        #x_hist[:, 1] #Stores all y-coordinates.
        #setting up the plot boundaries.
        x_min, x_max = x_hist[:, 0].min(), x_hist[:, 0].max()
        y_min, y_max = x_hist[:, 1].min(), x_hist[:, 1].max()
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
        #red ('r.-'):
        #Red dots (.) → Each visited point.
        #Red line (-) → Connects the points.
        plt.plot(x_hist[:, 0], x_hist[:, 1], 'r.-')

        plt.quiver(x_hist[:-1, 0], x_hist[:-1, 1],#Start points
                    x_hist[1:, 0] - x_hist[:-1, 0],#X-direction change
                      x_hist[1:, 1] - x_hist[:-1, 1], # Y-direction change
                   angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Contour Plot: {condition}')
        plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.png")
        plt.close()

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
    grad_norms = [np.linalg.norm(d_f(x))]
    x_history = [x.copy()]
    max_iter = 10000
    epsilon = 1e-6
    k = 0
    gradient = d_f(x)

    while k < max_iter and grad_norms[-1] > epsilon:
        
        H = d2_f(x)
        step = np.zeros_like(x)

        try:
            if condition in ["Levenberg-Marquardt", "Combined","Pure"]:
                if condition in ["Levenberg-Marquardt", "Combined"]:
                    H = modify_hessian(H)
            d = np.linalg.solve(H, -gradient)
            step = d
            if condition in ["Combined" ,"Damped"]:
                t, alpha1, beta = 1.0, 0.001, 0.75
                for _ in range(100):
                    if f(x + t * d) <= f(x) + alpha1 * t * gradient.dot(d):
                        break
                    t *= beta
                step = t * d
            x_new = x + step
            x = x_new
            f_vals.append(f(x))
            grad_norms.append(np.linalg.norm(d_f(x)))
            x_history.append(x.copy())
            k += 1

        except np.linalg.LinAlgError:
            break

    plt.figure()
    plt.plot(f_vals)
    plt.xlabel('Iterations')
    plt.ylabel('Function Value')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.png")
    plt.close()

    plt.figure()
    plt.plot(grad_norms)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.png")
    plt.close()

    if inital_point.ndim == 1 and len(inital_point) == 2:
        x_hist = np.array(x_history)
        x_min, x_max = x_hist[:, 0].min(), x_hist[:, 0].max()
        y_min, y_max = x_hist[:, 1].min(), x_hist[:, 1].max()
        x_pad = max(1e-5, 0.5 * (x_max - x_min))
        y_pad = max(1e-5, 0.5 * (y_max - y_min))
        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 100)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.array([f(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
        plt.figure()
        plt.contour(X, Y, Z, levels=50)
        plt.plot(x_hist[:, 0], x_hist[:, 1], 'r.-')
        plt.quiver(x_hist[:-1, 0], x_hist[:-1, 1], x_hist[1:, 0] - x_hist[:-1, 0], x_hist[1:, 1] - x_hist[:-1, 1],
                   angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)

        """ # Highlight the minimum point (1,1) for Rosenbrock function
        plt.scatter(1, 1, color='green', s=200, marker='*', label='True minimum (1,1)')
        
        # Highlight the start point
        plt.scatter(x_hist[0, 0], x_hist[0, 1], color='blue', s=150, marker='o', label='Start point')
        
        # Highlight the end point
        plt.scatter(x_hist[-1, 0], x_hist[-1, 1], color='red', s=150, marker='x', label='End point')
        
        plt.xlabel('x1', fontsize=12)
        plt.ylabel('x2', fontsize=12)
        plt.title(f'Contour Plot: {condition}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.colorbar(cp, label='Function value')
        
        # Add iteration counts at selected points
        n_points = len(x_hist)
        if n_points > 10:
            step = n_points // 5
            for i in range(0, n_points, step):
                if i > 0 and i < n_points - 1:  # Skip start and end points as they're already labeled
                    plt.annotate(f"{i}", (x_hist[i, 0], x_hist[i, 1]), 
                                xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.savefig(f"plots/{f.__name__}_{np.array2string(initial_point)}_{condition}_cont.png", dpi=300, bbox_inches='tight')
        plt.close() """
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Contour Plot: {condition}')
        plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.png")
        plt.close() 

    return x
#Helper Function to modify the Hessian matrix
def modify_hessian(H):
    eigvals = np.linalg.eigvalsh(H)
    lambda_min = np.min(eigvals)
    if lambda_min <= 0:
        mu = -lambda_min + 0.1
        return H + mu * np.eye(H.shape[0])
    return H