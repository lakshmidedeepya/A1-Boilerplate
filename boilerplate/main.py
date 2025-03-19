import os
from typing import Callable, Literal

from prettytable import PrettyTable
import numpy as np
import numpy.typing as npt

from algos import newton_method, steepest_descent
from functions import (
    trid_function,
    trid_function_derivative,
    trid_function_hessian,
    three_hump_camel_function,
    three_hump_camel_function_derivative,
    three_hump_camel_function_hessian,
    rosenbrock_function,
    rosenbrock_function_derivative,
    rosenbrock_function_hessian,
    styblinski_tang_function,
    styblinski_tang_function_derivative,
    styblinski_tang_function_hessian,
    func_1,
    func_1_derivative,
    func_1_hessian,
)

test_cases = [
    [
        trid_function,
        trid_function_derivative,
        trid_function_hessian,
        np.asarray([-2.0, -2]),
    ],
    [
        trid_function,
        trid_function_derivative,
        trid_function_hessian,
        np.asarray([-2.0, -2]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        three_hump_camel_function_hessian,
        np.asarray([-2.0, 1]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        three_hump_camel_function_hessian,
        np.asarray([2.0, -1]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        three_hump_camel_function_hessian,
        np.asarray([-2.0, -1]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        three_hump_camel_function_hessian,
        np.asarray([2.0, 1]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        rosenbrock_function_hessian,
        np.asarray([2.0, 2, 2, -2]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        rosenbrock_function_hessian,
        np.asarray([2.0, -2, -2, 2]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        rosenbrock_function_hessian,
        np.asarray([-2.0, 2, 2, 2]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        rosenbrock_function_hessian,
        np.asarray([3.0, 3, 3, 3]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        styblinski_tang_function_hessian,
        np.asarray([0.0, 0, 0, 0]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        styblinski_tang_function_hessian,
        np.asarray([3.0, 3, 3, 3]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        styblinski_tang_function_hessian,
        np.asarray([-3.0, -3, -3, -3]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        styblinski_tang_function_hessian,
        np.asarray([3.0, -3, 3, -3]),
    ],
    [
        func_1,
        func_1_derivative,
        func_1_hessian,
        np.asarray([3.0, 3]),
    ],
    [
        func_1,
        func_1_derivative,
        func_1_hessian,
        np.asarray([-0.5, 0.5]),
    ],
    [
        func_1,
        func_1_derivative,
        func_1_hessian,
        np.asarray([-3.5, 0.5]),
    ],
]


def main():
    if not os.path.isdir("plots"):
        os.mkdir("plots")

    table = PrettyTable()
    table.field_names = [
        "Test case",
        "Backtracking-Armijo",
        "Backtracking-Goldstein",
        "Bisection",
        "Pure",
        "Damped",
        "Levenberg-Marquardt",
        "Combined",
    ]
    dividers = [1, 5, 9, 13]
    for test_case_num, test_case in enumerate(test_cases):
        row = [
            test_case_num,
        ]
        for algo in table.field_names:
            if algo == "Test case":
                continue
            if algo == "Backtracking-Armijo" or algo == "Backtracking-Goldstein" or algo == "Bisection":
                try:
                    ans = steepest_descent(
                        test_case[0], test_case[1], test_case[3], algo
                    )
                    if type(ans) != np.ndarray:
                        print(
                            f"Wrong type of value returned in steepest descent with backtracking with {algo} condition"
                        )
                        print(
                            f"Test function was {test_case[0].__name__} with {test_case[3]} as starting point"
                        )
                        row += [None]
                    else:
                        row += [np.round(ans, 3)]
                except Exception as e:
                    print(
                        f"Error in steepest descent with backtracking with {algo} condition"
                    )
                    print(
                        f"Test function was {test_case[0].__name__} with {test_case[3]} as starting point"
                    )
                    print(e)
                    row += [None]
            elif (
                algo == "Pure"
                or algo == "Damped"
                or algo == "Levenberg-Marquardt"
                or algo == "Combined"
            ):
                try:
                    ans = newton_method(
                        test_case[0], test_case[1], test_case[2], test_case[3], algo
                    )
                    if type(ans) != np.ndarray:
                        print(f"Wrong type of value returned in {algo} Newton's method")
                        print(
                            f"Test function was {test_case[0].__name__} with {test_case[3]} as starting point"
                        )
                        row += [None]
                    else:
                        row += [np.round(ans, 3)]
                except Exception as e:
                    print(f"Error in {algo} Newton's method")
                    print(
                        f"Test function was {test_case[0].__name__} with {test_case[3]} as starting point"
                    )
                    print(e)
                    row += [None]
        table.add_row(row, divider=test_case_num in dividers)
    print(table) 


if __name__ == "__main__":
    main()
