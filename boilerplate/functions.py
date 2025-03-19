import numpy as np
import numpy.typing as npt


# ------------------------------------------------------------------------------


def trid_function(point: npt.NDArray[np.float64]) -> np.float64:
    return np.sum((point - 1) ** 2) - np.sum(point[1:] * point[:-1])


def trid_function_derivative(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    der = np.zeros_like(point)
    for i in range(point.shape[0]):
        der[i] = 2 * (point[i] - 1)
        if i != 0:
            der[i] -= point[i - 1]
        if i != point.shape[0] - 1:
            der[i] -= point[i + 1]
    return der


def trid_function_hessian(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    hes = np.zeros((point.shape[0], point.shape[0]))
    for i in range(point.shape[0]):
        hes[i][i] = 2
        if i != 0:
            hes[i][i - 1] = -1
        if i != point.shape[0] - 1:
            hes[i][i + 1] = -1

    return hes


# ------------------------------------------------------------------------------


def three_hump_camel_function(point: npt.NDArray[np.float64]) -> np.float64:
    return (
        2 * point[0] ** 2
        - 1.05 * point[0] ** 4
        + point[0] ** 6 / 6
        + point[0] * point[1]
        + point[1] ** 2
    )


def three_hump_camel_function_derivative(
    point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        [
            4 * point[0] - 4.2 * point[0] ** 3 + point[0] ** 5 + point[1],
            point[0] + 2 * point[1],
        ]
    )


def three_hump_camel_function_hessian(
    point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray([[4 - 12.6 * point[0] ** 2 + 5 * point[0] ** 4, 1], [1, 2]])


# ------------------------------------------------------------------------------


def rosenbrock_function(point: npt.NDArray[np.float64]) -> np.float64:
    return np.sum(100 * (point[1:] - point[:-1] ** 2) ** 2 + (point[:-1] - 1) ** 2)


def rosenbrock_function_derivative(
    point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    der = np.zeros_like(point)
    der[0] = 2 * (point[0] - 1) - 400 * (point[1] - point[0] ** 2) * point[0]
    der[-1] = 200 * (point[-1] - point[-2] ** 2)
    for i in range(1, point.shape[0] - 1):
        der[i] = (
            400 * point[i] ** 3
            + 202 * point[i]
            - 400 * point[i + 1] * point[i]
            - 200 * point[i - 1] ** 2
            - 2
        )
    return der


def rosenbrock_function_hessian(
    point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    hes = np.zeros((point.shape[0], point.shape[0]))
    hes[0][0] = 2 - 400 * point[1] + 1200 * point[0] ** 2
    hes[0][1] = -400 * point[0]
    hes[-1][-1] = 200
    hes[-1][-2] = -400 * point[-2]
    for i in range(1, point.shape[0] - 1):
        hes[i][i] = 1200 * point[i] ** 2 + 202 - 400 * point[i + 1]
        hes[i][i - 1] = -400 * point[i - 1]
        hes[i][i + 1] = -400 * point[i]
    return hes


# ------------------------------------------------------------------------------


def styblinski_tang_function(point: npt.NDArray[np.float64]) -> np.float64:
    return np.sum(point**4 - 16 * point**2 + 5 * point) / 2


def styblinski_tang_function_derivative(
    point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return (4 * point**3 - 32 * point + 5) / 2


def styblinski_tang_function_hessian(
    point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.diag(12 * point**2 - 32) / 2


# ------------------------------------------------------------------------------


def func_1(point: npt.NDArray[np.float64]) -> np.float64:
    return np.sum(np.sqrt(1 + point**2))


def func_1_derivative(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return point / np.sqrt(1 + point**2)


def func_1_hessian(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.diag(1 / ((1 + point**2) ** 1.5))


# ----------------------------------------------------------------------------
