import math
import numpy as np


def get_data():
    n = 4
    m = 1
    r = 1
    s = 2
    theta_true = np.array([4.6, 0.787])
    return n, m, r, s, theta_true


def get_f(t, x):
    T = 0.1
    e = math.exp(-x[2]*T)
    return np.array([[x[0] + (1 - e) * x[1] / x[2]],
                     [e * x[1]],
                     [x[2]],
                     [x[3]]])


def get_dfdx(t, x):
    T = 0.1
    e = math.exp(-x[2]*T)
    temp = 1. / x[2]
    return np.array([[1, temp * (1 - e), x[1] * (temp**2) * (e * (1 + T * x[2]) - 1), 0],
                     [0, e, -x[1] * T * e, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def get_psi(t, x):
    T = 0.1
    e = math.exp(-x[2] * T)
    temp = 1./x[2]
    return np.array([[x[3] * temp * (T - temp + temp * e)],
                     [x[3] * temp * (1 - e)],
                     [0],
                     [0]])


def get_u(t):
    return np.array([[75]])


def get_P0():
    # return np.array([[0.1, 0.1, 1, 0.1],
    #                  [0.1, 0, 0, 0],
    #                  [1, 0, 100, 0],
    #                  [0.1, 0, 0, 1]])
    return np.array([[0.1, 0.1, 1, 0.1],
                     [0.1, 0, 0, 0],
                     [1, 0, 110, 0],
                     [0.1, 0, 0, 1]])


def get_h(t, x):
    return np.array([x[0]])


def get_dhdx(t, x):
    return np.array([[1, 0, 0, 0]])


def get_R(t, x):
    return np.array([[0.1]])


def get_G(t):
    return np.array([[0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1]])


def get_Q(t):
    return np.array([[1e-4, 0],
                     [0, 1e-4]])


def get_x0(theta):
    return np.array([0, 0, theta[0], theta[1]])


def get_t0():
    """ Функция получения начального момента времени

        Returns
        -------
            float
                начальный момент времени
    """
    return 0.


def get_t_step():
    """ Функция получения шага времени

        Returns
        -------
            float
                шаг времени
    """
    return 1.


def get_t_next(t_prev):
    """ Функция получения следубщего момента времени

        Parameters
        ----------
            t_prev: float
                предыдущий момент времени

        Returns
        -------
            float
                следующий момент времени
    """
    step = get_t_step()
    return t_prev + step

