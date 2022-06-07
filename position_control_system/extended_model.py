import math
import numpy as np


def get_data():
    """ Функция получения общей информации об
        исследуемой расширенной модели

        Returns
        -------
            n, m, r: int, int, int
                размер вектора состояний,
                размер вектора измерений,
                размер вектора управления.
    """
    n = 4
    m = 1
    r = 1
    return n, m, r


def get_f(t, x):
    """ Функция получения вектора правой части
        расширенной модели

        Parameters
        ----------
            t: float
                текущий момент времени
            x: np.ndarray
                n-вектор состояния

        Returns
        -------
            np.ndarray
                n-вектор правой части
                расширенной модели
    """
    T = 0.1
    e = math.exp(-x[2]*T)
    return np.array([[x[0] + (1 - e) * x[1] / x[2]],
                     [e * x[1]],
                     [x[2]],
                     [x[3]]])


def get_F(t, x):
    """ Функция получения матрица частных производных
        вектора правой части расширенной модели
        по каждому состоянию

        Parameters
        ----------
            t: float
                текущий момент времени
            x: np.ndarray
                n-вектор состояния

        Returns
        -------
            np.ndarray
                (n x n)-матрица частных производных
                вектора правой части по каждому состоянию
    """
    T = 0.1
    e = math.exp(-x[2]*T)
    temp = 1. / x[2]
    return np.array([[1, temp * (1 - e), x[1] * (temp**2) * (e * (1 + T * x[2]) - 1), 0],
                     [0, e, -x[1] * T * e, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def get_psi(t, x):
    """ Функция получения матрицы управления
        расширенной модели

        Parameters
        ----------
            t: float
                текущий момент времени
            x: np.ndarray
                n-вектор состояния

        Returns
        -------
            np.ndarray
                (n x r)-матрица управления
    """
    T = 0.1
    e = math.exp(-x[2] * T)
    temp = 1./x[2]
    return np.array([[x[3] * temp * (T - temp + temp * e)],
                     [x[3] * temp * (1 - e)],
                     [0],
                     [0]])


def get_P0():
    """ Функция получения ковариационной матрицы оценки
        вектора начального состояния расширенной модели

        Returns
        -------
            np.ndarray
                (n x n)-ковариационная матрица
                вектора начального состояния
    """
    # return np.array([[0.1, 0.1, 1, 0.1],
    #                  [0.1, 0, 0, 0],
    #                  [1, 0, 100, 0],
    #                  [0.1, 0, 0, 1]])
    return np.array([[0.1, 0.1, 1, 0.1],
                     [0.1, 0, 0, 0],
                     [1, 0, 110, 0],
                     [0.1, 0, 0, 1]])


def get_h(t, x):
    """ Функция получения вектора измерения
        расширенной модели

        Parameters
        ----------
            t: float
                текущий момент времени
            x: np.ndarray
                n-вектор состояния

        Returns
        -------
            np.ndarray
                m-вектор измерения
    """
    return np.array([x[0]])


def get_H(t, x):
    """ Функция получения матрица частных производных
        вектора измерения расширенной модели
        по каждому состоянию

        Parameters
        ----------
            t: float
                текущий момент времени
            x: np.ndarray
                n-вектор состояния

        Returns
        -------
            np.ndarray
                (m x n)-матрица частных производных
                вектора измерения по каждому состоянию
    """
    return np.array([[1, 0, 0, 0]])


def get_u(t):
    """ Функция получения вектора управления

        Parameters
        ----------
            t: float
                текущий момент времени

        Returns
        -------
            np.ndarray
                r-вектор управления
    """
    return np.array([[75]])


def get_R(t, theta):
    """ Функция получения ковариационной матрицы ошибки измерения

        Parameters
        ----------
            theta: np.ndarray
                s-вектор параметров
            t: float
                текущий момент времени

        Returns
        -------
            np.ndarray
                (m x m)-ковариационная матрица ошибки измерения
    """
    return np.array([[0.1]])


def get_G(t):
    """ Функция получения матрицы возмущения
        расширенной модели

        Parameters
        ----------
            t: float
                текущий момент времени

        Returns
        -------
            np.ndarray
                (n x p)-матрица возмущения
    """
    return np.array([[0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1]])


def get_Q(t):
    """ Функция получения ковариационной матрицы шума
        расширенной модели

        Parameters
        ----------
            t: float
                текущий момент времени

        Returns
        -------
            np.ndarray
                (p x p)-ковариационная матрица шума модели
    """
    return np.array([[1e-4, 0],
                     [0, 1e-4]])


def get_x0(theta):
    """ Функция получения вектора начального состояния
        расширенной модели

        Returns
        -------
            np.ndarray
                n-вектор начального состояния
    """
    return np.array([0, 0, theta[0], theta[1]])