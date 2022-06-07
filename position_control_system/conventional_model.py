import math
import numpy as np


def get_data() -> tuple[int, int, int]:
    """ Функция получения общей информации об исследуемой модели

        Returns
        -------
            n, m, r: int, int, int
                размер вектора состояний,
                размер вектора измерений,
                размер вектора управления.
    """
    n = 2
    m = 1
    r = 1
    return n, m, r


def get_F(t, theta):
    """ Функция получения матрицы состояния

        Parameters
        ----------
            t: float
                текущий момент времени
            theta: np.ndarray
                s-вектор параметров

        Returns
        -------
            np.ndarray
                (n x n)-матрица состояния
    """
    T = 0.1
    e = math.exp(-theta[0] * T)
    return np.array([[1, (1. / theta[0]) * (1 - e)],
                     [0, e]])


def get_dFdtheta(t, theta):
    """ Функция получения матрицы значений частных производных
        матрицы состояния по параметрам

        Parameters
        ----------
            t: float
                текущий момент времени
            theta: np.ndarray
                s-вектор параметров

        Returns
        -------
            np.ndarray
                (s x n x n)-матрица частных производных
                матрицы состояния по параметрам
    """
    T = 0.1
    temp = 1. / theta[0]
    e = math.exp(-theta[0] * T)
    dFdtheta1 = [[0, temp * (-temp + e * (temp + T))],
                 [0, -T * e]]
    dFdtheta2 = [[0, 0],
                 [0, 0]]
    return np.array([dFdtheta1, dFdtheta2])


def get_Psi(t, theta):
    """ Функция получения матрицы управления

        Parameters
        ----------
            t: float
                текущий момент времени
            theta: np.ndarray
                s-вектор параметров

        Returns
        -------
            np.ndarray
                (n x r)-матрица управления
    """
    T = 0.1
    temp = 1. / theta[0]
    e = math.exp(-theta[0] * T)
    return np.array([[theta[1] * temp * (T - temp + temp * e)],
                     [theta[1] * temp * (1 - e)]])


def get_dPsidtheta(t, theta):
    """ Функция получения матрицы значений частных производных
        матрицы управления по параметрам

        Parameters
        ----------
            t: float
                текущий момент времени
            theta: np.ndarray
                s-вектор параметров

        Returns
        -------
            np.ndarray
                (s x n x r)-матрица частных производных
                матрицы управления по параметрам
    """
    T = 0.1
    temp = 1. / theta[0]
    e = math.exp(-theta[0] * T)
    dPsidtheta1 = [[theta[1] * (temp ** 2) * (-T + 2 * temp - e * (T + 2 * temp))],
                   [theta[1] * (temp ** 2) * (e * (T * theta[0] + 1) - 1)]]
    dPsidtheta2 = [[temp * (T - temp + temp * e)],
                   [temp * (1 - e)]]
    return np.array([dPsidtheta1, dPsidtheta2])


def get_H(t, theta):
    """ Функция получения матрицы измерения

        Parameters
        ----------
            t: float
                текущий момент времени
            theta: np.ndarray
                s-вектор параметров

        Returns
        -------
            np.ndarray
                (m x n)-матрица измерения
    """
    return np.array([[1, 0]])


def get_dHdtheta(t, theta):
    """ Функция получения матрицы значений частных производных
        матрицы измерения по параметрам

        Parameters
        ----------
            t: float
                текущий момент времени
            theta: np.ndarray
                s-вектор параметров

        Returns
        -------
            np.ndarray
                (s x m x n)-матрица частных производных
                матрицы измерения по параметрам
    """
    dHdtheta1 = [[0, 0]]
    dHdtheta2 = [[0, 0]]
    return np.array([dHdtheta1, dHdtheta2])


def get_dRdtheta(t, theta):
    """ Функция получения матрицы значений частных производных
        ковариационной матрицы ошибки измерения по параметрам

        Parameters
        ----------
            t: float
                текущий момент времени
            theta: np.ndarray
                s-вектор параметров

        Returns
        -------
            np.ndarray
                (s x m x m)-матрица частных производных ковариационной
                матрицы ошибки измерения по параметрам
    """
    dRdtheta1 = [0]
    dRdtheta2 = [0]
    return np.array([dRdtheta1, dRdtheta2])


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


def get_xt0():
    """ Функция получения вектора начального состояния

        Returns
        -------
            np.ndarray
                n-вектор начального состояния
    """
    return np.array([[0],
                     [0]])


def get_dxt0dtheta():
    """ Функция получения матрицы значений частных производных
        вектора начального состояния по параметрам

        Returns
        -------
            np.ndarray
                (s x n)-матрица частных производных вектора
                начального состояния по параметрам
    """
    dxt0dtheta1 = [[0],
                   [0]]
    dxt0dtheta2 = [[0],
                   [0]]
    return np.array([dxt0dtheta1, dxt0dtheta2])
