import re

import numpy as np
import numpy.linalg as npl
from models.extended_position_control_system import *
import matplotlib.pyplot as plt


def re_ekf(n, m, s, f_name, init_theta, theta_true):
    """ Процедура фильрации с помощью фильтра Калмана

        Parameters
        ----------
        n: int
            Размерность вектора состояний
        m: int
            Размерность вектора измерений
        n_obs: int
            Число наблюдений эксперимента
        x0: ndarray
            n-вектор-столбец начального состояния процесса
        y: ndarray
            (n_obs x m)-массив всех наблюдений, в каждый момент времени
        theta:
            l-вектор параметров

        Returns
        -------
        x_filtered, y_filtered: ndarray, ndarray
            (n_obs x n)-массив векторов состояния процесса после процедуры фильтрации
            в каждый момент времени,
            (n_obs x m)-массив векторов наблюдения после процедуры фильтрации
            в каждый момент времени
        """

    # блок подготовки
    # -----------------------------------------
    # Инициализация начальных условия для работы фильтра
    x_prev = get_x0(init_theta)
    p_prev = get_P0()
    x_filtered = np.array([x_prev])
    t = get_t0()
    N = 0
    # -----------------------------------------
    with open(f_name, 'r') as f:
        for line in f:
            # Ввод данных
            y = np.array(re.split('[ 	]', line)).astype(np.float)
            N += 1

            # блок предсказания
            # -----------------------------------------
            x_prediction, p_prediction = prediction(x_prev, p_prev, t)
            x_prediction = np.reshape(x_prediction, n)
            # -----------------------------------------

            # блок коррекции
            # -----------------------------------------
            t = get_t_next(t)
            x_prev, p_prev = update(n, x_prediction, p_prediction, y, t)
            x_filtered = np.append(x_filtered, np.array([x_prev]), axis=0)
            # -----------------------------------------
    visualize_filter_results_theta(N, n, s, x_filtered, theta_true)
    return x_filtered[N][n - s:]


def prediction(x_prev, p_prev, t):
    """ Блок предсказания для непрерывно-дискретного фильтра Калмана

    Parameters
    ----------
    n: int
        Размерность вектора состояний
    x_prev: ndarray
        n-вектор состояния на предыдущем шаге по времени (x(tk|tk)))
    p_prev: ndarray
        (n x n)-матрица предсказания на предыдущем шаге по времени (P(tk|tk))
    t_prev: float
        Предыдущий момент времени
    theta: ndarray
        l-вектор параметров
    Returns
    -------
    x_prediction, p_prediction: ndarray, ndarray
        Предсказанный n-вектор состояния (x(tk+1|tk)), предсказанная (n x n)-матрица P(tk+1|tk)
    """
    # блок подготовки
    # -----------------------------------------
    f = get_f(t, x_prev)
    dfdx = get_dfdx(t, x_prev)
    psi = get_psi(t, x_prev)
    u = get_u(t)
    G = get_G(t)
    Q = get_Q(t)
    # -----------------------------------------

    # Нахождение предсказанного вектора состояния x(tk+1|tk)
    x_prediction = f + psi @ u

    # Нахождение матрицы предсказания P(tk+1|tk)
    p_prediction = dfdx @ p_prev @ dfdx.T + G @ Q @ G.T
    return x_prediction, p_prediction


def update(n, x_prediction, p_prediction, y_curr, t):
    """ Блок коррекции для фильтра Калмана

    Parameters
    ----------
    n: int
        Размерность вектора состояний
    m: int
            Размерность вектора измерений
    x_prediction: ndarray
        Предсказанный n-вектор состояния (x(tk+1|tk))
    p_prediction: ndarray
        Предсказанная (n x n)-матрица P(tk+1|tk)
    y_curr: ndarray
        m-вектор измерений в текущий момент времени (y(tk+1))

    Returns
    -------
    x_filtered, y_filtered, p_filtered: ndarray, ndarray, ndarray
        n-вектор состояния процесса после процедуры коррекции (x(tk+1|tk+1))
        m-вектор наблюдения после процедуры коррекции (y(tk+1|tk+1))
        (n x n)-матрица предсказания P(tk+1|tk+1)
    """
    # блок подготовки
    # -----------------------------------------
    h = get_h(t, x_prediction)
    dhdx = get_dhdx(t, x_prediction)
    dhdx_t = dhdx.T
    R = get_R(t, x_prediction)
    # -----------------------------------------
    e = y_curr - h
    b = dhdx @ p_prediction @ dhdx_t + R
    k = p_prediction @ dhdx_t @ npl.pinv(b)
    x_filtered = x_prediction + k @ e            # Вычисление отфильтрованных состояний x(tk+1|tk+1)
    p_filtered = (np.eye(n) - k @ dhdx) @ p_prediction             # Вычисление P(tk+1|tk+1)
    return x_filtered, p_filtered


def visualize_filter_results_theta(N, n, s, x_filtered, theta_true):
    iterations = np.array(np.arange(get_t0(), N + 1, get_t_step()))
    plt.Figure()
    theta = np.zeros((N + 1, s))
    for j in range(N + 1):
        theta[j] = theta_true

    for i in range(s):
        plt.subplot(s, 1, i + 1)
        plt.plot(iterations, x_filtered[:, n-s+i], label='фильтр')
        plt.plot(iterations, theta[:, i], label='истина')
        plt.xlabel("k")
        label = 'theta' + str(i+1)
        plt.ylabel(label)
        plt.legend()
    plt.show()
