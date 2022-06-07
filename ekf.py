import re
import numpy.linalg as npl
from position_control_system.extended_model import *
from position_control_system.common_data import *


def re_ekf(init_theta, f_name, s):
    """ Функция рекуррентного оценивания параметров
        с импользованием расширенного фильтра Калмана

        Parameters
        ----------
            init_theta: np.ndarray
                s-вектор начальных значений параметров
            f_name: str
                имя файла с данными наблюдений
            s: int
                размер вектора параметров

        Returns
        ----------
            theta_est: np.ndarray
                (N + 1) x s-вектор оценок параметров
    """

    # Блок подготовки
    # -----------------------------------------
    # Инициализация начальных условия для работы фильтра
    n, m, r = get_data()
    x_prev = get_x0(init_theta)
    p_prev = get_P0()
    x_filtered = np.array([x_prev[n - s:]])
    t = get_t0()
    N = 0
    # -----------------------------------------
    # Открытие файла с данными наблюдений для чтения
    with open(f_name, 'r') as f:
        # Считывание нового наблюдения, если оно есть
        for line in f:
            # Сохранение считанного наблюдения в нужном формате
            y = np.array(re.split('[ 	]', line)).astype(np.float)
            # Увеличение счетчика наблюдений на единицу
            N += 1

            # Блок предсказания
            # -----------------------------------------
            x_prediction, p_prediction = prediction(x_prev, p_prev, t)
            x_prediction = np.reshape(x_prediction, n)
            # -----------------------------------------

            # Блок коррекции
            # -----------------------------------------
            t = get_t_next(t)
            x_prev, p_prev = update(n, x_prediction, p_prediction, y, t)
            # Сохранение полученной оценки
            x_filtered = np.append(x_filtered, np.array([x_prev[n - s:]]), axis=0)
            # -----------------------------------------

    return x_filtered


def prediction(x_prev, p_prev, t) -> tuple[np.ndarray, np.ndarray]:
    """ Блок предсказания для расширенного фильтра Калмана

        Parameters
        ----------
            x_prev: ndarray
                n-вектор состояния на предыдущем шаге по времени
            p_prev: ndarray
                (n x n)-ковариационная матрица ошибки оценивания
                на предыдущем шаге по времени
            t: float
                текущий момент времени

        Returns
        -------
            x_prediction, p_prediction: ndarray, ndarray
                n-вектор оценки одношагового прогнозирования,
                (n x n)-ковариационная матрица ошибки
                одношагового прогнозирования
    """

    # Обновление матриц модели
    # -----------------------------------------
    f = get_f(t, x_prev)
    F = get_F(t, x_prev)
    psi = get_psi(t, x_prev)
    u = get_u(t)
    G = get_G(t)
    Q = get_Q(t)
    # -----------------------------------------

    # Оценка одношагового прогнозирования
    x_prediction = f + psi @ u

    # Вычисление ковариационной матрицы ошибки
    # одношагового прогнозирования
    p_prediction = F @ p_prev @ F.T + G @ Q @ G.T

    return x_prediction, p_prediction


def update(n, x_prediction, p_prediction, y_curr, t) \
        -> tuple[np.ndarray, np.ndarray]:
    """ Блок коррекции для расширенного фильтра Калмана

        Parameters
        ----------
            n: int
                размер вектора состояния
            x_prediction: ndarray
                n-вектор оценки одношагового прогнозирования
            p_prediction: ndarray
                (n x n)-ковариационная матрица ошибки
                одношагового прогнозирования
            y_curr: ndarray
                m-вектор измерения в текущий момент времени
            t: float
                текущий момент времени

        Returns
        -------
            x_filtered, p_filtered: ndarray, ndarray
                n-вектор оценки фильтрации
                (n x n)-ковариационная матрица ошибки оценивания
    """

    # Обновление матриц модели
    # -----------------------------------------
    h = get_h(t, x_prediction)
    H = get_H(t, x_prediction)
    R = get_R(t, x_prediction)
    # -----------------------------------------

    # Вычисление коэффициента усиления Калмана
    # -----------------------------------------
    b = H @ p_prediction @ H.T + R
    k = p_prediction @ H.T @ npl.pinv(b)
    # -----------------------------------------

    # Оценка фильтрации
    # -----------------------------------------
    e = y_curr - h
    x_filtered = x_prediction + k @ e
    # -----------------------------------------

    # Обновление ковариационной матрицы ошибки оценивания
    p_filtered = (np.eye(n) - k @ H) @ p_prediction

    return x_filtered, p_filtered
