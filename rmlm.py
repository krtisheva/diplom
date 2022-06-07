import re
from grad_imf import *


def rmle(init_theta, f_name, s):
    """ Функция вычисления оценки параметров рекурентным
        методом максимального правдоподобия

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
    n, m, r = get_data()
    N = 0
    theta_est = np.array([init_theta])
    theta_curr = init_theta
    # -----------------------------------------

    # Открытие файла с данными наблюдений для чтения
    with open(f_name, 'r') as f:
        # Считывание нового наблюдения, если оно есть
        for line in f:
            # Сохранение считанного наблюдения в нужном формате
            y = np.array(re.split('[ 	]', line)).astype(np.float)
            # Увеличение счетчика наблюдений на единицу
            N += 1
            # Вычисление градиента критерия максимального правдоподобия
            # и информационной матрицы Фишера
            imf, grad = grad_imf_evaluation(theta_curr, N, n, m, r, s, y)
            # Вычисление новой оценки вектора параметров
            theta_curr = theta_curr - npl.pinv(imf) @ grad
            # Сохранение полученной оценки
            theta_est = np.append(theta_est, np.array([theta_curr]), axis=0)

    return theta_est
