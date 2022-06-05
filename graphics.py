import matplotlib.pyplot as pl
from models.position_control_system import *


# Построение графика зависимости оценок параметров от времени
def graphics_theta(theta_est, theta_true, N, s):
    """ Процедура построения графика зависимости
        оценок параметров от времени

        Parameters
        ----------
            theta_true: np.ndarray
                s-вектор значений истинных параметров
            theta_est: np.ndarray
                s-вектор значений оценок неизвестных параметров
            N: int
                текущее количество наблюдений
            s: int
                размер вектора параметров
    """
    iterations = np.array(np.arange(get_t0(), N + 1, get_t_step()))
    true = np.zeros(shape=(N + 1, s))
    for i in range(N + 1):
        true[i][0] = theta_true[0]
        true[i][1] = theta_true[1]

    for i in range(s):
        pl.Figure()
        pl.plot(iterations, theta_est[:, i], label='estimated')
        pl.plot(iterations, true[:, i], label='true')
        label = "theta" + str(i + 1)
        pl.xlabel("iteration")
        pl.ylabel(label)
        pl.legend()
        pl.show()
