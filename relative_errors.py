import math


def relative_error_theta(theta_true, theta_est, s):
    """ Функция вычисления относительной ошибки
        в пространстве параметров

        Parameters
        ----------
            theta_true: np.ndarray
                s-вектор значений истинных параметров
            theta_est: np.ndarray
                s-вектор значений оценок неизвестных параметров
            s: int
                размер вектора параметров

        Returns
        ----------
            float
                относительноая ошибка в пространстве параметров
    """
    error = 0.
    for i in range(s):
        error += math.fabs(theta_est[i] - theta_true[i]) / math.fabs(theta_true[i])
    return error / s
