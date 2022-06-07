import math


def relative_error_theta_i(theta_true, theta_est):
    """ Функция вычисления относительной ошибки
        оценки параметра

        Parameters
        ----------
            theta_true: np.ndarray
                истинное значение параметра
            theta_est: np.ndarray
                оценка параметра

        Returns
        ----------
            float
                относительноая ошибка оценки параметра
    """
    error = math.fabs(theta_est - theta_true) / math.fabs(theta_true)
    return error
