import numpy as np


def get_data_theta() -> tuple[int, np.ndarray]:
    """ Функция получения общей информации о неизвестных параметрах

        Returns
        -------
            s, theta_true: int, np.ndarray
                размер вектора параметров,
                s-вектор истинных параметров.
    """
    s = 2
    theta_true = np.array([4.6, 0.787])
    return s, theta_true


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
    """ Функция получения следующего момента времени

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
