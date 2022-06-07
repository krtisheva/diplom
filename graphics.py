import matplotlib.pyplot as pl
from position_control_system.common_data import *
from relative_errors import *


def visualize_results(N, s, theta_est_rmlm, theta_est_ekf, theta_true)\
        -> tuple[np.ndarray, np.ndarray]:
    """ Функция графического отображения результатов и
        вычисления относительной ошибки оценок параметров

        Parameters
        ----------
            N: int
                текущее количество наблюдений
            s: int
                размер вектора параметров
            theta_est_rmlm: np.ndarray
                s-вектор значений оценок неизвестных параметров, полученных
                рекуррентным методом максимального правдоподобия
            theta_est_ekf: np.ndarray
                s-вектор значений оценок неизвестных параметров, полученных
                с помощью расширенного фильтра Калмана
            theta_true: np.ndarray
                s-вектор значений истинных параметров

        Returns
        ----------
            error_rml, error_ekf: np.ndarray, np.ndarray
                s-вектор ошибки оценки, полученной рекуррентным методом
                максимального правдоподобия,
                s-вектор ошибки оценки, полученной с помощью
                расширенного фильтра Калмана.
    """

    # Блок подготовки
    # -----------------------------------------
    # Выделение памяти
    error_rml = np.zeros((N + 1, s))
    error_ekf = np.zeros((N + 1, s))
    theta = np.zeros((N + 1, s))
    # Получение вектора значений количества наблюдений
    # во все маменты времени
    iterations = np.array(range(N + 1))

    # Заполнение массивов
    for i in range(N + 1):
        theta[i] = theta_true
        # Нахождение относительных ошибок
        for j in range(s):
            error_rml[i][j] = relative_error_theta_i(theta_true[j], theta_est_rmlm[i][j])
            error_ekf[i][j] = relative_error_theta_i(theta_true[j], theta_est_ekf[i][j])
    # -----------------------------------------

    # Построение графика зависимости оценок параметров от количества наблюдений
    # -----------------------------------------
    pl.Figure()
    for i in range(s):
        pl.subplot(s, 1, i + 1)
        pl.plot(iterations, theta_est_rmlm[:, i], label='RMLM')
        pl.plot(iterations, theta_est_ekf[:, i], label='EKF')
        pl.plot(iterations, theta[:, i], '--', color='black', linewidth=1)
        label = 'theta' + str(i+1)
        pl.ylabel(label)
        pl.legend()
        pl.grid(True)
    pl.xlabel("k")
    pl.show()
    # -----------------------------------------

    # Построение графика зависимости ошибок оценок параметров от количества наблюдений
    visualize_results_error(s, error_rml, error_ekf, iterations)

    return error_rml[N], error_ekf[N]


def visualize_results_error(s, error_rml, error_ekf, iterations):
    """ Процедура графического отображения зависимости ошибок
        оценок параметров от количества наблюдений

        Parameters
        ----------
            s: int
                размер вектора параметров
            error_rml: np.ndarray
                s-вектор значений ошибок оценок неизвестных параметров,
                полученных рекуррентным методом максимального правдоподобия
            error_ekf: np.ndarray
                s-вектор значений ошибок оценок неизвестных параметров,
                полученных с помощью расширенного фильтра Калмана
            iterations: np.ndarray
                (N + 1)-вектор значений количества наблюдений
                во все маменты времени
    """

    # Построение графика зависимости ошибок оценок параметров от количества наблюдений
    pl.Figure()
    for i in range(s):
        pl.subplot(s, 1, i + 1)
        pl.plot(iterations, error_rml[:, i], label='RMLM')
        pl.plot(iterations, error_ekf[:, i], label='EKF')
        label = 'relative error theta' + str(i+1)
        pl.ylabel(label)
        pl.legend()
        pl.grid(True)
    pl.xlabel("k")
    pl.show()
