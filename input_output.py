from position_control_system.conventional_model import *
from position_control_system.common_data import *


def generate_data(N, theta_true, f_name):
    """ Процедура генерации наблюдений эксперимента

        Parameters
        ----------
            N: int
                текущее количество наблюдений
            theta_true: np.ndarray
                s-вектор истинных параметров
            f_name: str
                имя файла для записи сгенерированных данных
    """

    # Блок подготовки
    # -----------------------------------------
    n, m, r = get_data()
    mu = np.zeros(m)
    x = get_xt0()
    t = get_t0()
    # -----------------------------------------

    # Открытие файла для записи сгенерированных данных
    with open(f_name, 'w') as f:
        for k in range(N):
            # Обновление матриц модели
            # -----------------------------------------
            F = get_F(t, theta_true)
            Psi = get_Psi(t, theta_true)
            u = get_u(t)
            t = get_t_next(t)
            R = get_R(t, theta_true)
            H = get_H(t, theta_true)
            # -----------------------------------------

            # Вычисление значения вектора состояния и
            # вектора измерения с номером k + 1
            # -----------------------------------------
            x = F @ x + Psi @ u
            v = np.random.multivariate_normal(mu, R)
            y = np.reshape(np.matmul(H, x) + v, m)
            # -----------------------------------------

            # Вывод в файл сгенерированного вектора
            # измерения с номером k + 1
            # -----------------------------------------
            for j in range(m - 1):
                f.write(f'{(y[j]):f} ')
            f.write(f'{(y[m - 1]):f}\n')
            # -----------------------------------------


def output(theta_est, error, time, method, s, f):
    """ Процедура вывода полученных результатов

        Parameters
        ----------
            theta_est: np.ndarray
                s-вектор оценок параметров
            error: np.ndarray
                s-вектор ошибок оценок параметров
            time: float
                время работы метода
            method: str
                название метода
            s: int
                размер вектора параметров
            f:
                указатель на фай файл для записи результатов
    """
    # Вывод названия метода
    f.write(method.center(30))
    f.write('\n------------------------------\n')

    # Вывод полученной оценки вектора паметров
    f.write('%-5s' % 'theta')
    for i in range(s):
        f.write(' %10.4f' % theta_est[i])

    # Вывод ошибки оценки вектора паметров
    f.write('\n%-5s' % 'error')
    for i in range(s):
        f.write(' %10.4f' % error[i])

    # Вывод времени работы метода
    f.write('\n%-5s %10.4f\n' % ('time', time))
    f.write('------------------------------\n\n')
