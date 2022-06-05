from models.position_control_system import *


def generate_data(m, theta, N, f_name):
    """ Процедура генерации наблюдений эксперимента

        Parameters
        ----------
            m: int
                размер вектора измерений
            theta: np.ndarray
                s-вектор параметров
            N: int
                текущее количество наблюдений
            f_name: str
                имя файла для записи сгенерированных данных
    """

    mu = np.zeros(m)
    x = get_xt0()
    t = get_t0()

    # Открытие файла для записи сгенерированных данных
    with open(f_name, 'w') as f:
        for k in range(N):
            # Обновление матриц модели
            # -----------------------------------------
            F = get_F(t, theta)
            Psi = get_Psi(t, theta)
            u = get_u(t)
            t = get_t_next(t)
            R = get_R(t, theta)
            H = get_H(t, theta)
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
