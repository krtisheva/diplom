from models.position_control_system import *
import numpy as np
import numpy.linalg as npl


def eval_ci(i, n, s):
    """ Функция заполнения вспомогательной блочной матрицы

        Parameters
        ----------
            i: int
                текущее количество наблюдений
            n: int
                размер вектора состояний
            s: int
                размер вектора параметров

        Returns
        ----------
            ci: np.ndarray
                вспомогательная блочная матрица размером (n x n(s + 1))
    """
    ci = np.zeros(shape=(n, n * (s + 1)))
    I = np.eye(n)
    ci[:, i * n:(i + 1) * n] = I
    return ci


def grad_imf_evaluation(theta, N, n, m, r, s, y) \
        -> tuple[np.ndarray, np.ndarray]:
    """ Функция вычисления градиента критерия максимального
        правдоподобия и информационной матрицы Фишера

        Parameters
        ----------
            theta: np.ndarray
                s-вектор параметров
            N: int
                текущее количество наблюдений
            n: int
                размер вектора состояний
            m: int
                размер вектора измерений
            r: int
                размер вектора управления
            s: int
                размер вектора параметров
            y: np.ndarray
                текущее наблюдение

        Returns
        ----------
            imf, grad_N: np.ndarray, np.ndarray
                (s x s) информационная матрица Фишера,
                s-вектор градиента критерия максимального
                правдоподобия.
    """
    # Блок подготовки
    # -----------------------------------------
    # Выделение памяти
    imf = np.zeros(shape=(s, s))
    xA = np.zeros(shape=(n * (s + 1), 1))
    grad_N = np.zeros(s)

    # Инициализация начальных условий
    dxt0dtheta = get_dxt0dtheta()
    xA[:n] = get_xt0()
    for i in range(s):
        xA[(i + 1) * n:(i + 2) * n] = dxt0dtheta[i]
    # -----------------------------------------
    t = get_t0()
    for k in range(N):
        # Обновления матриц модели
        # -----------------------------------------
        F = get_F(t, theta)
        Psi = get_Psi(t, theta)
        u = get_u(t)
        dFdtheta = get_dFdtheta(t, theta)
        dPsidtheta = get_dPsidtheta(t, theta)
        t = get_t_next(t)
        H = get_H(t, theta)
        R = get_R(t, theta)
        inv_R = npl.inv(R)
        dHdtheta = get_dHdtheta(t, theta)
        dRdtheta = get_dRdtheta(t, theta)
        # -----------------------------------------

        # Заполнение расширенной матрицы состояния и
        # расширенной матрицы управления
        # -----------------------------------------
        FA = np.zeros(shape=(n * (s + 1), n * (s + 1)))
        PsiA = np.zeros(shape=(n * (s + 1), r))
        FA[:n, :n] = F
        PsiA[:n] = Psi
        for i in range(s):
            FA[(i + 1) * n:(i + 2) * n, (i + 1) * n:(i + 2) * n] = F
            FA[(i + 1) * n:(i + 2) * n, :n] = dFdtheta[i]
            PsiA[(i + 1) * n:(i + 2) * n] = dPsidtheta[i]
        # -----------------------------------------

        # Вычисление значения расширенного вектора состояния с номером k + 1
        # и математиеского ожидания
        xA = FA @ xA + PsiA @ u
        expected_value = xA @ xA.T

        # Вычисление градиента критерия максимального правдоподобия
        # и информационной матрицы Фишера для N наблюдений
        for i in range(s):
            if k == N - 1:
                # Вычисление вектора ошибки оценивания
                eps = y - H @ xA[:n]
                # Вычисление производной вектора ошибки оценивания по i-ому параметру
                depsdtheta = -dHdtheta[i] @ xA[:n] - H @ xA[(i + 1) * n:(i + 2) * n]
                # Вычисление i-ой компоненты вектора градиента критерия
                # максимального правдоподобия в момент N-ого наблюдения
                # -----------------------------------------
                a = np.transpose(depsdtheta) @ inv_R @ eps
                b = 1 / 2. * (eps.T @ inv_R @ dRdtheta[i, np.newaxis] @ inv_R @ eps)
                grad_N[i] = a - b
                # -----------------------------------------
            for j in range(s):
                # Блок вычисления приращения i-ого j-ого элемента информационной матрицы,
                # отвечающего текущему значению k
                # -----------------------------------------
                c0 = eval_ci(0, n, s)
                cj = eval_ci(j + 1, n, s)
                ci = eval_ci(i + 1, n, s)
                sp1 = dHdtheta[i] @ c0 @ expected_value @ c0.T @ np.transpose(dHdtheta[j]) @ inv_R
                sp2 = dHdtheta[i] @ c0 @ expected_value @ cj.T @ H.T @ inv_R
                sp3 = H @ ci @ expected_value @ c0.T @ np.transpose(dHdtheta[j]) @ inv_R
                sp4 = H @ ci @ expected_value @ cj.T @ H.T @ inv_R
                sp5 = np.reshape(dRdtheta[i] @ inv_R @ dRdtheta[j], (m, m)) @ inv_R
                imf[i][j] += np.trace(sp1) + np.trace(sp2) + np.trace(sp3) + np.trace(sp4) + np.trace(sp5)
                # -----------------------------------------
    return imf, grad_N
