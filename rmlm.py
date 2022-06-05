import re
from grad_imf import *
from graphics import *


def rmle(n, m, r, s, init_theta, theta_true, f_name):
    N = 0
    theta_arr = np.array([init_theta])
    theta_curr = init_theta

    with open(f_name, 'r') as f:
        for line in f:
            y = np.array(re.split('[ 	]', line)).astype(np.float)
            N += 1
            imf, grad = grad_imf_evaluation(theta_curr, N, n, m, r, s, y)
            theta_curr = theta_curr - npl.pinv(imf) @ grad
            theta_arr = np.append(theta_arr, np.array([theta_curr]), axis=0)

    graphics_theta(theta_arr, theta_true, N, s)
    return theta_arr[N]
