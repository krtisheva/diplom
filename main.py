import re
import time
from input_output import generate_data, output
from rmlm import rmle
from ekf import re_ekf
from graphics import *


s, theta_true = get_data_theta()
N = int(input('Введите число измерений:\n'))
f_in = 'data/position_control_system' + str(N)
f_out = 'results/position_control_system' + str(N)
answer = input('Хотите сгенерировать новые данные? (y/n)\n')
if answer == 'y':
    generate_data(N, theta_true, f_in)

answer = input('Введите начальное значение параметров через пробел\n')
init_theta = np.array(re.split('[ ]', answer)).astype(np.float)
start_rmlm = time.time()
theta_est_rmlm = rmle(init_theta, f_in, s)
end_rmlm = time.time()
start_ekf = time.time()
theta_est_ekf = re_ekf(init_theta, f_in, s)
end_ekf = time.time()
error_rmlm, error_ekf = visualize_results(N, s, theta_est_rmlm, theta_est_ekf, theta_true)

with open(f_out, 'w') as f:
    output(theta_est_rmlm[N], error_rmlm, end_rmlm-start_rmlm, 'RMLM', s, f)
    output(theta_est_ekf[N], error_ekf, end_ekf-start_ekf, 'EKF', s, f)

