from generation import *
#from rmlm import *
from ekf import *
from relative_errors import *


n, m, r, s, theta_true = get_data()
N = int(input('Введите число измерений:\n'))
f_name = 'data/position_control_system' + str(N)
answer = input('Хотите сгенерировать новые данные? (y/n)\n')
if answer == 'y':
    generate_data(m, theta_true, N, f_name)

#answer = input('Введите начальное значение параметров через пробел\n')
#init_theta = np.array(re.split('[ 	]', answer)).astype(np.float)
init_theta = np.array([3, 0.5])
#theta_est = rmle(n, m, r, s, init_theta, theta_true, f_name)
theta_est = re_ekf(n, m, s, f_name, init_theta, theta_true)
print('Ошибка в пространстве параметров = ', relative_error_theta(theta_true, theta_est, s))
