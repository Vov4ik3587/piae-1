# Лабораторная работа №1
# Вариант: 6
# Модель кубическая на отрезке. Планы для анализа 9-12. Использовать Л-оптимальный критерий

# %% Импортируем модули
import numpy as np
import matplotlib.pyplot as plt


# %% Сделали Классы

class Plan:

    def __init__(self, name, spectrum, weights):
        self.name_plan = name
        self.spectrum = spectrum
        self.weights = weights
        self.d_criterion = 0.
        self.a_criterion = 0.
        self.e_criterion = 0.
        self.f2_criterion = 0.
        self.l_criterion = 0.
        self.mv_criterion = 0.
        self.g_criterion = 0.


class Calculator:

    @staticmethod
    def D_criterion(disp_mat):  # min
        return np.linalg.det(disp_mat)

    @staticmethod
    def A_criterion(disp_mat):  # min
        return np.trace(disp_mat)

    @staticmethod
    def E_criterion(disp_mat):  # min
        eig = np.linalg.eig(disp_mat)
        return np.max(eig[0])

    @staticmethod
    def F2_criterion(disp_mat):  # m - число неизвестных параметров(тет), берем min
        return np.sqrt(0.2 * np.trace(np.linalg.matrix_power(disp_mat, 2)))

    @staticmethod
    def L_criterion(disp_mat):  # берем min
        lyambda = (np.linalg.eig(disp_mat))[0]
        lyambda_avg = np.mean(lyambda)
        return np.sum(np.square(lyambda - lyambda_avg))

    @staticmethod
    def MV_criterion(disp_mat):  # берем min
        return np.max(np.diagonal(disp_mat))

    @staticmethod
    def G_criterion(disp_mat, plan_tmp):  # min

        d = np.array(
            [np.vstack(Calculator.func(x)).T @ disp_mat @ np.vstack(Calculator.func(x)) for x in plan_tmp.spectrum]
        )
        return np.max(d)

    @staticmethod
    def calculate_info_matrix(plan_tmp):
        info_mat_tmp = np.array(
            [np.dot(p, np.vstack(Calculator.func(x)) @ np.vstack(Calculator.func(x)).T)
             for x, p in zip(plan_tmp.spectrum, plan_tmp.weights)
             ]
        )
        return sum(info_mat_tmp)

    @staticmethod
    def calculate_disp_matrix(info_mat_tmp):
        return np.linalg.inv(info_mat_tmp)

    @staticmethod
    def func(x):
        return np.array([1, x, x ** 2, x ** 3])


# %% Пункт 3. Вычисляем критерии для планов 9-12 и записываем полученные данные. Далее планы будут проранжированы

plans = (
    Plan('Plan №9', np.array([-1., -0.707, 0., 0.707, 1]), np.array([0.093, 0.248, 0.3178, 0.248, 0.093])),
    Plan('Plan №10', np.array([-1., -0.683, 0., 0.683, 1.]), np.array([0.107, 0.25, 0.286, 0.25, 0.107])),
    Plan('Plan №11', np.array([-1., -0.7379, 0., 0.7379, 1.]), np.array([0.1092, 0.2513, 0.2785, 0.2513, 0.1092])),
    Plan('Plan №12', np.array([-1., -0.7, 0., 0.7, 1.]), np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
)

for plan in plans:
    info_mat = Calculator.calculate_info_matrix(plan)
    disp_mat = Calculator.calculate_disp_matrix(info_mat)
    plan.d_criterion = Calculator.D_criterion(disp_mat)
    plan.a_criterion = Calculator.A_criterion(disp_mat)
    plan.e_criterion = Calculator.E_criterion(disp_mat)
    plan.f2_criterion = Calculator.F2_criterion(disp_mat)
    plan.l_criterion = Calculator.L_criterion(disp_mat)
    plan.mv_criterion = Calculator.MV_criterion(disp_mat)
    plan.g_criterion = Calculator.G_criterion(disp_mat, plan)

# %% Пункт 4. Выбрать один из планов (можно выбрать наилучший), веса выразить в виде функций от q,
# построить график изменения Л-критерия от q. Определить по графику оптимальное значение q и Л-критерия

spectrum = np.array([-1., -0.683, 0., 0.683, 1.])  # выбранный спектр
func_p = lambda q: np.array([q, 2.33 * q, 1 - 6.66 * q, 2.33 * q, q])

range_q = np.linspace(0.1, 0.15, 100)
l_criterion = []

for q in range_q:
    p_4 = func_p(q)
    info_mat_4 = sum(np.array(
        [np.dot(p, np.vstack(Calculator.func(x)) @ np.vstack(Calculator.func(x)).T)
         for x, p in zip(spectrum, p_4)
         ]
    ))
    disp_mat_4 = np.linalg.inv(info_mat_4)
    l_criterion.append(Calculator.L_criterion(disp_mat_4))
l_criterion = np.array(l_criterion)
plt.plot(range_q, l_criterion)
plt.xlabel('q')
plt.ylabel('L')
plt.grid(True)
plt.show()

l_min = np.min(l_criterion)
q_min = range_q[np.where(l_criterion == l_min)]
print(f'Минимальное значение Л-критерия: {l_min} достигается при q = {q_min}')
