import random
import math
from functools import reduce
from itertools import compress
from _pydecimal import Decimal
from scipy.stats import f, t
import numpy as np

x1min, x1max = -1, 2
x2min, x2max = -9, 6
x3min, x3max = -5, 8
x_avr_min = (x1min + x2min + x3min) / 3
x_avr_max = (x1max + x2max + x3max) / 3
m = 3
N = 15
ymin = 200 + x_avr_min
ymax = 200 + x_avr_max
p = 0.95

def generate_factors_table(raw_array):

    return [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], \
        row[0] * row[1] * row[2]]+list(map(lambda x: round(x ** 2, 5),\
         row))for row in raw_array]

def cochran_criteria(m, N, y_table, p=0.95):
    print("Перевірка однорідності дисперсій за критерієм Кохрена: ")
    y_variations = [np.var(i) for i in y_table]
    max_y_variation = max(y_variations)
    gp = max_y_variation/sum(y_variations)
    f1 = m - 1
    f2 = N
    q = 1-p
    gt = get_cochran_value(f1,f2, q)
    print(f"Gp = {gp:.3f}; Gt = {gt:.3f}; f1 = {f1}; f2 = {f2}; q = {q:.3f}")
    if gp < gt:
        print("Gp < Gt => дисперсія  однорідна  ")
        return True
    else:
        print("Gp > Gt => дисперсі неоднорідна ")
        return False

def x_i(i, raw_factors_table):
    try:
        assert i <= 10
    except:
        raise AssertionError("Error")
    with_null_factor = list(map(lambda x: [1] + x, generate_factors_table(raw_factors_table)))
    res = [row[i] for row in with_null_factor]
    return np.array(res)

def m_ij(*arrays):
    return np.average(reduce(lambda accum, el: accum*el, arrays))

def calculate_theoretical_y(x_table, b_coefficients, importance):

    x_table = [list(compress(row, importance)) for row in x_table]
    b_coefficients = list(compress(b_coefficients, importance))
    y_vals = np.array([sum(map(lambda x, b: x*b, row, b_coefficients)) for row in x_table])
    return y_vals

def get_cochran_value(f1, f2, q):
    partResult1 = q / f2 # (f2 - 1)
    params = [partResult1, f1, (f2 - 1) * f1]
    fisher = f.isf(*params)
    result = fisher/(fisher + (f2 - 1))
    return Decimal(result).quantize(Decimal('.0001'))

def get_student_value(f3, q):
    return Decimal(abs(t.ppf(q/2,f3))).quantize(Decimal('.0001'))

def get_fisher_value(f3,f4, q):
    return Decimal(abs(f.isf(q,f4,f3))).quantize(Decimal('.0001'))

x0 = [(x1max+x1min)/2, (x2max+x2min)/2, (x3max+x3min)/2]
detx = [abs(x1min - x0[0]), abs(x2min-x0[1]), abs(x3min-x0[2])]
l=1.215


raw_natur_table = [[x1min, x2min, x3min],
                   [x1min, x2max, x3max],
                   [x1max, x2min, x3max],
                   [x1max, x2max, x3min],

                   [x1min, x2min, x3max],
                   [x1min, x2max, x3min],
                   [x1max, x2min, x3min],
                   [x1max, x2max, x3max],

                   [-l*detx[0]+x0[0], x0[1], x0[2]],
                   [ l*detx[0]+x0[0], x0[1], x0[2]],
                   [x0[0], -l*detx[1]+x0[1], x0[2]],
                   [x0[0],  l*detx[1]+x0[1], x0[2]],
                   [x0[0], x0[1], -l*detx[2]+x0[2]],
                   [x0[0], x0[1],  l*detx[2]+x0[2]],
                   [x0[0],      x0[1],     x0[2]]]

raw_factors_table = [[-1, -1, -1],
                     [-1, +1, +1],
                     [+1, -1, +1],
                     [+1, +1, -1],

                     [-1, -1, +1],
                     [-1, +1, -1],
                     [+1, -1, -1],
                     [+1, +1, +1],

                     [-1.215, 0, 0],
                     [+1.215, 0, 0],
                     [0, -1.215, 0],
                     [0, +1.215, 0],
                     [0, 0, -1.215],
                     [0, 0, +1.215],
                     [0, 0, 0]]


factors_table = generate_factors_table(raw_factors_table)
print("Матриця кодованих значень Х")
for row in factors_table:
    print(row)

natur_table = generate_factors_table(raw_natur_table)
with_null_factor = list(map(lambda x: [1] + x, natur_table))

y_arr = [[random.random()*(ymax-ymin) + ymin for i in range(m)] for j in range(N)]
while not cochran_criteria(m, N, y_arr):
    m+=1
    y_arr = [[random.random()*(ymax - ymin) + ymin for i in range(m)] for j in range(N)]
y_i = np.array([np.average(row) for row in y_arr])
coefficients = [[m_ij(x_i(column, raw_factors_table)*x_i(row, raw_factors_table)) for column in range(11)] for row in range(11)]
free_values = [m_ij(y_i, x_i(i, raw_factors_table)) for i in range(11)]
beta_coef = np.linalg.solve(coefficients, free_values)

#Критерій Стьюдента
y_table = y_arr
print("\nПеревірка значимості коефіцієнтів регресії за критерієм Стьюдента: ")
average_variation = np.average(list(map(np.var, y_table)))
y_averages = np.array(list(map(np.average, y_table)))
variation_beta_s = average_variation/N/m
standard_deviation_beta_s = math.sqrt(variation_beta_s)
x_vals = [x_i(i, raw_factors_table) for i in range(11)]
t_i = np.array([abs(beta_coef[i])/standard_deviation_beta_s for i in range(len(beta_coef))])
f3 = (m-1)*N
q = 1-p
t = get_student_value(f3, q)
importance = [True if el > t else False for el in list(t_i)]

print("Оцінки коефіцієнтів bs: " + ", ".join(list(map(lambda x: str(round(float(x), 3)), beta_coef))),\
	"\nКоефіцієнти ts: " + ", ".join(list(map(lambda i: f"{i:.3f}", t_i))), \
	f"\nf3 = {f3}; q = {q:.3f}; tтабл = {t}")

beta_i = ["b0", "b1", "b2", "b3", "b12", "b13", "b23", "b123", "b11", "b22", "b33"]
x_i_names = list(compress(["1", "x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"], importance))
betas_to_print = list(compress(beta_coef, importance))

for i in range(len(importance)):
	if not importance[i]:
		print(f"{beta_i[i]} = 0 - незначимий")
print("Рівняння регресії без незначимих членів: y = ", end="")
for i in range(len(betas_to_print)):
	print(f" {betas_to_print[i]:+.3f}*{x_i_names[i]}", end="")

#критерій Фішера
d = len(list(filter(None, importance)))
y_table = y_arr
f3 = (m - 1) * N
f4 = N - d
q = 1-p

theor_y = calculate_theoretical_y(natur_table, beta_coef, importance)
y_averages = np.array(list(map(np.average, y_table)))
s_ad = m/(N-d)*(sum((theor_y-y_averages)**2))
y_variations = np.array(list(map(np.var, y_table)))
s_v = np.average(y_variations)
f_p = float(s_ad/s_v)
f_t = get_fisher_value(f3, f4, q)

print("\n\nПеревірка адекватності моделі за критерієм Фішера:",\
	"\nТеоретичні значення y для різних комбінацій факторів:")

for i in range(len(natur_table)):
	print(f"x1 = {natur_table[i][1]:>6.3f}; x2 = {natur_table[i][2]:>6.3f}; "
		f"x3 = {natur_table[i][3]:>7.3f}; y = {theor_y[i]:>8.3f}")
print(f"\nFp = {f_p:.3f}, Ft = {f_t:.3f}","\nFp < Ft => модель адекватна" if f_p < f_t else \
	"\nFp > Ft => модель неадекватна")
