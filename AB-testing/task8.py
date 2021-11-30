"""Домашнее задание.

Нужно написать класс TestChecker для оценки эксперимента, данные из которого поступают последовательно.

Последовательно будут подаваться по одному значению из контрольной и экспериментальной групп в метод add_data,
который может принять решение о остановке эксперимента или о необходимости дополнительных данных.
Метод make_decision будет вызван когда дополнительных данных получить больше нельзя и нужно принять решение.

ДАННЫЕ
В данных может быть некоторая периодичность, длина периода заранее известна и передаётся в класс при инициализации. 

ОЦЕНКА КАЧЕСТВА
Будем считать, что за каждый найденный реальный эффект мы получаем 1 единицу денег (TP),
а за каждое неверно внедрённое изменение теряем 1 единицу денег (FP).
Кроме того, каждая дополнительная пара наблюдений стоит 0.002 единицы денег, обозначим количество пар наблюдений L.
Всего будет проведено N аа и аб экспериментов, значит максимальный "выигрыш" равен N.
Далее посчитаем долю полученного выигрыша от максимального: P = (TP - FP - L / 500) / N.
Оценка будет определяться по формуле:
score_ = int(np.ceil((P - 0.07) / 0.03))
score = np.clip(score_, 0, 10)

Обратите внимание на скорость работы функции. В коде есть ограничение по времени.
Чтобы все тесты успели пройти проверку, нужно чтобы 1000 ААБ экспериментов оценивалось не более 1 минуты.
Скорость работы можно проверить в Colab'е https://colab.research.google.com .

Проверка будет осуществляться по 10_000 ААБ экспериментам аналогичным кодом.
Написанный класс должен быть "чистым", в нём нельзя накапливать и использовать данные о прошедших экспериментах.

БОНУСНЫЕ БАЛЛЫ
Топ-5 участников с лучшими результатами получат бонусные баллы.
1 место - 3 балла
2 место - 2 балла
3-5 места - 1 балл
"""

import time
from tqdm import tqdm

import numpy as np
from scipy import stats


EFFECT = 3
SAMPLE_SIZE = 210
MEAN = 100
STD = 10
alpha = 0.05
beta = 0.2
A = beta / (1 - alpha)
B = (1 - beta) / alpha


def get_data(effect, sample_size, period):
    """Возвращает данные эксперимента.

    return: control, pilot
    """
    control = np.random.normal(100, 10, sample_size)
    pilot = np.random.normal(100 + effect, 10, sample_size)
    list_data = [control, pilot]
    for idx, data in enumerate(list_data):
        sort_type = np.random.choice(['asc', 'desc', 'none'])
        if sort_type == 'none':
            continue
        matrix = data.reshape(sample_size // period, period)
        matrix.sort(axis=1)
        if sort_type == 'desc':
            matrix = np.flip(matrix, axis=1)
        sort_data = matrix.ravel()
        shift = np.random.randint(0, period)
        shift_data = np.hstack((sort_data[shift:], sort_data[:shift],))
        shuffle_indexes = np.random.randint(0, sample_size, (5, 2,))
        for i, j in shuffle_indexes:
            shift_data[i], shift_data[j] = shift_data[j], shift_data[i]
        list_data[idx] = shift_data
    return list_data

def pdf_one(x):
    """Функция плотности разницы средних при верности нулевой гипотезы."""
    return stats.norm.pdf(x, 0, np.sqrt(2) * STD)

def pdf_two(x):
    """Функция плотности разницы средних при верности альтернативной гипотезы."""
    return stats.norm.pdf(x, MEAN * (EFFECT/100), np.sqrt(2) * STD)

def test_sequential_wald(data_one, data_two, pdf_one, pdf_two, alpha, beta):
    """Последовательно проверяет отличие по мере поступления данных.
    
    pdf_one, pdf_two - функции плотности распределения при нулевой и альтернативной гипотезах
    
    Возвращает 1, если были найдены значимые отличия, иначе - 0. И кол-во объектов при принятии решения.
    """
    lower_bound = np.log(beta / (1 - alpha))
    upper_bound = np.log((1 - beta) / alpha)
    
    min_len = min([len(data_one), len(data_two)])
    data_one = data_one[:min_len]
    data_two = data_two[:min_len]
    delta_data = data_two - data_one
    
    pdf_one_values = pdf_one(delta_data)
    pdf_two_values = pdf_two(delta_data)
    
    z = np.cumsum(np.log(pdf_two_values / pdf_one_values))
    
    indexes_lower = np.arange(min_len)[z < lower_bound]
    indexes_upper = np.arange(min_len)[z > upper_bound]
    first_index_lower = indexes_lower[0] if len(indexes_lower) > 0 else min_len + 1
    first_index_upper = indexes_upper[0] if len(indexes_upper) > 0 else min_len + 1
    
    if first_index_lower < first_index_upper:
        return 0
    elif first_index_lower > first_index_upper:
        return 1
    else:
        return 0.5


class TestChecker:
    def __init__(self, period: int):
        """Класс для проверки гипотез при постепенном получении данных.

        period - в данных возможна периодичность с таким периодом.
        """
        self.period = period
        self.data_one = []
        self.data_two = []


    def add_data(self, control_value: float, pilot_value: float) -> float:
        """Принимает решение при добавлении данных.
        
        control_value, pilot_value - значения метрики новых измерений
        
        return: принимаем решение или продолжаем эксперимент
            0 - говорим, что эффекта нет
            0.5 - данных недостаточно для принятия решения
            1 - говорим, что эффект есть
        """

        self.data_one.append(control_value)
        self.data_two.append(pilot_value)
        result = test_sequential_wald(np.array(self.data_one), np.array(self.data_two), pdf_one, pdf_two, alpha, beta)

        return result

    def make_decision(self) -> float:
        """Принимает решение по имеющимся данным.

        return: принимем решение
            0 - говорим, что эффекта нет
            1 - говорим, что эффект есть
        """
        _, pvalue = stats.ttest_ind(self.data_one, self.data_two)
        if pvalue < alpha:
          return 1
        else:
          return 0


def run_experiment(control, pilot, period):
    count_step = 0
    test_checker = TestChecker(period)
    for control_value, pilot_value in zip(control, pilot):
        count_step += 1
        res = test_checker.add_data(control_value, pilot_value)
        if res in [0, 1]:
            return res, count_step
    return test_checker.make_decision(), count_step
        


if __name__ == "__main__":
    n_iter = 10000
    max_time = 60 * n_iter / 1000
    count_tp = 0
    count_fp = 0
    list_steps = []
    array_periods = np.random.choice([5, 7, 10, 14, 15, 21], n_iter)
    t1 = time.time()
    for idx, period in tqdm(enumerate(array_periods)):
        a_one, a_two = get_data(0, SAMPLE_SIZE, period)
        res_aa, steps = run_experiment(a_one, a_two, period)
        assert res_aa in [0, 1], 'res_aa not in [0, 1]'
        count_fp += res_aa == 1
        list_steps.append(steps)

        a, b = get_data(EFFECT, SAMPLE_SIZE, period)
        res_ab, steps = run_experiment(a, b, period)
        assert res_ab in [0, 1], 'res_ab not in [0, 1]'
        count_tp += res_ab == 1
        list_steps.append(steps)

        t2 = time.time()
        if t2 - t1 > max_time:
            print('Долго считает! На 1000 ААБ экспериментов более 1 минут.')
            print(f'Успел {idx} из {n_iter}.')
            break
    else:
        print(f'Время на оценку 1000 ААБ экспериментов: {(t2 - t1) / n_iter * 1000:0.2f} сек')
    your_money = count_tp - count_fp - np.sum(list_steps) / 500
    max_money = n_iter
    part_money = your_money / max_money
    score_ = int(np.ceil((part_money - 0.07) / 0.03))
    score = np.clip(score_, 0, 10)
    print(f'part_money = {part_money}')
    print(f'score = {score}')
