"""Домашнее задание №2.

1. Реализовать три функции для построения доверительных интервалов с помощью бутстрепа:
- get_ci_bootstrap_normal
- get_ci_bootstrap_percentile
- get_ci_bootstrap_pivotal

2. Придумать 4 пары наборов данных, для которых разные методы будут давать разные результаты.
Результат записать в словарь datasets, пример словаря приведён ниже в коде.
Примеры нужны только для указанных ключей. Размеры данных должны быть равны 10.
Расшифровка ключей словаря:
- 'normal' - метод проверки значимости отличий с помощью нормального доверительного интервала.
- 'percentile' - метод проверки значимости отличий с помощью ДИ на процентилях.
- 'pivotal' - метод проверки значимости отличий с помощью центрального ДИ.
- '1' - отличия средних значимы, '0' - иначе.
Пример:
'normal_1__percentile_0' - для данных по этому ключу метод проверки значимости отличий
с помощью нормального ДИ показывает наличие значимых отличий, а ДИ на процентилях нет.


За правильную реализацию каждой функции даётся 2 балла.
За каждый правильный пример данных даётся 1 балл.

Правильность работы функций будет проверятся сравнением значений с авторским решением.
Подход для проверки примеров данных реализован ниже в исполняемом коде.
"""


import numpy as np
from scipy import stats


def generate_bootstrap_data(data_one: np.array, data_two: np.array, size=10000):
    """Генерирует значения метрики, полученные с помощью бутстрепа."""
    bootstrap_data_one = np.random.choice(data_one, (len(data_one), size))
    bootstrap_data_two = np.random.choice(data_two, (len(data_two), size))
    res = bootstrap_data_two.mean(axis=0) - bootstrap_data_one.mean(axis=0)
    return res


def get_ci_bootstrap_normal(boot_metrics: np.array, pe_metric: float, alpha: float=0.05):
    """Строит нормальный доверительный интервал.

    boot_metrics - значения метрики, полученные с помощью бутстрепа
    pe_metric - точечная оценка метрики
    alpha - уровень значимости
    
    return: (left, right) - границы доверительного интервала.
    """
    q = stats.norm.ppf(1 - alpha / 2)
    delta = q * boot_metrics.std()
    left = pe_metric - delta
    right = pe_metric + delta
    return (left, right)


def get_ci_bootstrap_percentile(boot_metrics: np.array, pe_metric: float, alpha: float=0.05):
    """Строит доверительный интервал на процентилях.

    boot_metrics - значения метрики, полученные с помощью бутстрепа
    pe_metric - точечная оценка метрики
    alpha - уровень значимости
    
    return: (left, right) - границы доверительного интервала.
    """
    left = np.percentile(boot_metrics, alpha / 2 * 100)
    right = np.percentile(boot_metrics, (1 - alpha / 2) * 100)
    return (left, right)


def get_ci_bootstrap_pivotal(boot_metrics: np.array, pe_metric: float, alpha: float=0.05):
    """Строит центральный доверительный интервал.

    boot_metrics - значения метрики, полученные с помощью бутстрепа
    pe_metric - точечная оценка метрики
    alpha - уровень значимости
    
    return: (left, right) - границы доверительного интервала.
    """
    left = 2 * pe_metric - abs(np.percentile(boot_metrics, (1 - alpha / 2) * 100))
    right = 2 * pe_metric + abs(np.percentile(boot_metrics, alpha / 2 * 100))
    return (left, right)


datasets = {
    'normal_1__percentile_0': [np.random.normal(0, 1, size=10), np.random.normal(0, 2, size=10)], 
    'normal_0__percentile_1': [np.arange(10), np.arange(10)], 
    'percentile_1__pivotal_0': [np.random.normal(size=10), np.random.normal(size=10)], 
    'percentile_0__pivotal_1': [np.arange(10), np.arange(10)], 
}


funcname_to_func = {
    'normal': get_ci_bootstrap_normal,
    'percentile': get_ci_bootstrap_percentile,
    'pivotal': get_ci_bootstrap_pivotal
}


if __name__ == "__main__":
    for data_one, data_two in datasets.values():
        assert len(data_one) == len(data_two) == 10

    print(f'{"dataset_name": <24}|{"funcname": <11}|{"my_res": ^8}|{"res": ^5}|{"verdict": <9}\n', '-'*58)
    for dataset_name, (data_one, data_two) in datasets.items():
        pe_metric = np.mean(data_two) - np.mean(data_one)

        boot_metrics = generate_bootstrap_data(data_one, data_two)
        for funcname_res in dataset_name.split('__'):
            funcname, res = funcname_res.split('_')
            func = funcname_to_func[funcname]
            res = int(res)
            left, right = func(boot_metrics, pe_metric)
            my_res = 1 - int(left <= 0 <= right)
            verdict = 'correct' if res == my_res else 'error !'
            print(f'{dataset_name: <24}|{funcname: <11}|{my_res: ^8}|{res: ^5}|{verdict: <9}')