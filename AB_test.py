
import pandas as pd
import numpy as np
import math
from datetime import datetime

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Plan of the experiment
# What type of metric?

# metric_type = input(f'Do we use RATIO METRIC (Yes/No):')
#
# if metric_type == 'Yes':
#     # BOOTSTRAP
#     bootstrap = input(f'Do we use BOOTSTRAP? (Yes/No):')
#     if bootstrap == 'No':
#         # DELTA-METHOD
#         delta_method = input(f'Do we use DELTA-METHOD? (Yes/No):')
#         if delta_method == 'No':
#             # LINEARISATION
#             linearisation = input(f'Do we use LINEARISATION? (Yes/No):')
# #### Строим графики, чтобы визуально оценить выборку (функции box_plot(df), histogram(df), scatter_plot(df))
#             # OUTLIERS
#             outliers = input(f'Do we use OUTLIERS? (Yes/No):')
#             #STRATIFICATION
#             stratification = input(f'Do we use STRATIFICATION? (Yes/No):')
#             # CUPED
#             cuped = input(f'Do we use CUPED? (Yes/No):')
# else:
# #### Строим графики, чтобы визуально оценить выборку (функции box_plot(df), histogram(df), scatter_plot(df)).
# # Если мы хотим провести t - тест для метрики среднего, то необходимо, чтобы выборка была достаточного размера и не сильно
# # скошена. Есть смысл смотреть не только на распределение выборки, но и с помощью бутстрепа посмотреть на распределение среднего
# # выборки. Если среднее распределно нормально, значит ЦПТ работает.
# # В случае метрик не среднего, а, например, квантилей использовать бутстреп
#     # BOOTSTRAP
#     bootstrap = input(f'Do we use BOOTSTRAP? (Yes/No):')
#     if bootstrap == 'No':
#         # OUTLIERS (выбросы несвязанные с экспериментом)
#         outliers = input(f'Do we use outliers? (Yes/No):')
#         # STRATIFICATION (лучше всего использовать когда есть большая разница между стратами)
#         stratification = input(f'Do we use STRATIFICATION? (Yes/No):')
#         # CUPED (есть прошлые данные)
#         cuped = input(f'Do we use CUPED? (Yes/No):')

# Error Levels and Expected Effect:
#
# Define the acceptable Type I error rate (alpha).
alpha = 0.05
#
# Define the acceptable Type II error rate (beta) or power (1-beta).
beta = 0.2
#
# Define the Minimum Detectable Effect in % (MDE)
mde = 5

#Outlier treatment methods.
#Выбросы. Перед началом эксперимента на исторических данных удаляем выбросы и оцениваем дисперсию данных для дальнейшей
#оценки размера групп под эксперимент.

#Визуальный анализ данных:
def box_plot(df: pd.Series) -> None:
    sns.boxplot(df)
    plt.title('Boxplot for historical data')
    plt.show()

def histogram(df: pd.Series) -> None:
    sns.histplot(df, bins=30, kde=True, color='orange')
    plt.title('Histogram for historical data')
    plt.show()

def scatter_plot(df: pd.Series) -> None:
    sns.scatterplot(df, alpha=0.6)
    plt.title('Scatter plot for historical data')
    plt.show()

def QQ_plot(df: pd.Series) -> None:
    data = data.dropna()
    stats.probplot(df, dist="norm", plot=plt)
    plt.title("Q–Q Plot against Normal distribution")
    plt.xlabel("Теоретические квантили (N(0,1))")
    plt.ylabel("Наблюдаемые квантили")
    plt.grid(True)
    plt.show()

#Удаляем выбросы, если это возможно. Будьте внимательны с удалением выбросов. Слишком большие или маленькие значения
# не всегда являются выбросами. Нужно исходить из логики данных.
# И нужно быть внимательными, чтобы выбросы не были связаны с проводимым экспериментом:

# 1) IQR Удаляем все, что вне усов на Box plot. x < Q1 - 1.5 * IQR или x > Q3 + 1.5 * IQR
def Outlier_IQR(df: pd.DataFrame) -> pd.DataFrame:
    q1 = np.quantile(df, 0.25)
    q3 = np.quantile(df, 0.75)
    iqr = q3 - q1
    return df.loc[(df['metric'] >= q1 - 1.5 * iqr) & (df['metric'] <= q3 + 1.5 * iqr)]

# 2) Logical. Удаляем выбросы исходя из логики данных.
def process_outliers(df: pd.DataFrame, bounds: tuple[float, float], outlier_process_type: str) -> pd.DataFrame:
    """Возвращает новый датафрейм с обработанными выбросами в измерениях метрики.

    :param df (pd.DataFrame): таблица со значениями метрики
        со столбцами ['user_id', 'metric'].
    :param bounds (tuple[float, float]): нижняя и верхняя границы метрики. Всё что
        не попало между ними считаем выбросами.
    :param outlier_process_type (str): способ обработки выбросов. Возможные варианты:
        'drop' - удаляем измерение,
        'clip' - заменяем выброс на значение ближайшей границы (lower_bound, upper_bound).
    :return df: таблица со столбцами ['user_id', 'metric']
    """
    if outlier_process_type == 'drop':
        return df[(df['metric'] >= bounds[0]) & (df['metric'] <= bounds[1])]
    if outlier_process_type == 'clip':
        df['metric'] = df['metric'].clip(lower=bounds[0], upper=bounds[1])
        return df


#CUPED. При наличии исторических данных. Расчет идет на всей выборке.Используем для уменьшения дисперсии выборки. А дальше
# в зависимости от метрики используем нужный статистический критерий
#В случае, если мы хотим оценить sample size то нам понадобятся пред-предэкспериментальные данные и предэкспериментальные
def calculate_cuped_metric(df_metric: pd.DataFrame, df_cov: pd.DataFrame) -> pd.DataFrame:
    """Считает значения cuped-метрики. Расчет ведется на всех данных: контрольной и экспериментальной групп.
    Мы считаем, что влияние эксперимента на ковариацию незначительно. В более строгом случае можно рассчитывать параметр
    theta только на данных из контрольной группы

    :param df_metric (pd.DataFrame): таблица со значениями метрики во время эксперимента
        со столбцами ['user_id', 'metric'].
    :param df_cov (pd.DataFrame): таблица со значениями ковариаты
        со столбцами ['user_id', 'cov'].
    :return df: таблица со значениями cuped-метрики со столбцами ['user_id', 'metric'].
    """

    theta = np.cov(df_metric['metric'], df_cov['cov'])[0, 1] / np.var(df_cov['cov'])
    df = df_metric.merge(df_cov, how='left', on='user_id')
    df['metric'] = df['metric'] - theta * (df['cov'] - df['cov'].mean())
    return df[['user_id', 'metric']]


#Stratification. При наличии соотвествующей сплит системы.
# По историческим данным мы должны определить доли выбранных нами страт в генеральной совокупности;
# Для каждой из страт определить дисперсию нашей метрики. Нужно посчитать дисперсию по наблюдениям всех объектов из каждой страты отдельно;
# Оценить дисперсию пилотной и контрольной группы при стратифицированном семплировании по формуле.
# Посчитать необходимый размер групп, подставив полученную оценку дисперсии.

# Веса страт на исторических данных. ['strat'] - поле с значениями признака для стратификации
def stratified_weights(df: pd.DataFrame) -> pd.DataFrame:
    """param df(pd.DataFrame): ['user_id', 'metric', 'strat'] """
    weights = df['strat'].value_counts(normalize=True).reset_index()
    weights.columns = ['strat', 'weight']
    return weights


# МЕТРИКИ ОТНОШЕНИЯ

# Delta-method. Дисперсия метрики отношения
# Когда Дельта-метод работает хорошо:
# Большой размер выборки: Чем больше N, тем лучше.
# Незначительная нелинейность функции g() в диапазоне типичных значений X
# Базовые величины не сильно скошены (или N очень велико).
# Известны (или хорошо оцениваются) дисперсии и ковариации базовых величин.
def var_delta_method(df: pd.DataFrame) -> float:
    """param df(pd.DataFrame): ['user_id', 'metric_x', 'metric_y'] """
    df = df.groupby('user_id').agg(x=('metric_x', 'sum'), y=('metric_y', 'sum'))
    mean_x = df['x'].mean()
    mean_y = df['y'].mean()
    var_x = df['x'].var()
    var_y = df['y'].var()
    cov_xy = df['x'].cov(df['y'])
    return 1 / len(df) * (1 / mean_y ** 2 * var_x - 2 * mean_x / mean_y ** 3 * cov_xy + mean_x ** 2 / mean_y ** 4 * var_y)


# Linearisation. Логично использовать, если мы планируем использовать методы снижения дисперсии, например, CUPED.
# 1) Для подготовки эксперимента необходимо необходимо взять данные ковариаты на пред-предэкспериментальном
# периоде и данные метрики на предэкспериментальном
# 2) Выполнить линеаризацию того и того
# 3) Выполнить преобразование CUPED с помощью функции calculate_cuped_metric()
# 4) Оценить дисперсию преобразованной метрики
# 5) Оценить размер групп

# На исторических данных при подготовке эксперимента коэффициент при линеаризации мы рассчитываем на всех данных
# для большей точности

def calculate_linearized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Считает значения линеаризованной метрики.
    Параметр kappa (коэффициент в функции линеаризации) по данным из всех данных
    :param df (pd.DataFrame): датафрейм со значениями метрики.
        Значения в столбце 'user_id' могут быть не уникальны.
        Измерения для одного user_id считаем зависимыми, а разных user_id - независимыми.
        columns=['user_id', 'metric']
    :return lin_metrics: датафрейм со значениями линеаризованной метрики.
        columns=['user_id', 'metric']
    """
    df = df.groupby('user_id').agg(old_metric=('metric', 'sum'), num=('user_id', 'count')).reset_index()
    kappa = df['old_metric'].sum() / df['num'].sum()
    df['metric'] = df['old_metric'] - kappa * df['num']
    return df[['user_id', 'metric']]


#########################################
#Считаем дисперсию на исторических данных (используем данные после повышения чувствительности и линеаризации)
#########################################

# Можно на исторических данных оценить размер групп необходимых для проведения эксперимента на заданных уровнях мощности и alpha
# Для оценки дисперсии можно использовать тот же бутстреп на исторических данных или использовать при возможности другие менее
# емкие методы.
# При это важно учитывать ненормальность данных, если данные сильно скошены, в этом случае оценить размер группы лучше
# с помощью проведения синтетических тестов и подгона размера группы, удовлетворяющих требуемой мощности и alpha.
# При несильной скошенности допустимо использовать формулу для оценки размера группы, при этом истиный размер группы
# берем 10-15% больше

#Если данные распределены нормально (почти нормально):
def var_norm(df: pd.DataFrame) -> float:
    return np.var(df['metric'])

# Дисперсия для стратифицированных данных:
# В случай отсутствия возможно провести стратификацию, но при проведелении эксперимента будете использовать постстратификацию,
# то оценивать размер групп нужно по нестратифицированным данным, тогда можно брать консервативную оценку размера групп
def stratified_var(df: pd.DataFrame) -> float:
    """param df(pd.DataFrame): ['user_id', 'metric', 'strat'] """
    weights = stratified_weights(df)
    df_strat = df.groupby('strat')['metric'].var().reset_index().rename(columns={'metric': 'var'})
    df_strat = df_strat.merge(weights, how='inner', on='strat')
    df_strat['weight'] = df_strat['weight'] / df_strat['weight'].sum()
    return (df_strat['var'] * df_strat['weight']).sum()


#############################################
# SAMPLE SIZE. Knowing the error levels, MDE, and variance, we can calculate on historical data: Sample size
# При условии равных дисперсий в группах:
def group_size_equal_var(var: float, alpha: float, beta: float, mde: float) -> int:
    t_alpha = stats.norm.ppf(1 - alpha/2)
    t_beta = stats.norm.ppf(1 - beta)
    return math.ceil(2 * var * (t_alpha + t_beta)** 2 / mde ** 2)

# При условии разных дисперсий в группах:
def group_size(var_a: float, var_b: float, alpha: float, beta: float, mde: float) -> int:
    t_alpha = stats.norm.ppf(1 - alpha/2)
    t_beta = stats.norm.ppf(1 - beta)
    return math.ceil((var_a + var_b) * (t_alpha + t_beta)** 2 / mde ** 2)

#MDE. Knowing the error levels, sample size, and variance, we can calculate on historical data: MDE
def mde(df: pd.Series, alpha: float, beta: float, n: int) -> float:
    t_alpha = stats.norm.ppf(1 - alpha / 2)
    t_beta = stats.norm.ppf(1 - beta)
    sigma = df.var(ddof=0)
    return int(np.sqrt(2 * sigma) * (t_alpha + t_beta) / np.sqrt(n/2))

##################################
# В случае сильно скошенных данных или метрик (квантилиб медианы и т.п.) оценить размер групп для эксперимента можно с помощью бутстрепа.

#Нормальный доверительный интервал для бутстреп выборки
def bootstrap_norm_ci(point_estimate: float, boots_metrics: np.array, alpha: float=0.05) -> tuple:
    f_alpha = stats.norm.ppf(1 - alpha / 2)
    std = np.std(boots_metrics)
    left = point_estimate - f_alpha * std
    right = point_estimate + f_alpha * std
    return left, right

#Процентильный доверительный интервал для бутстреп выборки
def bootstrap_percentile_ci(boots_metrics: np.array, alpha:float=0.05) -> tuple:
    left = np.quantile(boots_metrics, alpha / 2)
    right = np.quantile(boots_metrics, 1 - alpha / 2)
    return left, right

#Центральный доверительный интервал для бутстреп выборки
def bootstrap_central_ci(point_estimate: float, boots_metrics: np.array, alpha: float=0.05) -> tuple:
    left = 2 * point_estimate - np.quantile(boots_metrics, 1 - alpha / 2)
    right = 2 * point_estimate - np.quantile(boots_metrics, alpha / 2)
    return left, right

# Бутстреп для квантиля. Доверительный интервал оценивается как персентиль
def bootstrap(df_control: np.array, df_test: np.array, quantile: float, iter=2000) -> tuple:
    metric = np.quantile(df_test, quantile) - np.quantile(df_control, quantile)
    boots_metric = []
    for _ in range(iter):
        a = np.random.choice(df_control, len(df_control), replace=True)
        b = np.random.choice(df_test, len(df_test), replace=True)
        boots_metric.append(np.quantile(b, quantile) - np.quantile(a, quantile))
    boots_metric = np.array(boots_metric)
    left, right = bootstrap_percentile_ci(boots_metric, alpha=0.05)
    return left, right

# Оценим размер групп для сильно скошенных данных с помощью бутстрепа. Находим ошибки 1-го и 2-го рода.
def group_size_bootstrap(df: pd.DataFrame, effect_add_type='all_const', effect=2, iter=3000) -> dict:
    """df (pd.DataFrame) -  данные ['user_id, 'metric'], в которых user_id уникальное значение;"""
    p_value = {}
    # p_value[0] - значение p_value А/А тест
    # p_value[1] - значение p_value А/В тест
    sample_number = range(200, 5000, 200)
    for n in sample_number:
        definition_aa = 0
        definition_ab = 0
        for _ in range(iter):
            f_a, f_b = np.random.choice(df['metric'], size=(2,n), replace=False)
            left, right = bootstrap(f_a, f_b, quantile=0.9)
            if left > 0 or right < 0:
                definition_aa += 1
            if effect_add_type == 'all_const':
                f_b = f_b + f_b.mean() * effect / 100
            else:
                f_b = f_b * (1 + effect / 100)
            left, right = bootstrap(f_a, f_b, quantile=0.9)
            if (left < 0) and (right > 0):
                definition_ab += 1
        p_value[n] = (definition_aa/iter, definition_ab/iter)
    return p_value



##################################

###############
CHECKING DESIGN
###############
# Для данных (при необходимости преобразованных, с повышением чувствительности) проводим А/А и А/В тесты для групп найденного
# размера. Функция сделана без учета стратификации. В случае, если необходимо проверить дизайн со стратификацией,
# нужно применить стратифицированное семплирование и постстратификацию с помощью функции get_ttest_strat_pvalue.
# В случае delta-method для метрики отношения расчитывем разность метрик отношения для обеих выборок и считаем диспресию
# с помощью функции var_delta_method и рассчитывем p_value. В данной функции это пока не реализовано.

def estimate_errors(df: pd.DataFrame, effect: float, n: int, effect_add_type: str ='all_const', alpha: float=0.05, iter: int = 5000):
    """df (pd.DataFrame) -  данные ['user_id, 'metric'], в которых user_id уникальное значение;

       effect_add_type (str): способ добавления эффекта для группы B.
       - 'all_const' - увеличить всем значениям в группе B на константу (b_metric_values.mean() * effect / 100).
       - 'all_percent' - увеличить всем значениям в группе B в (1 + effect / 100) раз.

       effect (float): размер эффекта в процентах.
       Пример, effect=3 означает, что ожидаем увеличение среднего на 3%.

       n (int): размер групп посчитанный ранее

       alpha (float): уровень значимости.

       iter (int): количество иттераций

       :return pvalues_aa (list[float]), pvalues_ab (list[float]), first_type_error (float), second_type_error (float):
        - pvalues_aa, pvalues_ab - списки со значениями pvalue
        - first_type_error, second_type_error - оценки вероятностей ошибок I и II рода."""

    assert 2 * n <= len(df), "Размер выборки слишком большой для заданного df"
    p_value = [[],[]]
    #p_value[0] - значения p_value А/А тест
    #p_value[1] - значения p_value А/В тест
    for _ in range(iter):
        a, b = np.random.choice(df['metric'], size = (2, n), replace = False)
        p_value[0].append(stats.ttest_ind(a,b).pvalue)
        if effect_add_type == 'all_const':
            b = b + df['metric'].mean() * effect / 100
        else:
            b = b * (1 + effect / 100)
        p_value[1].append(stats.ttest_ind(a,b).pvalue)
    return p_value[0], p_value[1], np.mean(np.array(p_value[0]) < alpha), np.mean(np.array(p_value[1]) > alpha)



#С помощью функции plot_pvalue_hist_ecdf мы можем посмотреть распределение p-value полученных в estimate_errors:
#График распределения p_value и гистограмма распределения
def plot_pvalue_hist_ecdf(pvalues: np.array) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    sns.histplot(pvalues, ax=ax1, bins=20, stat='density')
    ax1.plot([0, 1], [1, 1], 'k--')
    ax1.set(xlabel='p-value', ylabel='Density')

    sns.ecdfplot(pvalues, ax=ax2)
    ax2.plot([0,0], [1,1], 'k--')
    ax2.set(xlabel='p-value', ylabel='Probability')
    plt.show()

###################
#Проводим эксперимент. Применение выбранного статистического теста (t-тест для средних/линеаризованных метрик, бутстрэп для квантилей/скошенных данных).
#Расчет доверительных интервалов для разницы метрик.
#Расчет p-значений.

###################


# POSTSTRATIFIED Считаем p_value и использованием постстратификации для метрики среднего с использованием t-test.
def get_ttest_strat_pvalue(metrics_strat_a_group: pd.DataFrame, metrics_strat_b_group: pd.DataFrame) -> float:
    """Применяет постстратификацию, возвращает pvalue.

    Веса страт считаем по данным обеих групп.
    Предполагаем, что эксперимент проводится на всей популяции.
    Веса страт нужно считать по данным всей популяции.

    :param metrics_strat_a_group (pd.DataFrame): значения метрик и страт группы A.
        shape = (n, 2), columns=['metrics', 'strats'].
    :param metrics_strat_b_group (pd.DataFrame): значения метрик и страт группы B .
        shape = (n, 2), columns=['metrics', 'strats'].
    :return (float): значение p-value
    """
    # YOUR_CODE_HERE
    metrics = pd.concat([metrics_strat_a_group, metrics_strat_b_group], axis=0)
    weights = metrics['strats'].value_counts(normalize=True)
    metric_a = 0
    metric_b = 0
    var_a = 0
    var_b = 0
    for weight in weights.index:
        metric_a += metrics_strat_a_group.loc[metrics_strat_a_group['strats'] == weight, 'metrics'].mean() * weights[weight]
        var_a += metrics_strat_a_group.loc[metrics_strat_a_group['strats'] == weight, 'metrics'].var() * weights[weight]
        metric_b += metrics_strat_b_group.loc[metrics_strat_b_group['strats'] == weight, 'metrics'].mean() * weights[weight]
        var_b += metrics_strat_b_group.loc[metrics_strat_b_group['strats'] == weight, 'metrics'].var() * weights[weight]
    t_stat = abs(metric_a - metric_b) / (var_a / len(metrics_strat_a_group) + var_b / len(metrics_strat_b_group)) ** 0.5
    p_value = stats.norm.sf(t_stat) * 2
    return p_value

# Тест Манна-Уитни. Не забываем, что группы должны иметь одинаковые формы распределений. Не должно быть много совпадающих значений

def mannwhitney_ab_test(a: np.ndarray, b: np.ndarray, alternative: str = 'two-sided') -> float:
    """
    Проводит A/B тест с помощью критерия Манна–Уитни.

    :param a: Метрики группы A (1D array-like)
    :param b: Метрики группы B (1D array-like)
    :param alternative: Направление теста:
        - 'two-sided' (по умолчанию): различие в любую сторону
        - 'less': гипотеза, что группа A < группы B
        - 'greater': гипотеза, что группа A > группы B

    :return: p-value теста
    """
    stat, p_value = mannwhitneyu(a, b, alternative=alternative)
    return p_value


# Функция стратификации для распределения данных по группам (контрольная и экспериментальная)
# Не стоит делать слишком много групп при стратификации, размеры групп не должны быть слишком маленькими
def split_stratified(df: pd.DataFrame) -> pd.DataFrame:
    """Распределяет объекты по группам (контрольная и экспериментальная).
    :param df(pd.DataFrame): df с разбиением на страты [[user_id, strat]].
    :return groups (pd.DataFrame) [[user_id, strats, pilot]]: массив из 0 и 1,
    pilot:  0 - контрольная группа, 1 - экспериментальная.
    """
    # df = pd.DataFrame(strats, columns=['strat']).reset_index().rename(columns={'index': 'user_id'})
    df['pilot'] = 0
    for s in df['strat'].unique():
        d_a = df.loc[df['strat'] == s, 'user_id']
        df_a = np.random.choice(d_a, np.random.choice([len(d_a) // 2 + len(d_a) % 2, len(d_a) // 2]), replace=False)
        df.loc[df['user_id'].isin(df_a), ['pilot']] = 1
    return df
