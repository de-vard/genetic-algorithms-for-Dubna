from deap import base
from deap import creator
from deap import tools

import random
import numpy
import os

import image_test
import elitism_callback

import matplotlib.pyplot as plt
import seaborn as sns

# константы, связанные с задачей
POLYGON_SIZE = 3
NUM_OF_POLYGONS = 100

# вычисляем общее количество параметров в хромосоме:
# для каждого полигона есть:
# две координаты на вершину, 3 значения цвета, одно значение прозрачности (alpha)
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)

# константы генетического алгоритма:
POPULATION_SIZE = 200
P_CROSSOVER = 0.9  # вероятность кроссовера
P_MUTATION = 0.5  # вероятность мутации индивида
MAX_GENERATIONS = 5000
HALL_OF_FAME_SIZE = 20
CROWDING_FACTOR = 10.0  # фактор скученности для кроссовера и мутации

# установка seed для генератора случайных чисел:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# создаём экземпляр класса тестирования изображения:
imageTest = image_test.ImageTest("images/Mona_Lisa_head.png", POLYGON_SIZE)

# вычисляем общее количество параметров в хромосоме:
# для каждого полигона есть:
# две координаты на вершину, 3 значения цвета, одно значение прозрачности (alpha)
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)

# все значения параметров находятся в диапазоне от 0 до 1,
# позже они будут масштабированы
BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0  # границы для всех измерений

toolbox = base.Toolbox()

# определяем одну целевую функцию — минимизация приспособленности
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# создаём класс Individual на основе списка
creator.create("Individual", list, fitness=creator.FitnessMin)


# вспомогательная функция для создания случайных вещественных чисел,
# равномерно распределённых в диапазоне [low, up]
# предполагается, что диапазон одинаков для всех измерений
def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * NUM_OF_PARAMS, [up] * NUM_OF_PARAMS)]


# создаём оператор, который случайно возвращает вещественные числа в заданном диапазоне
toolbox.register("attrFloat", randomFloat, BOUNDS_LOW, BOUNDS_HIGH)

# создаём оператор, который заполняет экземпляр Individual
toolbox.register("individualCreator",
                 tools.initIterate,
                 creator.Individual,
                 toolbox.attrFloat)

# создаём оператор, который генерирует список индивидов
toolbox.register("populationCreator",
                 tools.initRepeat,
                 list,
                 toolbox.individualCreator)


# вычисление приспособленности с использованием MSE в качестве метрики различия
def getDiff(individual):
    return imageTest.getDifference(individual, "MSE"),
    # return imageTest.getDifference(individual, "SSIM"),


toolbox.register("evaluate", getDiff)

# генетические операторы:
toolbox.register("select", tools.selTournament, tournsize=2)

toolbox.register("mate",
                 tools.cxSimulatedBinaryBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR)

toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR,
                 indpb=1.0 / NUM_OF_PARAMS)


# сохранение лучшего текущего изображения каждые 100 поколений (используется как callback)
def saveImage(gen, polygonData):
    # только каждые 100 поколений
    if gen % 100 == 0:

        # создаём папку, если она не существует
        folder = "images/results/run-{}-{}".format(POLYGON_SIZE, NUM_OF_POLYGONS)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # сохраняем изображение в папку
        imageTest.saveImage(polygonData,
                            "{}/after-{}-gen.png".format(folder, gen),
                            "After {} Generations".format(gen))


# основной цикл генетического алгоритма
def main():
    # создаём начальную популяцию (поколение 0)
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # подготавливаем объект статистики
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)

    # определяем объект зала славы (hall of fame)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # выполняем генетический алгоритм с элитизмом и callback-функцией saveImage
    population, logbook = elitism_callback.eaSimpleWithElitismAndCallback(population,
                                                                          toolbox,
                                                                          cxpb=P_CROSSOVER,
                                                                          mutpb=P_MUTATION,
                                                                          ngen=MAX_GENERATIONS,
                                                                          callback=saveImage,
                                                                          stats=stats,
                                                                          halloffame=hof,
                                                                          verbose=True)

    # выводим лучшего найденного индивида
    best = hof.items[0]
    print()
    print("Best Solution = ", best)
    print("Best Score = ", best.fitness.values[0])
    print()

    # отображаем лучшее изображение рядом с эталонным
    imageTest.plotImages(imageTest.polygonDataToImage(best))

    # извлекаем статистику
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # строим графики статистики
    sns.set_style("whitegrid")
    plt.figure("Stats:")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # показываем оба графика
    plt.show()


if __name__ == "__main__":
    main()
