from deap import base, creator, tools
import random
import numpy
import os
import image_test
import elitism_callback
import matplotlib.pyplot as plt
import seaborn as sns

POLYGON_SIZE = 3  # количество вершин многоугольника (треугольник)
NUM_OF_POLYGONS = 100  # количество многоугольников

# Общее количество параметров в хромосоме
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)

# ПАРАМЕТРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА
POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.5
MAX_GENERATIONS = 5000
HALL_OF_FAME_SIZE = 20
CROWDING_FACTOR = 10.0

# Фиксируем seed для воспроизводимости
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

imageTest = image_test.ImageTest("images/Mona_Lisa_head.png", POLYGON_SIZE)
BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0
toolbox = base.Toolbox()
# Минимизируем значение fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Индивид — список вещественных чисел
creator.create("Individual", list, fitness=creator.FitnessMin)


def randomFloat(low, up):
    """Создаёт список случайных чисел в диапазоне [low, up]"""
    return [random.uniform(low, up) for _ in range(NUM_OF_PARAMS)]


toolbox.register("attrFloat", randomFloat, BOUNDS_LOW, BOUNDS_HIGH)

toolbox.register(
    "individualCreator",
    tools.initIterate,
    creator.Individual,
    toolbox.attrFloat
)

toolbox.register(
    "populationCreator",
    tools.initRepeat,
    list,
    toolbox.individualCreator
)


# FITNESS-ФУНКЦИЯ

def getDiff(individual):
    """Возвращает ошибку между изображениями (MSE)"""
    return imageTest.getDifference(individual, "MSE"),


toolbox.register("evaluate", getDiff)

# ГЕНЕТИЧЕСКИЕ ОПЕРАТОРЫ


toolbox.register("select", tools.selTournament, tournsize=2)

toolbox.register(
    "mate",
    tools.cxSimulatedBinaryBounded,
    low=BOUNDS_LOW,
    up=BOUNDS_HIGH,
    eta=CROWDING_FACTOR
)

toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=BOUNDS_LOW,
    up=BOUNDS_HIGH,
    eta=CROWDING_FACTOR,
    indpb=1.0 / NUM_OF_PARAMS
)


# CALLBACK: СОХРАНЕНИЕ ИЗОБРАЖЕНИЯ
def saveImage(gen, polygonData):
    """Сохраняет изображение каждые 100 поколений"""
    if gen % 100 == 0:
        folder = f"images/results/run-{POLYGON_SIZE}-{NUM_OF_POLYGONS}"
        os.makedirs(folder, exist_ok=True)
        imageTest.saveImage(polygonData, f"{folder}/after-{gen}-gen.png", f"After {gen} Generations")


# ОСНОВНОЙ ЦИКЛ
def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    population, logbook = elitism_callback.eaSimpleWithElitismAndCallback(
        population,
        toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=MAX_GENERATIONS,
        callback=saveImage,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    best = hof.items[0]
    print("Best fitness:", best.fitness.values[0])

    imageTest.plotImages(imageTest.polygonDataToImage(best))

    minFit, avgFit = logbook.select("min", "avg")

    sns.set_style("whitegrid")
    plt.plot(minFit, label="Min")
    plt.plot(avgFit, label="Avg")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
