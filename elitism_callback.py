from deap import tools
from deap import algorithms


def eaSimpleWithElitismAndCallback(population, toolbox, cxpb, mutpb, ngen, callback=None, stats=None, halloffame=None,
                                   verbose=__debug__):
    """
    Эволюционный алгоритм на основе eaSimple (DEAP) с:
    1. Элитизмом — лучшие особи сохраняются без изменений
    2. Callback-функцией — вызывается после каждого поколения
    """

    # Журнал для хранения статистики по поколениям
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Оценка особей без рассчитанной приспособленности
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Проверка наличия Hall of Fame (обязателен для элитизма)
    if halloffame is None:
        raise ValueError("Параметр halloffame не может быть None")

    # Обновление списка лучших особей
    halloffame.update(population)
    hof_size = len(halloffame.items)

    # Запись статистики для начального поколения
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Основной цикл по поколениям
    for gen in range(1, ngen + 1):

        # Отбор потомков (без элитных особей)
        offspring = toolbox.select(
            population,
            len(population) - hof_size
        )

        # Применение кроссовера и мутации
        offspring = algorithms.varAnd(
            offspring, toolbox, cxpb, mutpb
        )

        # Оценка новых особей
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Возврат элитных особей в популяцию
        offspring.extend(halloffame.items)

        # Обновление Hall of Fame
        halloffame.update(offspring)

        # Замена старой популяции новой
        population[:] = offspring

        # Запись статистики
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Вызов callback-функции
        if callback:
            callback(gen, halloffame.items[0])

    return population, logbook
