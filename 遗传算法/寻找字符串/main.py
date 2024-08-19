# 代码说明
# 随机字符串生成 (random_string 函数)：生成与目标字符串长度相同的随机字符串。
# 适应度计算 (fitness 函数)：计算一个个体与目标字符串的匹配度，匹配的字符数越多，适应度越高。
# 变异操作 (mutate 函数)：对字符串中的字符进行随机变异，以增加种群的多样性。
# 交叉操作 (crossover 函数)：通过交换父代的一部分生成新个体。
# 选择操作 (select 函数)：根据适应度选择种群中的优秀个体。
# 遗传算法主函数 (genetic_algorithm 函数)：控制遗传算法的整体流程，包括初始化种群、评估适应度、选择、交叉、变异、创建新种群和终止条件。
import random

# 目标字符串
TARGET_STRING = "HELLO WORLD"

# 遗传算法参数
POPULATION_SIZE = 100  # 种群大小
MUTATION_RATE = 0.01  # 变异率
CROSSOVER_RATE = 0.7  # 交叉率
GENERATIONS = 1000  # 最大代数


def random_string(length):
    """生成随机字符串"""
    return ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ ') for _ in range(length))


def fitness(individual):
    """计算适应度，目标字符串越接近，适应度越高"""
    return sum(1 for expected, actual in zip(TARGET_STRING, individual) if expected == actual)


def mutate(individual):
    """对个体进行变异"""
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
    return ''.join(individual)


def crossover(parent1, parent2):
    """交叉操作生成新个体"""
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]
    else:
        return parent1


def select(population):
    """选择操作，选择适应度高的个体"""
    return sorted(population, key=fitness, reverse=True)


def genetic_algorithm():
    # 初始化种群
    population = [random_string(len(TARGET_STRING)) for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        # 评估适应度
        population = select(population)

        # 如果找到了满意解，则退出
        if fitness(population[0]) == len(TARGET_STRING):
            return population[0], generation

        # 创建新种群
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.choices(population[:50], k=2)  # 选择适应度高的个体作为父代
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # 结果返回
    return select(population)[0], GENERATIONS


# 运行遗传算法
result, generations = genetic_algorithm()
print(f"找到的字符串: {result}")
print(f"经过的代数: {generations}")

