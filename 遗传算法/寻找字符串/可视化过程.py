# 代码说明
# 适应度记录 (best_fitnesses 和 average_fitnesses 列表)：在每一代中记录种群中最佳个体和平均个体的适应度。
# 可视化：
# 适应度变化曲线：绘制每一代的最佳适应度和平均适应度的变化趋势。
# 最终种群适应度分布：展示最终种群中个体适应度的频率分布。
# 运行步骤
# 安装 matplotlib 库：pip install matplotlib
# 运行代码后，生成两幅图：一幅展示适应度变化曲线，另一幅展示最终种群的适应度分布。
import random
import matplotlib.pyplot as plt

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

    # 记录适应度变化
    best_fitnesses = []
    average_fitnesses = []

    for generation in range(GENERATIONS):
        # 评估适应度
        population = select(population)

        # 记录适应度
        best_fitness = fitness(population[0])
        average_fitness = sum(fitness(ind) for ind in population) / len(population)
        best_fitnesses.append(best_fitness)
        average_fitnesses.append(average_fitness)

        # 如果找到了满意解，则退出
        if best_fitness == len(TARGET_STRING):
            return population[0], generation, best_fitnesses, average_fitnesses

        # 创建新种群
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.choices(population[:50], k=2)  # 选择适应度高的个体作为父代
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # 结果返回
    return select(population)[0], GENERATIONS, best_fitnesses, average_fitnesses


# 运行遗传算法
result, generations, best_fitnesses, average_fitnesses = genetic_algorithm()

# 可视化
plt.figure(figsize=(14, 7))

# 绘制适应度变化曲线
plt.subplot(1, 2, 1)
plt.plot(best_fitnesses, label='Best Fitness')
plt.plot(average_fitnesses, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Evolution')
plt.legend()

# 绘制最终种群的分布
plt.subplot(1, 2, 2)
plt.hist([fitness(ind) for ind in select([random_string(len(TARGET_STRING)) for _ in range(POPULATION_SIZE)])],
         bins=range(len(TARGET_STRING) + 1), edgecolor='black')
plt.xlabel('Fitness')
plt.ylabel('Frequency')
plt.title('Final Population Fitness Distribution')

plt.tight_layout()
plt.show()

print(f"找到的字符串: {result}")
print(f"经过的代数: {generations}")
