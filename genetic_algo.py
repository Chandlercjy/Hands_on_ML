import numpy as np


class GeneticAlgorithm(object):
    """遗传算法.

    Parameters:
    -----------
    cross_rate: float
        交配的可能性大小.
    mutate_rate: float
        基因突变的可能性大小.
    n_population: int
        种群的大小.
    n_iterations: int
        迭代次数.
    password: str
        欲破解的密码.
    """

    def __init__(self, cross_rate: float, mutation_rate: float, n_population: int, n_iterations: int, password: str):
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.n_population = n_population
        self.n_iterations = n_iterations
        self.password = password                                            # 要破解的密码
        # 要破解密码的长度
        self.password_size = len(self.password)
        self.password_ascii = np.fromstring(
            self.password, dtype=np.uint8)  # 将password转换成ASCII
        self.ascii_bounder = [32, 126+1]

    # 初始化一个种群
    def init_population(self):
        population = np.random.randint(low=self.ascii_bounder[0], high=self.ascii_bounder[1],
                                       size=(self.n_population, self.password_size)).astype(np.int8)

        return population

    # 将个体的DNA转换成ASCII
    def translateDNA(self, DNA):                 # convert to readable string
        return DNA.tostring().decode('ascii')

    # 计算种群中每个个体的适应度，适应度越高，说明该个体的基因越好
    def fitness(self, population):
        match_num = (population == self.password_ascii).sum(axis=1)

        return match_num

    # 对种群按照其适应度进行采样，这样适应度高的个体就会以更高的概率被选择
    def select(self, population):
        # add a small amount to avoid all zero fitness
        fitness = self.fitness(population) + 1e-4
        idx = np.random.choice(np.arange(
            self.n_population), size=self.n_population, replace=True, p=fitness/fitness.sum())

        return population[idx]

    # 进行交配
    def create_child(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            # select another individual from pop
            index = np.random.randint(0, self.n_population, size=1)
            cross_points = np.random.randint(0, 2, self.password_size).astype(
                np.bool)   # choose crossover points
            # mating and produce one child
            parent[cross_points] = pop[index, cross_points]
            #child = parent

        return parent

    # 基因突变
    def mutate_child(self, child):
        for point in range(self.password_size):
            if np.random.rand() < self.mutate_rate:
                child[point] = np.random.randint(
                    *self.ascii_bounder)  # choose a random ASCII index

        return child

    # 进化
    def evolution(self):
        population = self.init_population()  # 初始化随机数，范围为ascii编码的长度

        for i in range(self.n_iterations):
            # 通过(a==b)得范围含bool的数组，表示有多少个字相同，全部相加作为fitness
            fitness = self.fitness(population)

            best_person = population[np.argmax(fitness)]
            best_person_ascii = self.translateDNA(best_person)

            if i % 10 == 0:
                print(u'第%-4d次进化后, 基因最好的个体(与欲破解的密码最接近)是: \t %s' %
                      (i, best_person_ascii))

            if best_person_ascii == self.password:
                print(u'第%-4d次进化后, 找到了密码: \t %s' % (i, best_person_ascii))

                break

            population = self.select(population)  # 有放回选取，概率为fitness越大，概率越大
            population_copy = population.copy()

            for parent in population:  # 全部都生成children
                # 在总体中抽两条合成一个children,基因重组，并将替换到parent
                child = self.create_child(parent, population_copy)
                child = self.mutate_child(child)  # 基因突变
                parent[:] = child


def main():
    password = 'I love you!'     # 要破解的密码

    ga = GeneticAlgorithm(cross_rate=0.8, mutation_rate=0.01,
                          n_population=300, n_iterations=500, password=password)

    ga.evolution()


if __name__ == '__main__':
    main()
