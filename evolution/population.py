import numpy as np
import os

class Population:
    def __init__(self, size):
        self._size = size
        self._pop = []

    def __len__(self):
        return len(self._pop)

    def __str__(self):
        return f"Population Size: {len(self._pop)} - Best Fitness: {self.get(-1).fitness()}"

    def kill(self, size):
        self._pop = sorted(self._pop, key=lambda x: x.fitness())
        kill_size = min(size,len(self._pop))
        for _ in range(kill_size):
            del self._pop[0]

    def add(self, c):
        if len(self._pop) >= self._size:
            self.kill(1)
        self._pop.append(c)

    def clone(self):
        pop = Population(self._size)
        for c in self._pop:
            pop._pop.append(c.clone())
        return pop

    def select(self, size):
        selected = []
        for i in range(size):
            index = np.random.randint(len(self._pop))
            selected.append(self._pop[index])
        sorted(selected, key=lambda x: x.fitness())
        return selected[-1].clone()

    def get(self, index):
        self._pop = sorted(self._pop, key=lambda x: x.fitness())
        if abs(index) > len(self._pop):
            return None
        return self._pop[index].clone()

    def get_all(self):
        result = []
        for c in self._pop:
            result.append(c.clone())
        return result

    def save(self, folder):
        os.makedirs(folder)
        for i in range(len(self._pop)):
            self.get(i).save(os.path.join(folder, f"{i}.json"))

    def load(self, folder, width, height, init, mutate, fitness, behaviors):
        self._pop = []
        files = [fn for fn in os.listdir(folder) if ".json" in fn]
        for fn in files:
            c = Chromosome(width, height, init, mutate, fitness, behaviors)
            c.load(os.path.join(folder, fn))
            self._pop.append(c)
