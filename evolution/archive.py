import numpy as np
import os

class Archive:
    def __init__(self):
        self._map = {}

    def __len__(self):
        return len(self._map)

    def __str__(self):
        return f"Arhcive Size: {len(self._map)}\nValues:\n{str(self.keys())}"

    def keys(self, dim=-1, value=-1):
        if len(self._map) == 0:
            return np.array([])
        num_dim = len(list(self._map.keys())[0].split(","))
        keys = list(self._map.keys())
        result = []
        for key in keys:
            values = key.split(",")
            temp = []
            for v in values:
                temp.append(int(v))
            result.append(temp)
        result = np.array(result)
        if dim >= 0:
            result = np.array([k for k in result if k[dim] == value])
        return result

    def clone(self):
        archive = Archive()
        archive._map = self._map.copy()
        return archive

    def add(self, chromosome):
        key = ",".join([str(temp) for temp in chromosome.behaviors()])
        if key not in self._map or (key in self._map and chromosome.fitness() >= self._map[key].fitness()):
            if key in self._map and chromosome.fitness() == self._map[key].fitness() and np.random.random() < 0.5:
                return
            self._map[key] = chromosome

    def random(self):
        keys = list(self._map.keys())
        index = np.random.randint(len(keys))
        return self._map[keys[index]]

    def get(self, dimension):
        key = ",".join([str(temp) for temp in dimension])
        if key in self._map:
            return self._map[key]
        return None

    def get_all(self, dimensions):
        result = []
        for dim in dimensions:
            result.append(self.get(dim))
        return result

    def save(self, folder):
        os.makedirs(folder)
        for key in self._map.keys():
            self._map[key].save(os.path.join(folder, f"{key}.json"))

    def load(self, folder, width, height, init, mutate, fitness, behaviors):
        self._map = {}
        files = [fn for fn in os.listdir(folder) if ".json" in fn]
        for fn in files:
            key = fn.split(".json")[0]
            self._map[key] = Chromosome(width, height, init, mutate, fitness, behaviors)
            self._map[key].load(os.path.join(folder, f"{key}.json"))
