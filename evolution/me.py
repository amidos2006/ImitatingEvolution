from .chromosome import Chromosome
from .archive import Archive

class SMME:
    def __init__(self, start_size, width, height, model, init, fitness, behaviors, bins, target = None):
        self._width = width
        self._height = height

        self._model = model

        self._init_fn = init
        self._fitness_fn = fitness
        self._behaviors_fn = behaviors
        self._target_fn = target

        self._bins = bins

        self._map = Archive()
        for i in range(start_size):
            c = Chromosome(width, height, model, init, fitness, behaviors, bins)
            self._map.add(c)

    def update(self, epsilon = 0.25, times = 1):
        c = self._map.random()
        if self._target_fn == None:
            c = c.mutate(None, epsilon, times)
        else:
            locations = [c.behaviors() for c in self._map.get_all(self._map.keys())]
            c = c.mutate(self._target_fn(locations, self._bins, len(c.behaviors())), epsilon, times)
        self._map.add(c)

    def get_map(self):
        return self._map.clone()

    def get_best(self):
        chromosomes = self._map.get_all(self._map.keys())
        sorted(chromosomes, key=lambda c: c.fitness())
        return chromosomes[-1]

    def __len__(self):
        return len(self._map)

    def save(self, folder):
        self._map.save(folder)

    def load(self, folder):
        self._map.load(folder, self._width, self._height, self._model, self._init_fn,\
                       self._fitness_fn, self._behaviors_fn, self._bins)
