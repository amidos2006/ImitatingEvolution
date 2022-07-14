from .population import Population
from .chromosome import Chromosome

class SMES:
    def __init__(self, size, width, height, model, init, fitness, behaviors, bins, target = None):
        self._width = width
        self._height = height

        self._model = model

        self._init_fn = init
        self._fitness_fn = fitness
        self._behaviors_fn = behaviors
        self._target_fn = target

        self._bins = bins

        self._pop = Population(size)
        for i in range(size):
            c = Chromosome(width, height, model, init, fitness, behaviors, bins)
            self._pop.add(c)

    def update(self, death_perct = 0.5, tournment_size = 5, epsilon = 0.25, times = 1):
        death_size = int(death_perct * len(self._pop))
        self._pop.kill(death_size)
        for i in range(death_size):
            c = self._pop.select(tournment_size)
            if self._target_fn == None:
                c = c.mutate(None, epsilon, times)
            else:
                locations = [c.behaviors() for c in self._pop.get_all()]
                c = c.mutate(self._target_fn(locations, self._bins, len(c.behaviors())), epsilon, times)
            self._pop.add(c)

    def get_pop(self):
        return self._pop.clone()

    def get_best(self):
        return self._pop.get(-1)

    def __len__(self):
        return len(self._pop)

    def save(self, folder):
        self._pop.save(folder)

    def load(self, folder):
        self._pop.load(folder, self._width, self._model, self._height, self._init_fn,\
                       self._fitness_fn, self._behaviors_fn, self._bins)
