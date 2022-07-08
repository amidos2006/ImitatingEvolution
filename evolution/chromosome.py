import json
import numpy as np
from .helper import string2array

class Chromosome():
    def __init__(self, width, height, model, init, fitness, behaviors, bins):
        self._width = width
        self._height = height

        self._model = model

        self._init_fn = init
        self._fitness_fn = fitness
        self._behaviors_fn = behaviors
        self._bins = bins

        self._genes = self._init_fn(self._width, self._height)
        self._start_genes = self._genes.copy()
        self._actions = []
        self._fitness = -1
        self._behaviors = []

    def clone(self):
        c = Chromosome(self._width, self._height, self._model, self._init_fn,
                       self._fitness_fn, self._behaviors_fn, self._bins)
        c._genes = self._genes.copy()
        c._start_genes = self._start_genes.copy()
        c._actions = self._actions.copy()
        c._fitness = self._fitness
        c._behaviors = self._behaviors.copy()
        return c

    def erase_history(self):
        c = self.clone()
        c._actions = []
        c._start_genes = self._genes.copy()
        return c

    def mutate(self, target, epsilon, times):
        c = self.clone()
        c._fitness = -1
        c._behaviors = []
        times = np.random.randint(times) + 1
        for i in range(times):
            x,y=np.random.randint(self._width), np.random.randint(self._height)
            if np.random.random() < epsilon:
                value = np.random.randint(self._model._outputs)
                action = {"x": x, "y": y, "action": value}
            else:
                action = self._model.mutate(c._genes, x, y, target)
            action["behaviors"] = c.behaviors()
            if action["action"] > 0:
                c._genes[y][x] = action["action"] - 1
            c._actions.append(action)
        return c

    def behaviors(self):
        if len(self._behaviors) == 0:
            self._behaviors = self._behaviors_fn(self._genes, self._actions, self._bins)
        return self._behaviors

    def fitness(self):
        if self._fitness < 0:
            self._fitness = self._fitness_fn(self._genes, self._actions)
        return self._fitness

    def save(self, file_name):
        with open(file_name, 'w') as f:
            temp = {
                "width": self._width,
                "height": self._height,
                "genes": np.array2string(self._genes),
                "start": np.array2string(self._start_genes),
                "actions": self._actions,
                "fitness": self.fitness(),
                "behaviors": self.behaviors(),
            }
            f.write(json.dumps(temp))

    def load(self, file_name):
        with open(file_name, 'r') as f:
            temp = json.load(f)
            self._width = temp["width"]
            self._height = temp["height"]
            self._genes = string2array(temp["genes"])
            self._start_genes = string2array(temp["start"])
            self._actions = temp["actions"]
            self._fitness = temp["fitness"]
            self._behaviors = temp["behaviors"]
