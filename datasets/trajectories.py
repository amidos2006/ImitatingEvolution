import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import numpy as np
from .helper import transform_input

INIT_DATA = "init"
INBET_DATA = "inbet"
EVOL_DATA = "evol"

class Trajectories(Dataset):
    def __init__(self, chromosomes, size, channels, type, portion, bins, repeats, early_threshold = 1.0, data_type="train"):
        self.chromosomes = chromosomes
        self.size = size
        self.channels = channels
        self.type = type
        self.portion = portion
        self.bins = bins
        self.repeats = repeats
        self.early_threshold = early_threshold
        self.data_type = data_type

    def setup(self):
        levels = []
        targets = []
        actions = []
        if self.portion > 0 and len(self.chromosomes) > self.portion:
            self.chromosomes = sorted(self.chromosomes, key=lambda c: c.fitness())[-self.portion-1:-1]
        for c in self.chromosomes:
            for i in range(self.repeats):
                if self.type == INIT_DATA:
                    level = c._init_fn(c._width, c._height)
                    while c._fitness_fn(level, c._actions) >= c.fitness():
                        level = c._init_fn(c._width, c._height)
                else:
                    level = c._start_genes.copy()
                if type == EVOL_DATA:
                    pos = []
                    for act in c._actions:
                        pos.append({"x": act["x"], "y": act["y"]})
                else:
                    pos = []
                    for x in range(c._width):
                        for y in range(c._height):
                            pos.append({"x": x, "y": y})
                    np.random.shuffle(pos)
                early_stopping = len(pos) + 1
                for ai,p in enumerate(pos):
                    early_stopping -= 1
                    cl = transform_input(level, p, self.size, self.channels)
                    levels.append(cl)
                    targets.append(np.array(c.behaviors()) / self.bins)
                    if c._fitness_fn(level, c._actions) > self.early_threshold:
                        early_stopping = np.random.randint(len(pos) - ai)
                    if early_stopping > 0:
                        if type == EVOL_DATA:
                            actions.append(c._actions[ai]["action"])
                            if actions[-1] != 0:
                                level[p["y"]][p["x"]] = actions[-1] - 1
                        else:
                            if c._genes[p["y"]][p["x"]] == level[p["y"]][p["x"]]:
                                actions.append(0)
                            else:
                                actions.append(c._genes[p["y"]][p["x"]] + 1)
                                level[p["y"]][p["x"]] = c._genes[p["y"]][p["x"]]
                    else:
                        actions.append(0)
        self.levels, self.targets, self.actions = np.array(levels), np.array(targets), np.array(actions)

    def __getitem__(self, index):
        level = self.levels[index]
        target = self.targets[index]
        actions = self.actions[index]
        return level.reshape(self.channels, level.shape[0], level.shape[1]), target.reshape(self.channels, target.shape[0]), actions

    
    def __len__(self):
        return len(self.levels)