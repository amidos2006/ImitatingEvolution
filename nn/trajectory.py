import numpy as np
from .helper import transform_input

INIT_DATA = "init"
INBET_DATA = "inbet"
EVOL_DATA = "evol"

def extract_data(chromosomes, size, channels, type, portion, bins, repeats, early_threshold = 1.0):
    levels = []
    targets = []
    actions = []
    if portion > 0 and len(chromosomes) > portion:
        chromosomes = sorted(chromosomes, key=lambda c: c.fitness())[-portion-1:-1]
    for c in chromosomes:
        for i in range(repeats):
            if type == INIT_DATA:
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
                cl = transform_input(level, p, size, channels)
                levels.append(cl)
                targets.append(np.array(c.behaviors()) / bins)
                if c._fitness_fn(level, c._actions) > early_threshold:
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
    return np.array(levels), np.array(targets), np.array(actions)
