import numpy as np

def string2array(string):
    string = string.replace("[", "").replace("]", "")
    rows = string.split("\n")
    result = []
    for r in rows:
        result.append(np.fromstring(r, dtype=int, sep=' '))
    return np.array(result)

def random_target(locations, max_value, bc_size):
    # maybe later will use the locations to specify where to visit
    return np.random.randint(max_value, size=bc_size)
