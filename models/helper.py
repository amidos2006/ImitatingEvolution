import numpy as np

def transform_input(level, position, size, channels):
    x,y=position["x"], position["y"]
    cl = np.pad(level,size//2)
    cl = cl[y:y+size,x:x+size]
    if channels > 1:
        cl = np.eye(channels)[cl].swapaxes(0, 2)
    return cl
