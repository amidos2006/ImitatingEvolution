import numpy as np
from .helper import get_horz_symmetry, get_longest_path, get_number_regions, get_num_actions
from .helper import get_range_reward, get_num_tiles, discretize
from PIL import Image
import os

def init(width, height):
    return np.random.randint(2, size=(height, width))

def fitness(genes, actions):
    number_regions = get_number_regions(genes, [1])
    regions = get_range_reward(number_regions, 1, 1, 1,\
                                           genes.shape[0] * genes.shape[1] / 10)
    longest = get_range_reward(get_longest_path(genes, [1]),\
                                           genes.shape[0] * genes.shape[1] / 2,\
                                           genes.shape[0] * genes.shape[1] / 2)
    action = (np.array([act["action"] for act in actions]) != 0).sum() / (len(actions) + 1.0)

    added = 0
    if number_regions == 1:
        added = longest
    return (regions + added) / 2.0 #+ action / 100.0

def behaviors(genes, actions, bins):
    longest = discretize(get_range_reward(get_longest_path(genes, [1]),\
                                               genes.shape[0] * genes.shape[1] / 2,\
                                               genes.shape[0] * genes.shape[1] / 2), bins)
    vert_symmetry = discretize(get_range_reward(get_horz_symmetry(genes.transpose()),\
                                                genes.shape[0] * genes.shape[1] / 2,\
                                                genes.shape[0] * genes.shape[1] / 2), bins)
    horz_symmetry = discretize(get_range_reward(get_horz_symmetry(genes),\
                                                genes.shape[0] * genes.shape[1] / 2,\
                                                genes.shape[0] * genes.shape[1] / 2), bins)
    empty_tiles = discretize(get_range_reward(get_num_tiles(genes, [1]),\
                                                 genes.shape[0] * genes.shape[1] / 2,\
                                                 genes.shape[0] * genes.shape[1] / 2), bins)
    return [horz_symmetry, longest]

def stopping(genes):
    return get_number_regions(genes, [1]) == 1

def render(genes):
    scale = 16
    graphics = [
        Image.open(os.path.dirname(__file__) + "/_binary/solid.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_binary/empty.png").convert('RGBA')
    ]
    lvl = np.pad(genes, 1)
    lvl_image = Image.new("RGBA", (lvl.shape[1]*scale, lvl.shape[0]*scale), (0,0,0,255))
    for y in range(lvl.shape[1]):
        for x in range(lvl.shape[0]):
            lvl_image.paste(graphics[lvl[y][x]], (x*scale, y*scale, (x+1)*scale, (y+1)*scale))
    return lvl_image
