import numpy as np
from .helper import get_horz_symmetry, get_longest_path, get_number_regions, get_num_actions
from .helper import get_range_reward, get_num_tiles, discretize, get_distance_length
from PIL import Image

def init(width, height):
    return np.random.randint(8, size=(height, width))

def fitness(genes, actions):
    number_player = get_num_tiles(genes, [2])
    player = get_range_reward(number_player, 1, 1, 1,\
                                genes.shape[0] * genes.shape[1] / 10)
    number_key = get_num_tiles(genes, [3])
    key = get_range_reward(number_key, 1, 1, 1,\
                                genes.shape[0] * genes.shape[1] / 10)
    number_door = get_num_tiles(genes, [4])
    door = get_range_reward(number_door, 1, 1, 1,\
                                genes.shape[0] * genes.shape[1] / 10)
    stats = (player + key + door) / 3.0

    player_key = get_distance_length(genes, [2], [3], [1, 2, 3, 5, 6, 7])
    key_door = get_distance_length(genes, [3], [4], [1, 2, 3, 4, 5, 6, 7])
    playable = 0
    if player_key > 0:
        playable += 1.0
    if key_door > 0:
        playable += 1.0
    playable /= 2.0

    sol_length = get_range_reward(player_key + key_door,\
                                    genes.shape[0] * genes.shape[1] / 2,\
                                    genes.shape[0] * genes.shape[1] / 2)

    action = (np.array([act["action"] for act in actions]) != 0).sum() / (len(actions) + 1.0)

    added = 0
    if number_player == 1 and number_key == 1 and number_door == 1 and playable == 1:
        added = sol_length
    return (stats + playable + added) / 3.0 #+ action / 100.0

def behaviors(genes, actions, bins):
    player_key = get_distance_length(genes, [2], [3], [1, 2, 3, 5, 6, 7])
    key_door = get_distance_length(genes, [3], [4], [1, 2, 3, 4, 5, 6, 7])
    longest = discretize(get_range_reward(player_key + key_door,\
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
    return [longest, empty_tiles]

def render(genes):
    scale = 16
    self._graphics = [
        Image.open(os.path.dirname(__file__) + "/zelda/solid.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/zelda/empty.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/zelda/player.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/zelda/key.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/zelda/door.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/zelda/spider.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/zelda/bat.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/zelda/scorpion.png").convert('RGBA'),
    ]
    lvl = np.pad(genes, 1)
    lvl_image = Image.new("RGBA", (lvl.shape[1]*scale, lvl.shape[0]*scale), (0,0,0,255))
    for y in range(lvl.shape[1]):
        for x in range(lvl.shape[0]):
            lvl_image.paste(self._graphics[lvl[y][x]], (x*scale, y*scale, (x+1)*scale, (y+1)*scale))
    return lvl_image