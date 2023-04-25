import numpy as np
from .helper import get_horz_symmetry, get_longest_path, get_number_regions, get_num_actions
from .helper import get_range_reward, get_num_tiles, discretize, get_distance_length, get_normalized_value
from PIL import Image
import os

def init(width, height):
    return np.random.randint(8, size=(height, width))

def fitness(genes, actions):
    number_player = get_num_tiles(genes, [2])
    player = get_range_reward(number_player, 1, 1, 0,\
                                genes.shape[0] * genes.shape[1])
    number_key = get_num_tiles(genes, [3])
    key = get_range_reward(number_key, 1, 1, 0,\
                                genes.shape[0] * genes.shape[1])
    number_door = get_num_tiles(genes, [4])
    door = get_range_reward(number_door, 1, 1, 0,\
                                genes.shape[0] * genes.shape[1])
    number_enemies = get_num_tiles(genes, [5, 6, 7])
    enemies = get_range_reward(number_enemies, 0, genes.shape[0] * genes.shape[1] / 10,\
                               0, genes.shape[0] * genes.shape[1])
    stats = (player + key + door + enemies) / 4.0

    player_key, player_key_value = get_distance_length(genes, [2], [3], [1, 2, 3, 5, 6, 7])    
    key_door, key_door_value = get_distance_length(genes, [3], [4], [1, 2, 3, 4, 5, 6, 7])
    temp = np.prod(genes.shape)
    playable = 2.0 - player_key_value / temp - key_door_value / temp 
    playable /= 2.0

    sol_length = get_range_reward(player_key + key_door,\
                                    genes.shape[0] * genes.shape[1] / 2,\
                                    genes.shape[0] * genes.shape[1] / 2)

    action = (np.array([act["action"] for act in actions]) != 0).sum() / (len(actions) + 1.0)

    added = 0
    if number_player == 1 and number_key == 1 and number_door == 1:
        added += playable
        if playable == 1:
            added += sol_length
    return (stats + added) / 3.0 #+ action / 100.0

def behaviors(genes, actions, bins):
    player_key, _ = get_distance_length(genes, [2], [3], [1, 2, 3, 5, 6, 7])
    key_door, _ = get_distance_length(genes, [3], [4], [1, 2, 3, 4, 5, 6, 7])
    sol_length = discretize(get_normalized_value(player_key + key_door,\
                                                0, genes.shape[0] * genes.shape[1] / 2), bins)
    vert_symmetry = discretize(get_normalized_value(get_horz_symmetry(genes.transpose()),\
                                                    0, genes.shape[0] * genes.shape[1] / 2), bins)
    horz_symmetry = discretize(get_normalized_value(get_horz_symmetry(genes),\
                                                    0, genes.shape[0] * genes.shape[1] / 2), bins)
    empty_tiles = discretize(get_normalized_value(get_num_tiles(genes, [1]),\
                                                    0, genes.shape[0] * genes.shape[1] / 2), bins)
    return [sol_length, empty_tiles]

def stopping(genes):
    number_player = get_num_tiles(genes, [2])
    number_key = get_num_tiles(genes, [3])
    number_door = get_num_tiles(genes, [4])
    player_key = get_distance_length(genes, [2], [3], [1, 2, 3, 5, 6, 7])
    key_door = get_distance_length(genes, [3], [4], [1, 2, 3, 4, 5, 6, 7])
    return number_player == 1 and number_key == 1 and number_door == 1 and player_key > 0 and key_door > 0

def stats(genes):
    player_key = get_distance_length(genes, [2], [3], [1, 2, 3, 5, 6, 7])
    key_door = get_distance_length(genes, [3], [4], [1, 2, 3, 4, 5, 6, 7])
    playable = 0
    if player_key > 0:
        playable += 1.0
    if key_door > 0:
        playable += 1.0
    playable /= 2.0
    return {
        "player": int(get_num_tiles(genes, [2])),
        "key": int(get_num_tiles(genes, [3])),
        "door": int(get_num_tiles(genes, [4])),
        "empty": int(get_num_tiles(genes, [1])),
        "sol_length": int(player_key + key_door),
        "playable": float(playable)
    }

def render(genes):
    scale = 16
    graphics = [
        Image.open(os.path.dirname(__file__) + "/_zelda/solid.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_zelda/empty.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_zelda/player.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_zelda/key.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_zelda/door.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_zelda/spider.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_zelda/bat.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_zelda/scorpion.png").convert('RGBA'),
    ]
    lvl = np.pad(genes, 1)
    lvl_image = Image.new("RGBA", (lvl.shape[1]*scale, lvl.shape[0]*scale), (0,0,0,255))
    for y in range(lvl.shape[0]):
        for x in range(lvl.shape[1]):
            lvl_image.paste(graphics[lvl[y][x]], (x*scale, y*scale, (x+1)*scale, (y+1)*scale))
    return lvl_image
