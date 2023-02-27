import numpy as np
from .helper import get_horz_symmetry, get_longest_path, get_number_regions, get_num_actions
from .helper import get_range_reward, get_num_tiles, discretize, get_distance_length, get_normalized_value
from ._sokoban.engine import State,BFSAgent,AStarAgent
from PIL import Image
import os

def _run_game(genes):
    solver_power = 5000
    lvl = np.pad(genes, 1)
    gameCharacters="# @$."
    lvlString = ""
    for i in range(lvl.shape[0]):
        for j in range(lvl.shape[1]):
            lvlString += gameCharacters[lvl[i][j]]
            if j == lvl.shape[1]-1:
                lvlString += "\n"
    state = State()
    state.stringInitialize(lvlString.split("\n"))
    aStarAgent = AStarAgent()
    bfsAgent = BFSAgent()
    sol,solState,iters = bfsAgent.getSolution(state, solver_power)
    if solState.checkWin():
        return 0, len(sol)
    sol,solState,iters = aStarAgent.getSolution(state, 1, solver_power)
    if solState.checkWin():
        return 0, len(sol)
    sol,solState,iters = aStarAgent.getSolution(state, 0.5, solver_power)
    if solState.checkWin():
        return 0, len(sol)
    sol,solState,iters = aStarAgent.getSolution(state, 0, solver_power)
    if solState.checkWin():
        return 0, len(sol)
    return solState.getHeuristic(), 0

def init(width, height):
    return np.random.randint(5, size=(height, width))

def fitness(genes, actions):
    number_player = get_num_tiles(genes, [2])
    player = get_range_reward(number_player, 1, 1, 0,\
                                genes.shape[0] * genes.shape[1])
    number_crates = get_num_tiles(genes, [3])
    crates = get_range_reward(number_crates, 2, genes.shape[0] * genes.shape[1])
    number_targets = get_num_tiles(genes, [4])
    crate_target = get_range_reward(abs(number_crates - number_targets), 0, 0, 0,\
                                genes.shape[0] * genes.shape[1] / 10)
    stats = (player + crates + crate_target) / 3.0

    added = 0
    if number_player == 1 and number_crates > 0 and number_crates == number_targets:
        heuristic, sol_length = _run_game(genes)
        heuristic = get_range_reward(heuristic, 0, 0, 0,\
                                    (genes.shape[0] + genes.shape[1]) * number_crates)
        sol_length = get_range_reward(sol_length,\
                                    genes.shape[0] * genes.shape[1] * number_crates,\
                                    genes.shape[0] * genes.shape[1] * number_crates)
        added = (heuristic + sol_length) / 2.0

    action = (np.array([act["action"] for act in actions]) != 0).sum() / (len(actions) + 1.0)

    return (stats + added) / 2.0 #+ action / 100.0

def behaviors(genes, actions, bins):
    number_player = get_num_tiles(genes, [2])
    number_crates = get_num_tiles(genes, [3])
    number_targets = get_num_tiles(genes, [4])
    sol_length = 0
    if number_player == 1 and number_crates > 0 and number_crates == number_targets:
        _, sol_length = _run_game(genes)

    sol_length = discretize(get_normalized_value(sol_length,\
                                                 0, genes.shape[0] * genes.shape[1] * number_crates), bins)
    vert_symmetry = discretize(get_normalized_value(get_horz_symmetry(genes.transpose()),\
                                                    0, genes.shape[0] * genes.shape[1] / 2), bins)
    horz_symmetry = discretize(get_normalized_value(get_horz_symmetry(genes),\
                                                    0, genes.shape[0] * genes.shape[1] / 2), bins)
    empty_tiles = discretize(get_normalized_value(get_num_tiles(genes, [1]),\
                                                    0, genes.shape[0] * genes.shape[1] / 2), bins)
    return [sol_length, empty_tiles]

def stopping(genes):
    number_player = get_num_tiles(genes, [2])
    number_crates = get_num_tiles(genes, [3])
    number_targets = get_num_tiles(genes, [4])
    if number_player == 1 and number_crates > 0 and number_crates == number_targets:
        _, sol_length = _run_game(genes)
        if sol_length > 0:
            return True
    return False

def stats(genes):
    number_player = get_num_tiles(genes, [2])
    number_crates = get_num_tiles(genes, [3])
    number_targets = get_num_tiles(genes, [4])
    sol_length = 0
    if number_player == 1 and number_crates > 0 and number_crates == number_targets:
        _, sol_length = _run_game(genes)
    return {
        "player": int(number_player),
        "crates": int(number_crates),
        "targets": int(number_targets),
        "empty": int(get_num_tiles(genes, [1])),
        "sol_length": int(sol_length),
    }

def render(genes):
    scale = 16
    graphics = [
        Image.open(os.path.dirname(__file__) + "/_sokoban/solid.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_sokoban/empty.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_sokoban/player.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_sokoban/crate.png").convert('RGBA'),
        Image.open(os.path.dirname(__file__) + "/_sokoban/target.png").convert('RGBA')
    ]
    lvl = np.pad(genes, 1)
    lvl_image = Image.new("RGBA", (lvl.shape[1]*scale, lvl.shape[0]*scale), (0,0,0,255))
    for y in range(lvl.shape[0]):
        for x in range(lvl.shape[1]):
            lvl_image.paste(graphics[lvl[y][x]], (x*scale, y*scale, (x+1)*scale, (y+1)*scale))
    return lvl_image
