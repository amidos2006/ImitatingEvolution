import games.binary as binary, games.zelda as zelda, games.sokoban as sokoban
import json
import os

def get_game(game_name):
    if not os.path.exists(f"config/games/{game_name}.json"):
        raise NotImplementedError(f'{game_name} is not implemented yet.')
    with open(f"config/games/{game_name}.json") as f:
        result = json.load(f)
    result["name"] = game_name
    if game_name == "binary":
        result["init"] = binary.init
        result["fitness"] = binary.fitness
        result["behaviors"] = binary.behaviors
        result["stopping"] = binary.stopping
        result["render"] = binary.render
        result["stats"] = binary.stats
    if game_name == "zelda":
        result["init"] = zelda.init
        result["fitness"] = zelda.fitness
        result["behaviors"] = zelda.behaviors
        result["stopping"] = zelda.stopping
        result["render"] = zelda.render
        result["stats"] = zelda.stats
    if game_name == "sokoban":
        result["init"] = sokoban.init
        result["fitness"] = sokoban.fitness
        result["behaviors"] = sokoban.behaviors
        result["stopping"] = sokoban.stopping
        result["render"] = sokoban.render
        result["stats"] = sokoban.stats
    return result
