# Immitating Evolution
This project concerns itself with a series of research projects based on the paper [Learning to Generate Levels by Imitaing Evolution](https://arxiv.org/pdf/2206.05497.pdf), available on arxiv.

Search-based procedural content generation (PCG) is a well-known method used for level generation in games. Its key advantage is that it is generic and able to satisfy functional constraints. However, due to the heavy computational costs to run these algorithms online, search-based PCG is rarely utilized for real-time generation. In this repo, we introduce a new type of iterative level generator using machine learning. We train models to imitate the evolutionary process and use the model to generate levels. This trained model is able to modify noisy levels sequentially to create better levels without the need for a fitness function during inference.

We demonstrate using several game environments, including a binary maze environment, how ML models can be trained to imitate mutation.

# Table of Contents
- [Setup](#setup)
- [Run Files](#run-files)
- [Games](#games)
- [Evolutionary Algorithms](#evolutionary-algorithms)
- [Trained Generator](#trained-generator)

# Setup
This project requires `torch` (for the neural networks), you could use `tensorflow` or any other libraries but make sure to reimplement the functions and classes inside [nn folder](https://github.com/amidos2006/ImmitatingEvolution/tree/main/nn). Beside `torch`, you need `numpy`, `Pillow` for rendering games, and `tqdm` to show progress.

You could install all these libraries on your own or easily install them using [`requirements.txt`](https://github.com/amidos2006/ImmitatingEvolution/blob/main/requirements.txt) by:
```
pip install -r requirements.txt
```

# Run Files
There are currently 3 entry points in the project:
- [run_es.py](#run_espy)
- [run_inference.py](#run_inferencepy)
- [run_me.py](#run_mepy)

## run_es.py
This entrypoint runs a simple mu+lamda evolution strategy algorithm (check [Evolutionary Algorithms section](#evolutionary-algorithms) for more details). To run you can simply run the file directly
```
python run_es.py
```
This will run the evolution for 3 times so we will end up with 3 trained networks from 3 independent runs. You can change from the command line some of the settings to allow for different games and different experiments. Here a list of all the commandline arguments:
- `--config` or `-c` specify all the upcoming parameters in a json file format.
- `--game` or `-g` to specify which game to play [`binary`, `zelda`, `sokoban`].
- `--observation` or `-o` to specify the percentage of the observation. Values range between (0, 2] as they are percentage of the largest dimension.
- `--type` or `-t` specifies the type of trajectory extraction method [`evol`, `inbet`, `init`].
- `--number` or `-n` specifies how many times to run the evolution and train a network (default: `3`).
- `--train` or `--no-train` is a flag parameter to allow for either `assisted` or `normal` evolution (default: `--train`).
- `--conditional` or `--no-conditional` is a flag parameter to allow for either `conditional` or `non conditional` neural network to be used during imitating evolution (default: `--no-conditional`).

For example, if you want to train a network on game of `zelda` with full observation (`1`) and using random walks for trajectory creation:
```
python run_es.py -g zelda -o 1 -t init
```

## run_me.py
This entrypoint runs a simple MAP-Elites algorithm. To run you can simply run the file directly
```
python run_me.py
```
This will run the evolution for 3 times so we will end up with 3 trained networks from 3 independent runs. You can change from the command line some of the settings to allow for different games and different experiments. Here a list of all the commandline arguments:
- `--config` or `-c` specify all the upcoming parameters in a json file format.
- `--game` or `-g` to specify which game to play [`binary`, `zelda`, `sokoban`].
- `--observation` or `-o` to specify the percentage of the observation. Values range between (0, 2] as they are percentage of the largest dimension.
- `--type` or `-t` specifies the type of trajectory extraction method [`evol`, `inbet`, `init`].
- `--number` or `-n` specifies how many times to run the evolution and train a network (default: `3`).
- `--train` or `--no-train` is a flag parameter to allow for either `assisted` or `normal` evolution (default: `--train`).
- `--conditional` or `--no-conditional` is a flag parameter to allow for either `conditional` or `non conditional` neural network to be used during imitating evolution (default: `--no-conditional`).

For example, if you want to train a network on game of `zelda` with full observation (`1`), using random walks for trajectory creation, and conditional network:
```
python run_me.py -g zelda -o 1 -t init --conditional
```

## run_inference.py
This entrypoint runs trained model inferences on randomly generated environment maps. You can control what to run based on command line arguments or config file that contains these values. To run the inference file there is two basic ways. Either using command line argument:
```
python run_inference.py -m results/es/2/binary_evol_2_8_noncond_assisted/1999/
```
or using config file that contains all the command line args:
```
python run_inference.py -c config/inference/binary_100.json
```
The hyperparameters for the files are:
- `--config` or `-c` is a config file that encompass all the coming parameters so if you use it, it will overwrite all the rest.
- `--model` or `-m` is the folder for the model that need testing
- `--number` or `-n` is the number of maps to generate during inference
- `--type` or `-t` is the type of action taken in the network [`greedy` or `softmax`].
- `--visualize` or `--no-visualize` is to record the inference in a mp4 file (beware it is very slow so it might take forever).

To use trained model from previous experiment (`results/es/2/zelda_evol_2_8_noncond_assisted/1999/`) to generate `1000` maps and using `greedy` network.
```
python run_inference.py -m results/es/2/zelda_evol_2_8_noncond_assisted/1999/ -n 1000 -t greedy
```

# Games
Currently there is three game environment in this project:
- [Binary](#binary)
- [Zelda](#zelda)
- [Sokoban](#sokoban)

Each game provide 6 functions:
- `init(width, height)`: return a 2D random numpy array of width, height that satisfies the specified game.
- `fitness(level, actions)`: return the fitness function for the current level and actions taken to reach that level.
- `behavior(level, actions, bins)`: return a numpy array of integer numbers that represents the behavior characteristic of the current input level and actions with values between [0, bins).
- `stopping(level)`: return true if the level is satisfactory. Used mainly during inference.
- `stats(level)`: return the characteristic of the level in form of python object.
- `render(level)`: render the level into a Pillow Image and return it.

To add more games, you need to create a new file that have all the above functions, also you will need to add a config file in [`config/games/`](https://github.com/amidos2006/ImmitatingEvolution/tree/main/config/games), and finally you will need to register the game in `get_game(game_name)` function in the [`games/__init__.py`](https://github.com/amidos2006/ImmitatingEvolution/blob/main/games/__init__.py).

For the config file, you need to provide the following parameters in the file:
- `width`: the width of the maps
- `height`: the height of the maps
- `num_tiles`: the number of different tiles each tile can have in the map.
- `num_behaviors`: the number of different behaviors that behavior function returns.
- `behavior_bins`: the number of bins that each behavior dimension have values between [0, behavior_bins).

## Binary
The binary game environment is a maze environment. Tiles are either passable or not. Fitness is calculated by observing connectivity and the "longest-shortest" path between any two pairs of tiles. Both of these factors contribute equally to map fitness.

There are currently 4 possible dimensions to be considered for ME in Binary:
- longest_path
- vertical symmetry
- horizontal symmetry
- empty_tiles

## Zelda

## Sokoban

# Evolutionary Algorithms

- [Mu+Lambda ES](#mulambda-es)
- [MAP-Elites](#map-elites)

## Mu+Lambda ES

The config file contains the hyperparameters for the Mu+Lambda evolution strategy. These suppose to be fixed between the different experiments. Here is a list of the hyperparameters:
- `pop_size` is the size of the population.
- `death_perct` is the percentage of population that get killed every generation.
- `tournment_size` is the tournament size for the tournament selection.
- `gen_number` is the number of generations that the evolution strategy is running for.
- `mutation_length` is the maximum amount of tiles that can be mutated.
- `epsilon` is the percentage of total random mutation and not derived from the trained network in case of assisted evolution.
- `periodic_save` is the number of generation after which the algorithm will save everything.

## MAP-Elites

The config file contains the hyperparameters for the MAP-Elites algorithm. These suppose to be fixed between the different experiments. Here is a list of the hyperparameters:
- `start_size` is the size of the starting amount of chromosomes.
- `iterations` is the number of updates that the MAP-Elites goes through.
- `mutation_length` is the maximum amount of tiles that can be mutated.
- `epsilon` is the percentage of total random mutation and not derived from the trained network in case of assisted evolution.
- `periodic_save` is the number of iterations after which the algorithm will save everything.

