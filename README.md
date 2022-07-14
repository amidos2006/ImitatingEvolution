# Immitating Evolution

# Table of Contents
- [Project Description](#project-description)
- [Run Files](#run-files)
- [Games](#games)
- [Evolution Methods](#evolution-methods)


# Project Description
This project concerns itself with a series of research projects based on the paper [Learning to Generate Levels by Imitaing Evolution](https://arxiv.org/pdf/2206.05497.pdf), available on arxiv.

Search-based procedural content generation (PCG) is a well-known method used for level generation in games. Its key advantage is that it is generic and able to satisfy functional constraints. However, due to the heavy computational costs to run these algorithms online, search-based PCG is rarely utilized for real-time generation. In this repo, we introduce a new type of iterative level generator using machine learning. We train models to imitate the evolutionary process and use the model to generate levels. This trained model is able to modify noisy levels sequentially to create better levels without the need for a fitness function during inference.

We demonstrate using several game environments, including a binary maze environment, how ML models can be trained to imitate mutation.

# Setup

# Run Files
There are currently 3 entry points in the project:
- run_es.py
- run_inference.py
- run_me.py

## run_es.py
This entrypoint runs a simple mu-lamda algorithm on the binary environment (more in the [Games](#games) section). The evolution strategy hyperparameters can be set in `config/evolution/es.json`. The file has command line argument to set other hyperparameters so you can easily change without need to change the settings file.

### How to run
To run you can simply run the file directly
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

### Config File
The config file contains the hyperparameters for the Mu+Lambda evolution strategy. These suppose to be fixed between the different experiments. Here is a list of the hyperparameters:
- `pop_size` is the size of the population.
- `death_perct` is the percentage of population that get killed every generation.
- `tournment_size` is the tournament size for the tournament selection.
- `gen_number` is the number of generations that the evolution strategy is running for.
- `mutation_length` is the maximum amount of tiles that can be mutated.
- `epsilon` is the percentage of total random mutation and not derived from the trained network in case of assisted evolution.
- `periodic_save` is the number of generation after which the algorithm will save everything.

## run_me.py
This entrypoint runs a simple MAP-Elites algorithm on the binary environment. There are hyperparameters set at the top of the file (TODO move theses into a settings.yml for mass experimentation) regarding model settings, ME settings, and data creation (i.e. mutation-trajectory) settings.

## run_inference.py
This entrypoint runs trained model inferences on randomly generated binary environment maps. There are hyperparameters set at the top of the file (TODO move theses into a settings.yml for mass experimentation) regarding model and inference settings.

# Games
Currently there is only one game environment (and some utility functions) in this project.

## Binary
The binary game environment is a maze environment. Tiles are either passable or not. Fitness is calculated by observing connectivity and the "longest-shortest" path between any two pairs of tiles. Both of these factors contribute equally to map fitness.

There are currently 4 possible dimensions to be considered for ME in Binary:
- longest_path
- vertical symmetry
- horizontal symmetry
- empty_tiles

## Zelda

## Sokoban

# Evolution Methods

## ES (mu+labmda)
## MAP-Elites
