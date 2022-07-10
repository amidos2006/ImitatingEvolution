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

# Run Files
There are currently 3 entry points in the project:
- run_es.py
- run_inference.py
- run_me.py

## run_es.py
This entrypoint runs a simple mu-lamda algorithm on the binary environment (more in the [Games](#games) section). There are hyperparamters set at the top of the file (TODO move these to a settings.yml for mass experimentation), regarding model settings, ES settings, and data creation (i.e. mutation-trajectory) settings. 

## run_inference.py
This entrypoint runs trained model inferences on randomly generated binary environment maps. There are hyperparameters set at the top of the file (TODO move theses into a settings.yml for mass experimentation) regarding model and inference settings.

## run_me.py
This entrypoint runs a simple MAP-Elites algorithm on the binary environment. There are hyperparameters set at the top of the file (TODO move theses into a settings.yml for mass experimentation) regarding model settings, ME settings, and data creation (i.e. mutation-trajectory) settings.

# Games
Currently there is only one game environment (and some utility functions) in this project.

## Binary
The binary game environment is a maze environment. Tiles are either passable or not. Fitness is calculated by observing connectivity and the "longest-shortest" path between any two pairs of tiles. Both of these factors contribute equally to map fitness.

There are currently 4 possible dimensions to be considered for ME in Binary:
- longest_path
- vertical symmetry
- horizontal symmetry
- empty_tiles

# Evolution Methods

## ES (mu+labmda)
## MAP-Elites