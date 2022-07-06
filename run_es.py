from tqdm import trange, tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import games.binary as binary

from evolution.helper import random_target
from evolution.es import SMES

from nn.model import SOFTMAX_ACT, GREEDY_ACT, SMNN
from nn.trajectory import EVOL_DATA, INIT_DATA, INBET_DATA, extract_data
from nn.train import train


if __name__ == "__main__":
    # game parameters
    problem_name = "binary"                           # name of the problems for saving purposes
    width = 14                                        # width of the generated level
    height = 14                                       # height of the generated level
    num_tiles = 2                                     # number of tiles in the level
    num_behaviors = 2                                 # number of behavior characteristic
    behavior_bins = 20                                # number of bins after discretize
    init = binary.init                                # initialization function for problem
    fitness = binary.fitness                          # fitness function for the problem
    behaviors = binary.behaviors                      # behavior characteristic function for the problem

    # Evolution Parameters
    pop_size = 100                                    # population size
    death_perct = 0.5                                 # percentage of killed chromosomes
    tournment_size = 5                                # size of tournment in tournment selection
    gen_number = 2000                                 # number of generations
    mutation_length = 1                               # how many tiles to mutate
    epsilon = 0.25                                    # probability of doing random mutation not from model
    periodic_save = 20                                # how often to save

    # Data Creation Parameter
    window_size = 8                                   # cropped view of the observation (can be any value)
    portion = 10                                      # amount of chromosomes used to train the network
    increase_data = 1                                 # increase data size by that value (doesn't work with EVOL_DATA)
    data_creation = EVOL_DATA                         # method of creating the data
    append_data = False                               # new data is generated beside old ones

    # Training Parameters
    allow_train = True                                # allow training neural network
    train_period = 100                                # frequency of training the network
    train_epochs = 2                                  # number of epochs used in the middle of evolution
    batch_size = 32                                   # minibatch size during training
    learning_rate = 0.00001                           # optimizer learning rate
    reset_model = True                                # reset the model weights
    optimizer_fn = optim.Adam                         # used optimizer
    loss_fn = nn.CrossEntropyLoss                     # used loss function

    # Model Parameter
    conditional = False                               # is the model conditional or not
    mutate_type = SOFTMAX_ACT                         # type of mutation when using the network

    # extra parameters
    num_experiments = 3                               # number of experiments to run
    save_folder = f"{problem_name}_{data_creation}_{train_epochs}_{['normal','assitant'][allow_train]}"

    for experiment in range(num_experiments):
        if conditional:
            cond_length = num_behaviors
            target = random_target
        else:
            cond_length = 0
            target = None
        model = SMNN(window_size, num_tiles, cond_length, mutate_type)
        evolver = SMES(pop_size, width, height, model, init, fitness, behaviors, behavior_bins, target)
        total_levels, total_targets, total_actions = np.array([]).reshape((0,window_size,window_size)),\
                                                     np.array([]).reshape((0,num_behaviors)), np.array([]).reshape((0))
        pbar = trange(gen_number)
        for i in pbar:
            evolver.update(death_perct, tournment_size, epsilon, mutation_length)
            pbar.set_postfix_str(f"Best Fitness: {evolver.get_best().fitness()}")
            model_changed = False
            if (allow_train and ((i > 0 and i % train_period == 0) or i == gen_number-1)) or (i == gen_number-1):
                if reset_model:
                    model.reset_parameters()
                optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
                loss = loss_fn()
                if data_creation == EVOL_DATA:
                    increase_data = 1
                levels, targets, actions = extract_data(evolver.get_pop().get_all(), window_size, model._channels,\
                                                        data_creation, portion, behavior_bins, increase_data)
                if append_data:
                    total_levels = np.concatenate((total_levels, levels))
                    total_targets = np.concatenate((total_targets, targets))
                    total_actions = np.concatenate((total_actions, actions))
                else:
                    total_levels = levels
                    total_targets = targets
                    total_actions = actions
                train(model, optimizer, loss, train_epochs, batch_size, total_levels, total_targets, total_actions)
                model_changed = True

            if i % periodic_save == 0 or i == gen_number - 1:
                import os
                import shutil
                model_path = f"results/es/{experiment}/{save_folder}/{i}"
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                os.makedirs(model_path)
                evolver.save(os.path.join(model_path, "population"))
                if model_changed or i == 0:
                    torch.save(model, os.path.join(model_path, "model"))
