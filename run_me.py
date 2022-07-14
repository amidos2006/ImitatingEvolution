from tqdm import trange, tqdm
import numpy as np
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim

from games import get_game

from evolution import get_evol_param
from evolution.helper import random_target
from evolution.me import SMME

from nn import get_train_param, save_model
from nn.model import SOFTMAX_ACT, GREEDY_ACT, SMNN
from nn.trajectory import EVOL_DATA, INIT_DATA, INBET_DATA, extract_data
from nn.train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Evolutionary Imitation using MAP-Elites')
    parser.add_argument('--config', '-c', default="",
                        help='all the settings in a json file')
    parser.add_argument('--game', '-g', default="binary",
                        help='the game that we need to evolve and test (default: binary)')
    parser.add_argument('--observation', '-o', type=float, default=0.5,
                        help='the observation window size as percentage of biggest dimension (default: 0.5)')
    parser.add_argument('--number', '-n', type=int, default=3,
                        help='number of time to repeat the experiment (default: 3)')
    parser.add_argument('--train', action="store_true",
                        help='flag parameter to train in the middle of evolution')
    parser.add_argument('--no-train', dest="train", action="store_false")
    parser.set_defaults(train=True)
    parser.add_argument('--conditional', action="store_true",
                        help='flag parameter to train a conditional network instead of normal one')
    parser.add_argument('--no-conditional', dest="conditional", action="store_false")
    parser.set_defaults(conditional=False)
    parser.add_argument('--type', '-t', default=EVOL_DATA,
                        help='method of creating data set for training (values: evol, init, inbet)')
    args = parser.parse_args()

    if len(args.config) > 0:
        with open(args.config) as f:
            temp = json.load(f)
            args.game = temp["game"]
            args.observation = temp["observation"]
            args.number = temp["number"]
            args.type = temp["type"]
            args.conditional = temp["conditional"]
            args.train = temp["train"]

    # game parameters
    game_name = args.game                             # name of the problems for saving purposes
    game_info = get_game(game_name)
    width = game_info["width"]                        # width of the generated level
    height = game_info["height"]                      # height of the generated level
    num_tiles = game_info["num_tiles"]                # number of tiles in the level
    num_behaviors = game_info["num_behaviors"]        # number of behavior characteristic
    behavior_bins = game_info["behavior_bins"]        # number of bins after discretize
    init = game_info["init"]                          # initialization function for problem
    fitness = game_info["fitness"]                    # fitness function for the problem
    behaviors = game_info["behaviors"]                # behavior characteristic function for the problem

    # Evolution Parameters
    evol_info = get_evol_param("me")
    start_size = evol_info["start_size"]              # initial number of chromosomes to start MAP-Elites
    iterations = evol_info["iterations"]              # number of fitness evalutations for MAP-Elites
    mutation_length = evol_info["mutation_length"]    # how many tiles to mutate
    epsilon = evol_info["epsilon"]                    # probability of doing random mutation not from model
    periodic_save = evol_info["periodic_save"]        # how often to save

    train_info = get_train_param()
    # Data Creation Parameter
    max_size = max(width,height)
    window_size = int(args.observation*max_size)      # cropped view of the observation (can be any value)
    if args.type == EVOL_DATA or args.type == INBET_DATA or args.type == INIT_DATA:
        data_creation = args.type                     # method of creating the data
    else:
        raise TypeError(f"{args.type} is not one of the appropriate data creation types (evol, inbet, init)")
    portion = int(train_info["portion"] *\
                    behavior_bins**num_behaviors)     # amount of chromosomes used to train the network
    increase_data = train_info["increase_data"]       # increase data size by that value (doesn't work with EVOL_DATA unless early_threshold < 1)
    append_data = train_info["append_data"]           # new data is generated beside old ones
    early_threshold = train_info["early_threshold"]   # threshold where a level is considered good enough when reach that level

    # Training Parameters
    allow_train = args.train                          # allow training neural network
    train_epochs = train_info["train_epochs"]         # number of epochs used in the middle of evolution
    train_period = int(train_info["train_interval"]*\
                        iterations)                   # frequency of training the network
    batch_size = train_info["batch_size"]             # minibatch size during training
    learning_rate = train_info["learning_rate"]       # optimizer learning rate
    reset_model = train_info["reset_model"]           # reset the model weights
    optimizer_fn = optim.Adam                         # used optimizer
    loss_fn = nn.CrossEntropyLoss                     # used loss function

    # Model Parameter
    conditional = args.conditional                    # is the model conditional or not
    mutate_type = SOFTMAX_ACT                         # type of mutation when using the network

    # extra parameters
    num_experiments = args.number                     # number of experiments to run
    save_folder = f"{game_name}_{data_creation}_{train_epochs}_{window_size}_{['noncond','cond'][conditional]}_{['normal','assisted'][allow_train]}"

    for experiment in range(num_experiments):
        if conditional:
            cond_length = num_behaviors
            target = random_target
        else:
            cond_length = 0
            target = None
        model = SMNN(window_size, num_tiles, cond_length, mutate_type)
        evolver = SMME(start_size, width, height, model, init, fitness, behaviors, behavior_bins, target)
        total_levels, total_targets, total_actions = np.array([]).reshape((0,window_size,window_size)),\
                                                     np.array([]).reshape((0,num_behaviors)), np.array([]).reshape((0))
        pbar = trange(iterations)
        for i in pbar:
            evolver.update(epsilon, mutation_length)
            pbar.set_postfix_str(f"Map Size: {len(evolver)}")
            model_changed = False
            if (allow_train and i > 0 and i % train_period == 0) or (i == iterations-1):
                if reset_model:
                    model.reset_parameters()
                optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
                loss = loss_fn()
                if data_creation == EVOL_DATA and evolver.get_best().fitness() <= early_threshold:
                    increase_data = 1
                archive = evolver.get_map()
                levels, targets, actions = extract_data(archive.get_all(archive.keys()), window_size, model._channels,\
                                                        data_creation, portion, behavior_bins, increase_data,\
                                                        early_threshold)
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

            if i % periodic_save == 0 or i == iterations - 1:
                import os
                import shutil
                model_path = f"results/me/{experiment}/{save_folder}/{i}"
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                os.makedirs(model_path)
                evolver.save(os.path.join(model_path, "archive"))
                if model_changed or i == 0:
                    save_model(model, model_path)
