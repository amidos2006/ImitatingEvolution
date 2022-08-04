from tqdm import trange, tqdm
from multiprocessing import Pool, current_process
from functools import partial

import argparse
import numpy as np
import json
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim

from games import get_game

from evolution import get_evol_param
from evolution.helper import random_target
from evolution.es import SMES

from nn import get_train_param, save_model
from nn.model import SOFTMAX_ACT, GREEDY_ACT, SMNN
from nn.trajectory import EVOL_DATA, INIT_DATA, INBET_DATA, extract_data
from nn.train import train

def run_experiment(experiment, game_name, game_info, evol_info, train_info, args):
    # game parameters
    width = game_info["width"]                        # width of the generated level
    height = game_info["height"]                      # height of the generated level
    num_tiles = game_info["num_tiles"]                # number of tiles in the level
    num_behaviors = game_info["num_behaviors"]        # number of behavior characteristic
    behavior_bins = game_info["behavior_bins"]        # number of bins after discretize
    init = game_info["init"]                          # initialization function for problem
    fitness = game_info["fitness"]                    # fitness function for the problem
    behaviors = game_info["behaviors"]                # behavior characteristic function for the problem

    # evolution parameters
    pop_size = evol_info["pop_size"]                 # population size
    death_perct = evol_info["death_perct"]           # percentage of killed chromosomes
    tournment_size = evol_info["tournment_size"]     # size of tournment in tournment selection
    gen_number = evol_info["gen_number"]             # number of generations
    mutation_length = evol_info["mutation_length"]   # the maximum amount of tiles to mutate
    epsilon = evol_info["epsilon"]                   # probability of doing random mutation not from model
    periodic_save = evol_info["periodic_save"] *\
                                        gen_number   # how often to save

    # data creation parameters
    max_size = max(width,height)
    window_size = int(args.observation*max_size)      # cropped view of the observation (can be any value)
    if args.type == EVOL_DATA or args.type == INBET_DATA or args.type == INIT_DATA:
        data_creation = args.type                     # method of creating the data
    else:
        raise TypeError(f"{args.type} is not one of the appropriate data creation types (evol, inbet, init)")
    portion = int(train_info["portion"] * pop_size)   # amount of chromosomes used to train the network
    increase_data = train_info["increase_data"]       # increase data size by that value (doesn't work with EVOL_DATA unless early_threshold < 1)
    append_data = train_info["append_data"]           # new data is generated beside old ones
    early_threshold = train_info["early_threshold"]   # threshold where a level is considered good enough when reach that level

    # training parameters
    allow_train = args.train                          # allow training neural network
    train_epochs = train_info["train_epochs"]         # number of epochs used in the middle of evolution
    train_period = int(train_info["train_interval"]*\
                        gen_number)                   # frequency of training the network
    batch_size = train_info["batch_size"]             # minibatch size during training
    learning_rate = train_info["learning_rate"]       # optimizer learning rate
    reset_model = train_info["reset_model"]           # reset the model weights
    optimizer_fn = optim.Adam                         # used optimizer
    loss_fn = nn.CrossEntropyLoss                     # used loss function

    # model parameters
    conditional = args.conditional                    # is the model conditional or not
    mutate_type = SOFTMAX_ACT                         # type of mutation when using the network

    # extra parameters
    save_folder = f"{game_name}_{data_creation}_{train_epochs}_{window_size}_{['noncond','cond'][conditional]}_{['normal','assisted'][allow_train]}"


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

    current = current_process()
    if len(current._identity) == 0:
        pbar = trange(gen_number)
    else:
        pbar = trange(gen_number, position=current._identity[0] - 1)
    # pbar = atpbar(range(gen_number), name=mp.current_process().name)
    for i in pbar:
        evolver.update(death_perct, tournment_size, epsilon, mutation_length)
        pbar.set_postfix_str(f"Best Fitness: {evolver.get_best().fitness()}")
        model_changed = False
        if (allow_train and i > 0 and i % train_period == 0) or (i == gen_number-1):
            if reset_model:
                model.reset_parameters()
            optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
            loss = loss_fn()
            if data_creation == EVOL_DATA and evolver.get_best().fitness() <= early_threshold:
                increase_data = 1
            levels, targets, actions = extract_data(evolver.get_pop().get_all(), window_size, model._channels,\
                                                    data_creation, portion, behavior_bins,\
                                                    increase_data, early_threshold)
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
                save_model(model, model_path, game_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Evolutionary Imitation using Mu+Lambda ES')
    parser.add_argument('--config', '-c', default="",
                        help='all the settings in a json file')
    parser.add_argument('--game', '-g', default="binary",
                        help='the game that we need to evolve and test (default: binary)')
    parser.add_argument('--observation', '-o', type=float, default=0.5,
                        help='the observation window size as percentage of biggest dimension (default: 0.5)')
    parser.add_argument('--number', '-n', type=int, default=3,
                        help='number of time to repeat the experiment (default: 3)')
    parser.add_argument('--train', action="store_true",
                        help='allow to train in the middle of evolution')
    parser.add_argument('--no-train', dest="train", action="store_false")
    parser.set_defaults(train=True)
    parser.add_argument('--conditional', action="store_true",
                        help='train a conditional network instead of normal one')
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

    # info grabbing
    game_name = args.game                             # name of the problems for saving purposes
    game_info = get_game(game_name)
    evol_info = get_evol_param("es")
    workers = evol_info["workers"]                   # number of concurrent experiments to run
    train_info = get_train_param()
    num_experiments = args.number                     # number of experiments to run

    experiment_partial = partial(run_experiment,
        game_name=game_name,
        game_info=game_info,
        evol_info=evol_info,
        train_info=train_info,
        args=args
    )
    if workers > 1:
        with Pool(workers) as p:
            p.map(experiment_partial, [i for i in range(num_experiments)])
            flush()
    else:
        experiment_partial(0)
