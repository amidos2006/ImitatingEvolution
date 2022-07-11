import matplotlib.pyplot as plt
from matplotlib import animation
from moviepy.editor import ImageSequenceClip

from games import get_game

from nn.model import SOFTMAX_ACT, GREEDY_ACT, SMNN
from nn.helper import transform_input

import numpy as np
from tqdm import tqdm, trange

import os
import shutil
import argparse

import torch
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on trained Evolutionary Imitation')
    # parser.add_argument('model',
    #                     help='the model that we need to use for inference')
    parser.add_argument('--game', '-g', default="binary",
                        help='the game that we need to evolve and test (default: binary)')
    args = parser.parse_args()

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
    stopping = game_info["stopping"]                  # stopping criteria during inference
    render = game_info["render"]                      # render function for the level

    max_iterations = width * height
    input_size = 8
    conditional = False
    action_type = SOFTMAX_ACT                         # type of mutation when using the network
    random_order = False

    visualize = False
    number_times = 1
    model_path = "results/es/2/binary_evol_2_assisted/1999"
    animation_path = "results/animations"

    for i in range(number_times):
        print(f"Generating {i} Level: ")
        target = []
        for i in range(num_behaviors):
            target.append(np.random.randint(behavior_bins) / (1.0 * behavior_bins))
        channels = num_tiles
        if num_tiles <= 2:
            channels = 1

        model = torch.load(os.path.join(model_path, "model"))
        start = init(width, height)

        print(f"\tStart Fitness: {fitness(start, [])}")
        print(f"\tStart Fitness: {behaviors(start, [], behavior_bins)}")

        tiles = []
        for y in range(height):
            for x in range(width):
                tiles.append({"x": x, "y": y})

        frames = []
        level = start.copy()
        with torch.no_grad():
            for i in trange(max_iterations, leave=False):
                if random_order:
                    np.random.shuffle(tiles)
                change = False
                all_obs = []
                for t in tiles:
                    x,y = t["x"], t["y"]
                    obs = transform_input(level, {"x":x,"y":y}, input_size, channels)
                    all_obs.append(obs)
                all_obs = np.array(all_obs)
                if conditional:
                    values = model(torch.tensor(all_obs.reshape(-1,channels,input_size,input_size)).float(),\
                                       torch.tensor(np.array(target).reshape(1,-1).repeat(len(all_obs), axis=0)).float())
                else:
                    values = model(torch.tensor(all_obs.reshape(-1,channels,input_size,input_size)).float(), None)
                values = F.softmax(values, dim=1).numpy()
                for ai,t in enumerate(tiles):
                    x,y = t["x"], t["y"]
                    if action_type == SOFTMAX_ACT:
                        action = np.random.choice(list(range(num_tiles + 1)), p=values[ai].flatten())
                    else:
                        action = values[ai].argmax().item()
                    if action > 0:
                        level[y][x] = action - 1
                        change = True
                    frames.append(render(level.copy()))
                if stopping(level):
                    print(f"\n\tFinished: {i}")
                    break
                if not change and action_type != SOFTMAX_ACT:
                    print(f"\tStablized: {i}")
                    break

        print(f"\tFinal Fitness: {fitness(level, [])}")
        print(f"\tFinal Fitness: {behaviors(level, [], behavior_bins)}")

        if visualize:
            print("\tAnimating...")
            for i in range(20):
                frames.append(frames[-1].copy())

            def init_display():
                im.set_data(start)
                return [im]

            def animate_display(i):
                im.set_data(frames[i])
                return [im]

            fig = plt.figure()
            plt.axis('off')
            im = plt.imshow(start)
            plt.tight_layout()

            anim = animation.FuncAnimation(fig, animate_display, init_func=init_display,\
                                           frames=len(frames), interval=15, blit=True)
            plt.close(anim._fig)

            # Call function to display the animation
            if not os.path.exists(animation_path):
                os.makedirs(animation_path)
            prev_anims = [f for f in os.listdir(animation_path) if "animation" in f]
            final_path = os.path.join(animation_path, f"animation_{len(prev_anims)}.mp4")
            anim.save(final_path)
            print(f"\tAnimation is in: {final_path}")
