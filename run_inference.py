import matplotlib.pyplot as plt
from matplotlib import animation
from moviepy.editor import ImageSequenceClip

import games.binary as binary
from games.helper import get_number_regions

from nn.model import SOFTMAX_ACT, GREEDY_ACT, SMNN
import numpy as np
from nn.helper import transform_input

import os
import shutil

import torch
import torch.nn.functional as F

if __name__ == "__main__":
    width = 14
    height = 14
    num_tiles = 2                                     # number of tiles in the level
    num_behaviors = 2                                 # number of behavior characteristic
    behavior_bins = 20                                # number of bins after discretize
    init = binary.init
    fitness = binary.fitness
    behaviors = binary.behaviors

    max_iterations = width * height
    input_size = 8
    conditional = False
    action_type = GREEDY_ACT                         # type of mutation when using the network
    random_order = False

    model_path = "results/es/0/binary_evol_2_assitant/1500"
    animation_path = "results/animations"


    target = []
    for i in range(num_behaviors):
        target.append(np.random.randint(behavior_bins) / (1.0 * behavior_bins))
    channels = num_tiles
    if num_tiles <= 2:
        channels = 1

    model = torch.load(os.path.join(model_path, "model"))
    start = init(width, height)

    print(f"Start Fitness: {fitness(start, [])}")
    print(f"Start Fitness: {behaviors(start, [], behavior_bins)}")

    tiles = []
    for y in range(height):
        for x in range(width):
            tiles.append({"x": x, "y": y})

    frames = []
    done = False
    level = start.copy()
    with torch.no_grad():
        for i in range(max_iterations):
            if random_order:
                np.random.shuffle(tiles)
            change = False
            for t in tiles:
                x,y = t["x"], t["y"]
                obs = transform_input(level, {"x":x,"y":y}, input_size, channels)
                if conditional:
                    values = model(torch.tensor(obs.reshape(1,channels,input_size,input_size)).float(),\
                                   torch.tensor(np.array(target).reshape(1,-1)).float())
                else:
                    values = model(torch.tensor(obs.reshape(1,channels,input_size,input_size)).float(), None)
                values = F.softmax(values, dim=1).numpy()
                if action_type == SOFTMAX_ACT:
                    action = np.random.choice(list(range(num_tiles + 1)), p=values.flatten())
                else:
                    action = values.argmax().item()
                if action > 0:
                    level[y][x] = action - 1
                    change = True

                frames.append(binary.render(level.copy()))
                if get_number_regions(level, [1]) == 1:
                    print(f"One Region: {i}")
                    done = True
                    break
            if done:
                break
            if not change and action_type != SOFTMAX_ACT:
                print(f"Stablized: {i}")
                break

    print(f"Final Fitness: {fitness(level, [])}")
    print(f"Final Fitness: {behaviors(level, [], behavior_bins)}")

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
    anim.save(os.path.join(animation_path, f"animation_{len(prev_anims)}.mp4"))
    # save_gif("test.gif", np.array(frames), 60, 50)
