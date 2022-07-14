import json
import os
import torch

def get_train_param():
    if not os.path.exists(f"config/train/train.json"):
        raise FileNotFoundError(f'config/train/train.json is missing.')
    if not os.path.exists(f"config/train/trajectory.json"):
        raise FileNotFoundError(f'config/train/trajectory.json is missing.')
    with open(f"config/train/train.json") as f:
        train = json.load(f)
    with open(f"config/train/trajectory.json") as f:
        traj = json.load(f)
    result = {}
    for key in train:
        result[key] = train[key]
    for key in traj:
        result[key] = traj[key]
    return result

def save_model(model, model_path, game_name):
    with open(os.path.join(model_path, "extra.json"), 'w') as f:
        extra_params = {}
        extra_params["conditional"] = not model._nocond
        extra_params["size"] = model._size
        extra_params["channels"] = model._channels
        extra_params["outputs"] = model._outputs
        extra_params["type"] = model._type
        extra_params["game"] = game_name
        f.write(json.dumps(extra_params))
    torch.save(model, os.path.join(model_path, "model"))

def load_model(model_path):
    model = torch.load(os.path.join(model_path, "model"))
    with open(os.path.join(model_path, "extra.json")) as f:
        extra_params = json.load(f)
        model._nocond = not extra_params["conditional"]
        model._size = extra_params["size"]
        model._channels = extra_params["channels"]
        model._outputs = extra_params["outputs"]
        model._type = extra_params["type"]
    return model, extra_params["game"]
