import numpy as np
import torch
from tqdm import trange, tqdm

def train(model, optimizer, loss_fn, epochs, batch, levels, targets, actions):
    epbar = trange(epochs, leave=False)
    for epoch in epbar:
        order = np.array(range(len(actions)))
        np.random.shuffle(order)
        in_lvls, in_t, in_act = levels[order], targets[order], actions[order]
        pbar = trange(len(in_act) - batch, leave=False)
        total_loss = 0
        for i in pbar:
            batch_lvls, batch_t, batch_act = in_lvls[i:i+batch], in_t[i:i+batch], in_act[i:i+batch]
            optimizer.zero_grad()
            if model._nocond:
                batch_output = model(torch.tensor(batch_lvls.reshape(batch, model._channels, model._size, model._size)).float(), None)
            else:
                batch_output = model(torch.tensor(batch_lvls.reshape(batch, model._channels, model._size, model._size)).float(),\
                      torch.tensor(batch_t.reshape(batch, batch_t.shape[1])).float())
            loss = loss_fn(batch_output, torch.tensor(batch_act).long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix_str(f"Loss: {total_loss / (i + 1.0)}")
        if total_loss == 0:
            epbar.close()
            break
