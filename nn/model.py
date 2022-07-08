import torch
from torch import nn
import torch.nn.functional as F
from .helper import transform_input

GREEDY_ACT = "greedy"
SOFTMAX_ACT = "softmax"

class SMNN(nn.Module):
    def __init__(self, size, tiles, length = 0, type = SOFTMAX_ACT):
        super(SMNN, self).__init__()

        self._nocond = length == 0
        self._size = size
        self._channels = tiles
        if tiles <= 2:
            self._channels = 1
        self._outputs = tiles + 1
        self._type = type

        self._conv1 = nn.Conv2d(self._channels, 32, 3, padding='same')
        self._max1 = nn.MaxPool2d(2)
        self._conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self._max2 = nn.MaxPool2d(2)
        self._conv3 = nn.Conv2d(64, 128, 3, padding='same')
        input_values = int(size / 4 * size / 4 * 128  + length)
        self._linear1 = nn.Linear(input_values, 256)
        self._linear2 = nn.Linear(256, self._outputs)

    def forward(self, x, t):
        x = F.relu(self._max1(self._conv1(x)))
        x = F.relu(self._max2(self._conv2(x)))
        x = F.relu(self._conv3(x))
        x = x.view(x.shape[0],-1)
        if not self._nocond:
            x = torch.cat([x, t], 1)
        x = F.relu(self._linear1(x))
        x = self._linear2(x)
        return x

    def mutate(self, level, x, y, target):
        with torch.no_grad():
            c_lvl = transform_input(level, {"x":x, "y":y}, self._size, self._channels)
            if self._nocond:
                values = self(torch.tensor(c_lvl.copy().reshape(1,self._channels,self._size,self._size)).float(), None)
            else:
                values = self(torch.tensor(c_lvl.copy().reshape(1,self._channels,self._size,self._size)).float(),\
                            torch.tensor(target.copy().reshape(1,-1)).float())
            values = F.softmax(values, dim=1).numpy()
            if type == SOFTMAX_ACT:
                value = np.random.choice(list(range(self._outputs)), p=values.flatten())
            else:
                value = values.argmax().item()
        return {"x": int(x), "y": int(y), "action": int(value)}

    def reset_parameters(self):
        self._conv1.reset_parameters()
        self._conv2.reset_parameters()
        self._conv3.reset_parameters()
        self._linear1.reset_parameters()
        self._linear2.reset_parameters()
