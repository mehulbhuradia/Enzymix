import json
import torch

from af_db import ProtienStructuresDataset
from torch.utils.data import DataLoader

dataset = ProtienStructuresDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

coords, one_hot, v, edges = next(iter(train_dataloader))

print(coords.shape)
print(one_hot.shape)
print(v.shape)
print(len(edges))
print(edges[0].shape)

# with open('./processed/A1RWP0_tensors_609.json', 'r') as file:
#   data = json.load(file)

# print(data.keys())

# print(torch.Tensor(data['coords']))

# import os
# paths = []
# for pdb in os.listdir('./processed'):
#   paths.append('./processed/' + pdb)
# # print(paths)


# with open(paths[0], 'r') as file:
#   data = json.load(file)

# print(data.keys())

