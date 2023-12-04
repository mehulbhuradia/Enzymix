import os
import json
import torch
from torch.utils.data import Dataset



class ProtienStructuresDataset(Dataset):
    def __init__(self, path='./processed'):
        self.paths = []
        for pdb in os.listdir(path):
          self.paths.append('./processed/' + pdb)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path=self.paths[idx]
        with open(file_path, 'r') as file:
          data = json.load(file)
        coords=torch.Tensor(data['coords'])
        one_hot=torch.Tensor(data['one_hot'])
        v = torch.Tensor(data['v'])
        edges=[]
        for e in data['edges']:
          edges.append(torch.tensor(e, dtype=torch.int64))
        return coords, one_hot, v, edges