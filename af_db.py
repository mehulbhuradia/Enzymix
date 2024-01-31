import os
import json
import torch
from torch.utils.data import Dataset

class ProtienStructuresDataset(Dataset):
  def __init__(self, path='./processed_big_atoms',max_len=300):
    self.paths = []
    for pdb in os.listdir(path):
      length=pdb.split('_')[2].split('.')[0]
      if int(length) <= max_len:
        self.paths.append(path+"/" + pdb)

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):
    file_path=self.paths[idx]
    with open(file_path, 'r') as file:
      data = json.load(file)
    coords=torch.Tensor(data['coords'])
    one_hot=torch.Tensor(data['one_hot'])
    edges=[]
    for e in data['edges']:
      edges.append(torch.tensor(e, dtype=torch.int64))
    return coords, one_hot, 0, edges, file_path
  
  def get_item_by_uniprotid(self, uniprotid):
    for path in self.paths:
      if uniprotid in path:
        with open(path, 'r') as file:
          data = json.load(file)
        coords=torch.Tensor(data['coords'])
        one_hot=torch.Tensor(data['one_hot'])
        edges=[]
        for e in data['edges']:
          edges.append(torch.tensor(e, dtype=torch.int64))
        return coords, one_hot, 0, edges, path