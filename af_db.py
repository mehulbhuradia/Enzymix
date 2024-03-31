import os
import json
import torch
from torch.utils.data import Dataset

class ProtienStructuresDataset(Dataset):
  def __init__(self, path='./processed_big_atoms',max_len=300, min_len=50):
    self.paths = []
    for pdb in os.listdir(path):
      length=pdb.split('_')[2].split('.')[0]
      if int(length) <= max_len and int(length) > min_len:
        self.paths.append(path+"/" + pdb)

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):
    file_path=self.paths[idx]
    with open(file_path, 'r') as file:
      data = json.load(file)
    coords=torch.tensor(data['coords'], dtype=torch.float32)
    one_hot=torch.tensor(data['one_hot'], dtype=torch.float32)
    coords = coords[:, :3]
    edges=[]
    for e in data['edges']:
      edges.append(torch.tensor(e, dtype=torch.int64))
    return coords, one_hot, edges, file_path
  
  def get_item_by_uniprotid(self, uniprotid):
    for path in self.paths:
      if uniprotid in path:
        with open(path, 'r') as file:
          data = json.load(file)
        coords=torch.Tensor(data['coords'])
        one_hot=torch.Tensor(data['one_hot'])
        coords = coords[:, :3]
        edges=[]
        for e in data['edges']:
          edges.append(torch.tensor(e, dtype=torch.int64))
        return coords, one_hot, edges, path