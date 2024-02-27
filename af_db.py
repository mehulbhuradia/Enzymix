import os
import json
import torch
from torch.utils.data import Dataset
from egnn_clean import get_edges_batch

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
    coords=torch.tensor(data['coords'], dtype=torch.float32)
    one_hot=torch.tensor(data['one_hot'], dtype=torch.float32)
    g_CA_coords,g_C_coords,g_N_coords = torch.chunk(coords, chunks=3, dim=-1)
    stacked_tensor = torch.cat((g_CA_coords,g_C_coords,g_N_coords),dim=-2)
    edges,_=get_edges_batch(stacked_tensor.shape[0],1)
    one_hot_CA = torch.tensor([1, 0, 0]).unsqueeze(0).expand(one_hot.shape[-2],-1)
    one_hot_C = torch.tensor([0, 1, 0]).unsqueeze(0).expand(one_hot.shape[-2],-1)
    one_hot_N = torch.tensor([0, 0, 1]).unsqueeze(0).expand(one_hot.shape[-2],-1)
    # one_hot_CA = torch.cat((one_hot,one_hot_CA),dim=-1)
    # one_hot_C = torch.cat((one_hot,one_hot_C),dim=-1)
    # one_hot_N = torch.cat((one_hot,one_hot_N),dim=-1)
    one_hot = torch.cat((one_hot,one_hot,one_hot),dim=-2)
    one_hot_atoms = torch.cat((one_hot_CA,one_hot_C,one_hot_N),dim=-2)
    return stacked_tensor, one_hot, edges, file_path, one_hot_atoms