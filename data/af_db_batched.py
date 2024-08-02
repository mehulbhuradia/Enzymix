import os
import json
import torch
from torch.utils.data import Dataset
from model.egnn_complex import get_edges_batch

def split_array(input_array, chunk_size):
  return [input_array[i:i + chunk_size] for i in range(0, len(input_array), chunk_size)]

class ProtienStructuresDataset(Dataset):
  def __init__(self, path='./processed_big_atoms',max_len=100,min_len=50,batch_size=8,only_ca=False,test_size=0):
    self.only_ca = only_ca
    self.lengths = []
    length_map = {}
    self.batches = []
    self.test_set = []
    count = 0
    # Exclusing the first 100 pdb files as the test set
    for pdb in os.listdir(path):
      count += 1
      if count < test_size:
        self.test_set.append(path + "/" + pdb)
        continue
      length=pdb.split('_')[2].split('.')[0]
      if int(length) <= max_len and int(length) >= min_len:
        self.lengths.append(int(length))
        length_map[int(length)] = length_map.get(int(length), []) + [path + "/" + pdb]
    for key in length_map:
      self.batches.extend(split_array(length_map[key], batch_size))

  def __len__(self):
    return len(self.batches)

  def __getitem__(self, idx):
    batch = self.batches[idx]
    one_hots = []
    coords = []
    for file_path in batch:  
      with open(file_path, 'r') as file:
        data = json.load(file)
      if self.only_ca:
        coords.append(torch.tensor(data['coords'], dtype=torch.float32)[:, :3]) # only carbon alpha atoms
      else:
        coords.append(torch.tensor(data['coords'], dtype=torch.float32))
      one_hots.append(torch.tensor(data['one_hot'], dtype=torch.float32))
    res_len = one_hots[0].shape[0]
    one_hot = torch.cat(one_hots, dim=0)
    coords = torch.cat(coords, dim=0)
    edges,_=get_edges_batch(res_len,len(batch))
    return coords, one_hot, edges, len(batch)
  
  def size(self):
    return len(self.lengths)
  
  def get_test_set(self):
    return self.test_set