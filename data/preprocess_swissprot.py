import graphein.protein as gp
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
import os

from egnn_clean import get_edges_batch

import json
import torch

from torch_geometric.utils import from_networkx


config_n = gp.ProteinGraphConfig(**{"granularity": "N"})
config_c = gp.ProteinGraphConfig(**{"granularity": "C"})
config_ca = gp.ProteinGraphConfig(**{"node_metadata_functions": [amino_acid_one_hot],"granularity": "CA"})

def pdb_to_tensors_big_atoms(path):
  # Returns 3 tensors, a dgl graph and a tensor of edges
  # g_CA_coords : coordinates of C alpha atoms Nx3
  # g_CA_one_hot : one hot encoding of the amino acid type Nx20
  # g_v: SO(3) vector of the amino acid Nx3

  g_N = gp.construct_graph(config=config_n, path=path)
  g_N_coords = from_networkx(g_N).graph_coords.to(dtype=torch.float32)


  g_C = gp.construct_graph(config=config_c, path=path)
  g_C_coords = from_networkx(g_C).graph_coords.to(dtype=torch.float32)


  g_CA_nx = gp.construct_graph(config=config_ca, path=path)
  g_CA = from_networkx(g_CA_nx)

  g_CA_coords = g_CA.graph_coords.to(dtype=torch.float32)
  one_hot = g_CA.amino_acid_one_hot

  coords=torch.cat((g_CA_coords,g_C_coords,g_N_coords),dim=-1) 
  #fully connected edges
  edges,_=get_edges_batch(coords.shape[0],1)
  return coords, one_hot,edges


saved_ids=[]

directory_path = 'D:/swissprot_pdb_v4'

files = os.listdir(directory_path)
for file in files[22000:25000]:
  f_path = os.path.join(directory_path, file)
  if os.path.isfile(f_path) and file.endswith('.pdb'):  
    protid = file.split('-')[1]
    save_dict = {}
    save_dict['UNIPROT_ID'] = protid
    path=f_path
    g_CA_coords, g_CA_one_hot,edges = pdb_to_tensors_big_atoms(path)
    save_dict['num_nodes'] = g_CA_coords.shape[0]
    if save_dict['num_nodes'] <= 150 and save_dict['num_nodes'] >= 40:
      saved_ids.append(protid)
      save_dict['coords'] = g_CA_coords.tolist()
      save_dict['one_hot'] = g_CA_one_hot.tolist()
      save_dict['edges'] = [ten.tolist() for ten in edges]
      with open('./swiss_p/'+protid+'_tensors_'+str(save_dict['num_nodes'])+'.json', 'w') as file:
        json.dump(save_dict, file)
