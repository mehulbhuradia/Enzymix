import graphein.protein as gp
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot

from diffab.modules.common.geometry import construct_3d_basis
from diffab.modules.common.so3 import rotation_to_so3vec

from egnn_clean import get_edges_batch

import json
import torch

from torch_geometric.utils import from_networkx


config_n = gp.ProteinGraphConfig(**{"granularity": "N"})
config_c = gp.ProteinGraphConfig(**{"granularity": "C"})
config_ca = gp.ProteinGraphConfig(**{"node_metadata_functions": [amino_acid_one_hot],"granularity": "CA"})

def pdb_to_tensors(path):
  # Returns 3 tensors, a dgl graph and a tensor of edges
  # g_CA_coords : coordinates of C alpha atoms Lx3
  # g_CA_one_hot : one hot encoding of the amino acid type Lx20
  # g_v: SO(3) vector of the amino acid Lx3
  # edges: list of 2 tensors of shape  E
  
  g_N = gp.construct_graph(config=config_n, path=path)
  g_N_coords = from_networkx(g_N).graph_coords.to(dtype=torch.float32)

  g_C = gp.construct_graph(config=config_c, path=path)
  g_C_coords = from_networkx(g_C).graph_coords.to(dtype=torch.float32)

  g_CA_nx = gp.construct_graph(config=config_ca, path=path)
  g_CA = from_networkx(g_CA_nx)

  g_CA_coords = g_CA.graph_coords.to(dtype=torch.float32)
  g_CA_one_hot = g_CA.amino_acid_one_hot

  g_R = construct_3d_basis(g_CA_coords,g_C_coords,g_N_coords)

  g_v = rotation_to_so3vec(g_R)

  #fully connected edges
  edges,_=get_edges_batch(g_CA_coords.shape[0],1)
  return g_CA_coords, g_CA_one_hot,g_v,edges


with open('downloaded_from_alpha.json', 'r') as file:
  data = json.load(file)

saved_ids=[]

for protid in data:
  protid = protid.upper()
  save_dict = {}
  save_dict['UNIPROT_ID'] = protid
  path="./af_structures/"+protid+".pdb"
  g_CA_coords, g_CA_one_hot,g_v,edges = pdb_to_tensors(path)
  save_dict['num_nodes'] = g_CA_coords.shape[0]
  if save_dict['num_nodes'] < 600:
    saved_ids.append(protid)
    save_dict['coords'] = g_CA_coords.tolist()
    save_dict['one_hot'] = g_CA_one_hot.tolist()
    save_dict['v'] = g_v.tolist()
    save_dict['edges'] = [ten.tolist() for ten in edges]
    with open('./processed/'+protid+'_tensors_'+str(save_dict['num_nodes'])+'.json', 'w') as file:
      json.dump(save_dict, file)

with open('processed_ids.json', 'w') as file:
  json.dump(saved_ids, file)
