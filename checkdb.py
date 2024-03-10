from af_db import ProtienStructuresDataset
import torch
from egnn_clean import get_edges_batch

dataset = ProtienStructuresDataset()

# print(len(dataset[0]))
one_hot,edges = dataset[0]

print(one_hot.shape)
print(edges[0].shape)



# print(get_edges_batch(4,2))

# print(e1[0].shape)
# print(e2.shape)


# g_CA_coords,g_C_coords,g_N_coords = torch.chunk(coords, chunks=3, dim=-1)
# stacked_tensor = torch.cat((g_CA_coords,g_C_coords,g_N_coords),dim=-2)
# edges,_=get_edges_batch(stacked_tensor.shape[0],1)
# one_hot_CA = torch.tensor([1, 0, 0]).unsqueeze(0).expand(one_hot.shape[-2],-1)
# one_hot_C = torch.tensor([0, 1, 0]).unsqueeze(0).expand(one_hot.shape[-2],-1)
# one_hot_N = torch.tensor([0, 0, 1]).unsqueeze(0).expand(one_hot.shape[-2],-1)
# one_hot_CA = torch.cat((one_hot,one_hot_CA),dim=-1)
# one_hot_C = torch.cat((one_hot,one_hot_C),dim=-1)
# one_hot_N = torch.cat((one_hot,one_hot_N),dim=-1)
# one_hot_atoms = torch.cat((one_hot_CA,one_hot_C,one_hot_N),dim=-2)

# stacked_tensor = torch.cat((tensor1, tensor2, tensor3), dim=1)