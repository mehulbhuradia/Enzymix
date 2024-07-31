from af_db import ProtienStructuresDataset
from tqdm.auto import tqdm

import torch

# Initialize variables to keep track of the running sum and squared sum
running_sum = torch.zeros(9)
running_squared_sum = torch.zeros(9)
count = 0
running_max = torch.zeros(9) - float('inf')
running_min = torch.zeros(9) + float('inf')



dataset = ProtienStructuresDataset(path="./brenda_processed", max_len=600)

for i in tqdm(range(len(dataset))):
    coords, one_hot, edges, path = dataset[i]
    count += coords.shape[0]
    running_sum += torch.sum(coords, dim=0)
    running_squared_sum += torch.sum(coords ** 2, dim=0)
    running_max = torch.maximum(running_max, torch.max(coords, dim=0).values)
    running_min = torch.minimum(running_min, torch.min(coords, dim=0).values)

mean_values = running_sum / count
std_values = torch.sqrt((running_squared_sum / count) - (mean_values ** 2))

print("Mean:", mean_values)
print("Standard Deviation:", std_values)
print("Max:", running_max)
print("Min:", running_min)


# Mean: tensor([-0.7497,  0.5599, -0.2493, -0.7673,  0.5536, -0.2367, -0.7379,  0.5644,
#         -0.2577])
# Standard Deviation: tensor([14.9381, 12.4760, 15.7061, 14.9017, 12.4261, 15.6763, 14.9080, 12.4432,
#         15.6748])
# Max: tensor([121.7290, 113.0500, 175.3370, 121.4960, 112.1810, 174.9030, 121.5880,
#         114.4600, 174.5900])
# Min: tensor([-127.1380, -118.6200, -130.9770, -126.6150, -118.1410, -130.5090,
#         -127.1750, -118.6170, -130.0030])