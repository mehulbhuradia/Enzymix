import os

path='./processed_big_atoms'
max_len=600
paths = []
di={}
for pdb in os.listdir(path):
    length=int(pdb.split('_')[2].split('.')[0])
    if int(length) <= max_len:
        di[length]=di.get(length,0)+1

di=dict(sorted(di.items()))
print(di)