import os

path='./brenda_processed'

for pdb in os.listdir(path):
    length=pdb.split('_')[2].split('.')[0]
    if int(length) <= 50:
        print(pdb)
