import csv
import matplotlib.pyplot as plt
# Specify the path to your CSV file
# csv_file_path = '2x24_val.csv'

import json
import numpy as np
with open('data_water.json', 'r') as file:
    data = json.load(file)

with open('data.json', 'r') as file:
    data2 = json.load(file)

data.extend(data2)
reactions={}

for prot in data:
    if prot['ec'] not in reactions:
        reactions[prot['ec']]=[]
    reactions[prot['ec']].append(prot['uniprot'])

for key in reactions:
    reactions[key]=set(reactions[key])

# print(reactions)
# Open the CSV file
# with open(csv_file_path, 'r') as file:
#     # Create a CSV reader object
#     csv_reader = csv.reader(file)
    
#     # Read all rows into a list
#     rows = list(csv_reader)

#     # Iterate through each row and print the value in the first column
#     for row in rows:
#         if row:  # Check if the row is not empty
#             prot=row[0].split("'")[1]
#             total_similar_ec=0
#             for key in reactions:
#                 if prot in reactions[key]:
#                     total_similar_ec+=len(reactions[key])
#             row.append(total_similar_ec)        

# # Write the updated data back to the CSV file
# with open(csv_file_path, 'w', newline='') as file:
#     # Create a CSV writer object
#     csv_writer = csv.writer(file)
    
#     # Write the updated rows to the file
#     csv_writer.writerows(rows)

for key in reactions:
    reactions[key]=len(reactions[key])

D = reactions

D_sorted = dict(sorted(D.items(), key=lambda item: item[1]))

plt.plot(range(len(D_sorted)), list(D_sorted.values()))
plt.axhline(y=np.mean(list(D_sorted.values())), color='b', linestyle='--', linewidth=2, label='Average:'+str(np.mean(list(D_sorted.values()))))
plt.legend()
plt.show()
