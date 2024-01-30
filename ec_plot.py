import json
import numpy as np
with open('data.json', 'r') as file:
    data = json.load(file)


reactions={}
for prot in data:
    reactions[prot['ec']]=reactions.get(prot['ec'],0)+1
import matplotlib.pyplot as plt

D = reactions

D_sorted = dict(sorted(D.items(), key=lambda item: item[1]))

plt.plot(range(len(D_sorted)), list(D_sorted.values()))
plt.axhline(y=np.mean(list(D_sorted.values())), color='b', linestyle='--', linewidth=2, label='Average:'+str(np.mean(list(D_sorted.values()))))
plt.legend()
plt.show()

# print((reactions))    