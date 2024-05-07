# Open a json file called tm_scores.json and load it into a dict
import json

with open('tm_scores.json', 'r') as file:
    tm_scores = json.load(file)

# Load all the values into a list
tm_scores_list = list(tm_scores.values())
tm_scores_list = [float(i) for i in tm_scores_list]

# sort the list
tm_scores_list.sort(key = float)

# print(tm_scores_list)

# plot the list
import matplotlib.pyplot as plt


plt.plot(tm_scores_list)
plt.ylabel('Training TM Score')
plt.title('Training TM Scores of generated structures')
plt.show()