import csv
import matplotlib.pyplot as plt

# Specify the CSV file name
csv_file_name = "2x24.csv"

# Read the CSV file and extract the values from the second column
with open(csv_file_name, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Get the values from the second column (assuming there are 24 values in the second row)
    second_column_values = [float(row[1]) for row in csv_reader]

# Sort the values
sorted_values = sorted(second_column_values)

# Plot the sorted values
plt.plot(sorted_values)
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Sorted Values from the Second Column')
plt.show()
