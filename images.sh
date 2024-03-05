#!/bin/bash

# Specify the path to your Python script
python_script="visall.py"

# Specify the number of iterations for the loop
num_iterations=10  # You can change this to the desired number of iterations

# Loop through the specified number of iterations
for ((i=1; i<=$num_iterations; i++)); do
    echo "Running iteration $i"
    
    # Run the Python script with the --n argument and the current iteration
    python3 "$python_script" --n "$i"
    
    # Add any additional logic or delay between iterations if needed
    # For example, sleep 1 for a 1-second delay between iterations
    sleep 1
done

echo "Script completed."
