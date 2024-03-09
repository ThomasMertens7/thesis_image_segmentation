import pandas as pd

# Define an empty list to store the data
data = []

# Open the text file
with open('results.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Split the line by commas
        line_data = line.strip().split(', ')
        # Append the split line to the data list
        data.append(line_data)

for line in data:
    line[1] = line[1] + ", " + line[2] + ", " + line[3] + ", " + line[4]
    del line[2]
    del line[2]
    del line[2]


for row in data:
    print(row)
    print(len(row))


#Colummns
columns = ['image', 'initial_contour', 'edge_indicator', 'alpha', 'sigma', 'lambda', 'precision']

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Write the DataFrame to an Excel file
df.to_excel('data.xlsx', index=False)