import numpy as np
import matplotlib.pyplot as plt

data = [
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 35, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 35,
    30, 30, 35, 30, 30, 30, 30, 30, 30, 30, 34, 38, 34, 42, 34, 30, 42, 34, 34, 34,
    34, 42, 34, 30, 38, 34, 34, 30, 34, 42, 34, 30, 30, 58, 34, 34, 30, 34, 30, 34,
    30, 38, 34, 54, 30, 34, 30, 58, 30, 50, 34, 34, 42, 46, 30, 34, 30, 38, 30, 46,
    50, 30, 34, 30, 58, 34, 30, 34, 30, 58, 58, 58, 34, 38, 30, 34, 42, 46, 42, 30,
    38, 30, 30, 30, 39, 32, 39, 33, 34
]

mean = np.mean(data)
# Creating the histogram
plt.hist(data, bins=10, edgecolor='black', alpha=0.7)

# Adding a vertical line at x = 3.3
plt.axvline(x=mean, color='red', linestyle='--', label='gemiddelde')

# Customizing the plot
plt.title('Data verdeling outer_iteraties')
plt.xlabel('Waarde')
plt.ylabel('Frequentie')
plt.legend()

# Show the plot
plt.show()
