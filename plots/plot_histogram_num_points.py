import numpy as np
import matplotlib.pyplot as plt

# Example data
data = [
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 75, 50, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 138, 100,
    62, 113, 100, 100, 100, 100, 100, 150, 100, 100, 100, 100, 100, 100, 138, 100,
    100, 112, 100, 162, 100, 100, 100, 75, 50, 100, 100, 100, 100, 150, 100, 112,
    100, 100, 100, 100, 100, 100, 100, 138, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 96, 110, 101, 116
]


mean = np.mean(data)
# Creating the histogram
plt.hist(data, bins=10, edgecolor='black', alpha=0.7)

# Adding a vertical line at x = 3.3
plt.axvline(x=mean, color='red', linestyle='--', label='gemiddelde')

# Customizing the plot
plt.title('Data verdeling num_points')
plt.xlabel('Waarde')
plt.ylabel('Frequentie')
plt.legend()

# Show the plot
plt.show()
