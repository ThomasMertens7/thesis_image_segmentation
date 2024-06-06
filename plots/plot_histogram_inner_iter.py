import numpy as np
import matplotlib.pyplot as plt

# Example data
data = [
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 20, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        19, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 23, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 29, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 29, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 11, 15,
        15, 15, 19, 15, 15, 15, 15, 29, 29, 29, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 16
]

mean = np.mean(data)
# Creating the histogram
plt.hist(data, bins=10, edgecolor='black', alpha=0.7)

# Adding a vertical line at x = 3.3
plt.axvline(x=mean, color='red', linestyle='--', label='gemiddelde')

# Customizing the plot
plt.title('Data verdeling inner_iteraties')
plt.xlabel('Waarde')
plt.ylabel('Frequentie')
plt.legend()

# Show the plot
plt.show()
