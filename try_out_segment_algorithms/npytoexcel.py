import numpy as np
import pandas as pd

array = np.load('dog-1.npy')
df = pd.DataFrame(array)

df.to_excel('output.xlsx', index=False)

