import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================
arr = np.array([[1, 2, 3], [4, 5, 6]])
print('Array type is', type(arr))
print(arr)

# ==================

data = np.array(['S', 'U', 'L', 'A', 'G', 'N', 'A'])
ser = pd.Series(data)
print(ser)

# ==================
a = [10, 20, 30]
b = [50, 60, 70]
plt.plot(a, b)
plt.title("Simple Plot")
plt.show()