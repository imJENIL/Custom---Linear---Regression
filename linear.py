import numpy as np
import matplotlib as plt
from linear_custom import Linear

x = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10, 11])

model = Linear(learning_rate= 0.01, n_iterations=1000)
model.fit(x, y)

y_pred = model.pred(x)

print(y_pred)