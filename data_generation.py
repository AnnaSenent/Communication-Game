"""
This script generates a data set in the format expected by the sum game
"""

import numpy as np

np.random.seed(123)

N = 20
N_INPUT = 200
N_INTEGERS = 2

integers = np.random.randint(1, N + 1, N_INPUT * N_INTEGERS)
integers_to_matrix = np.reshape(integers, (N_INPUT, N_INTEGERS))
labels = np.sum(integers_to_matrix, axis=1)

# Save to txt file
data = np.hstack([integers_to_matrix, labels.reshape(-1, 1)])
np.savetxt('data/validation.txt', data, fmt='%d')
