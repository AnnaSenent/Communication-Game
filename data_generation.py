"""
This script generates a data set in the format expected by the sum game
"""

import numpy as np

np.random.seed(123)

N = 20
N_INPUT = 25000
N_INTEGERS = 2

integers = np.random.randint(0, N, N_INPUT * N_INTEGERS)
integers_to_matrix = np.reshape(integers, (N_INPUT, N_INTEGERS))
labels = np.sum(integers_to_matrix, axis=1)

# Save to txt file
train_data, train_labels, val_data, val_labels = integers_to_matrix[:int(N_INPUT*0.8)], \
                                                 labels[:int(N_INPUT*0.8)], \
                                                 integers_to_matrix[int(N_INPUT*0.8):], \
                                                 labels[int(N_INPUT*0.8):]

train = np.hstack([train_data, train_labels.reshape(-1, 1)])
val = np.hstack([val_data, val_labels.reshape(-1, 1)])

np.savetxt('data/train.txt', train, fmt='%d')
np.savetxt('data/validation.txt', val, fmt='%d')
