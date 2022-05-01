
import numpy as np

np.random.seed(123)

N = 20
n_input = 100
n_integers_to_sum = 2

integers = np.random.randint(0, N, n_input * n_integers_to_sum)
integers_to_matrix = np.reshape(integers, (n_input, n_integers_to_sum))
labels = np.sum(integers_to_matrix, axis=1)

# Save to txt file

file = open('data/dataset.txt', 'w')
for integers, label in zip(integers_to_matrix, labels):
    data = np.concatenate((integers, np.reshape(label, (1,)))).reshape((1, 3))
    np.savetxt(file, data, fmt='%d')
file.close()