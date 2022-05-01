# Convert to one-hot format

one_hot_data = np.empty((n_input, N*n_integers_to_sum))
for i in range(len(integers_to_matrix)):
    initialize = np.zeros(N*n_integers_to_sum)
    initialize[integers_to_matrix[0][0]] = 1
    initialize[integers_to_matrix[0][1] + N] = 1
    one_hot_data[i] = initialize

one_hot_labels = np.empty((n_input, labels.max()+1))
for i in range(len(labels)):
    initialize = np.zeros((labels.max()+1))
    initialize[labels[i]] = 1
    one_hot_labels[i] = initialize