import numpy as np


def standardise_loop(data):
    """Standardise each column using nested Python loops."""
    values = np.asarray(data, dtype=float).tolist()
    rows = len(values)
    cols = len(values[0])

    means = []
    stds = []
    for j in range(cols):
        total = 0.0
        for i in range(rows):
            total += values[i][j]
        mean = total / rows

        sq_sum = 0.0
        for i in range(rows):
            sq_sum += (values[i][j] - mean) ** 2
        means.append(mean)
        stds.append((sq_sum / rows) ** 0.5)

    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = (values[i][j] - means[j]) / stds[j]
    return np.array(result)


def standardise_np(data):
    """Standardise each column using broadcasting (no loops)."""
    data = np.asarray(data, dtype=float)
    return (data - data.mean(axis=0)) / data.std(axis=0)