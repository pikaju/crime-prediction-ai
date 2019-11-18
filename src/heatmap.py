import numpy as np
from bisect import bisect_left
import png


def generate(data,
             index_range,
             timestamp_interval,
             size,
             min_coordinates,
             max_coordinates):
    """
    Generates a 2D numpy array highlighting locations in which crimes occurred.
    """
    result = np.zeros(size, np.float32)
    for i in range(*index_range):
        e = data[i]
        # Ignore elements whose timestamp is outside of the range.
        if timestamp_interval[0] < e[0] < timestamp_interval[1]:
            # Extract coordinates from data point.
            coordinates = e[1:]
            # Normalize coordinates to be between (0.0, 0.0) and (1.0, 1.0)
            coordinates = (coordinates - min_coordinates) / \
                (np.array(max_coordinates) - np.array(min_coordinates))
            # Scale coordinates to fall in the range of the numpy array's size.
            coordinates *= size
            x = int(coordinates[0])
            y = int(coordinates[1])
            if x < 0 or x >= result.shape[0] or y < 0 or y >= result.shape[1]:
                continue

            result[x, y] += 1.0

    return result


def generate_timeinterval(data,
                          timestamp_interval,
                          size,
                          min_coordinates,
                          max_coordinates):
    timestamps = data[:, 0]
    hi = len(timestamps) - 1
    index_range = (bisect_left(timestamps, timestamp_interval[0], hi=hi),
                   bisect_left(timestamps, timestamp_interval[1], hi=hi))
    return generate(data,
                    index_range,
                    timestamp_interval,
                    size,
                    min_coordinates,
                    max_coordinates)


def save(filename, heatmap):
    image = (heatmap * 255).astype(np.int8)
    png.from_array(image, mode='L').save(filename)
