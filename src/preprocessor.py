import argparse
import csv
from datetime import datetime
import numpy as np
import heatmap

chicago_min_coordinates = (42.029579, -87.947709)
chicago_max_coordinates = (41.641167,  -87.521108)
resolution = (64, 64)
time_interval = 3600 * 24


def load(filename):
    """
    Loads a CSV file from the Chicago Data Portal.
    Returns a list of each crime's timestamp, latitude and longitude.
    """
    data = []
    with open(filename, newline='\n') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            try:
                timestamp = datetime.timestamp(
                    datetime.strptime(row[2], '%m/%d/%Y %I:%M:%S %p'))
                latitude = float(row[-3])
                longitude = float(row[-2])
                data.append((timestamp, latitude, longitude))
            except:
                pass

    return data


def generate_heatmaps(data):
    start = int(data[0][0]) + 3600 * 24 * 8
    stop = start + 3600 * 24 * 365 * 4  # int(data[-1][0])
    step = 3600 * 24

    heatmaps = []

    for timestamp in range(start, stop, step):
        hm = heatmap.generate_timeinterval(data,
                                           (timestamp - 3600 * 24, timestamp),
                                           resolution,
                                           chicago_min_coordinates,
                                           chicago_max_coordinates)
        hm /= 16.0
        heatmaps.append(hm)

    return np.array(heatmaps)


def main():
    parser = argparse.ArgumentParser(description='Preprocess the crime data.')
    parser.add_argument(
        'mode', type=str, help='Preprocessing mode, can be "raw" or "heatmaps"')
    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('output', type=str, help='Preprocessed output file')
    args = parser.parse_args()

    if args.mode == 'raw':
        data = load(args.input)
        data.sort()
        data = np.array(data)
        np.save(args.output, data)
    elif args.mode == 'heatmaps':
        data = np.load(args.input)
        heatmaps = generate_heatmaps(data)
        np.save(args.output, heatmaps)


if __name__ == '__main__':
    main()
