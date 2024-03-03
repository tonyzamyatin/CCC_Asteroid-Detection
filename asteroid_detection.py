from itertools import groupby

from scipy.ndimage import label, find_objects
from typing import List, Dict, Tuple

import numpy as np


class Image:
    """A class to represent an image with a timestamp and pixel values."""
    def __init__(self, timestamp: int, pxl_vals: np.ndarray):
        """
        Initialize the Image object with a timestamp and pixel values.

        :param timestamp: An integer representing the time the image was taken.
        :param pxl_vals: A 2D numpy array representing the pixel values of the image.
        """
        self.timestamp = timestamp
        self.pxl_vals = pxl_vals
        self.row_count, self.col_count = pxl_vals.shape


def readFile(file: str):
    with open(f"input/{file}.inp", 'r') as f:
        meta_line = f.readline().split()
        meta = {"start": int(meta_line[0]), "end": int(meta_line[1]), "img_count": int(meta_line[2])}
        images = []
        for i in range(meta["img_count"]):
            img_meta_line = f.readline().split()
            timestamp = int(img_meta_line[0])
            pxl_vals = []
            row_cnt = int(img_meta_line[1])
            for _ in range(row_cnt):
                row_line = f.readline().split()
                row = np.array([int(pxl) for pxl in row_line])
                pxl_vals.append(row)
            image = Image(timestamp, np.array(pxl_vals))
            images.append(image)
    return meta, images


def write_aggregated_occurrences_to_file(aggregated_data, file):
    with open(f"output/{file}.out", 'w') as f:
        for shape, first, last, count in aggregated_data:
            f.write(f"{first} {last} {count}\n")


def label_image(image: Image) -> Tuple[Image, int]:
    # Create a binary image where 0 represents no intensity and 1 represents positive intensity
    binary_image = (image.pxl_vals > 0).astype(int)

    # Use label function to detect connected components
    labels, num_features = label(binary_image)
    labelled_img = Image(image.timestamp, labels)
    return labelled_img, num_features


def contains_asteroid(image: Image) -> bool:
    labeled_image, num_features = label_image(image)
    # If num_features is at least 1, there is an asteroid
    return num_features > 0


def identify_asteroids(images: List[Image]) -> List[bool]:
    return [contains_asteroid(img) for img in images]


def filter_for_observations(images: List[Image]) -> List[Image]:
    filter_mask = identify_asteroids(images)
    filtered_images = []
    for img, found in zip(images, filter_mask):
        if found:
            filtered_images.append(img)

    return filtered_images


def get_asteroid_shape(labelled_image: np.ndarray, label_num: int) -> np.ndarray:
    """

    :param labelled_image: 2D np.ndarray representing pixel values of labelled image
    :param label_num: number of labels/features in the image
    :return:
    """
    slices = find_objects(labelled_image)[label_num - 1]  # Get the slices for the current label
    asteroid_slice = labelled_image[slices] == label_num  # Extract the slice with the asteroid
    return asteroid_slice


def get_asteroid_observations(labelled_images: List[Image]) -> List[Dict[str, int | bytes]]:
    """
    Converts on the list of labelled images to a list of asteroid observations, where an observation is defined as a (timestamp, asteroid shape)
    dict. The asteroid shape is byte-encoded to allow for hashing.
    :param labelled_images: a list of images, i.e. pixel values of asteroid shape are set to 1 all others to 0.
    :return: a list of equal length as labelled_images containing asteroid observations.
    """
    observations = []
    for labelled_img in labelled_images:
        ast_shape = get_asteroid_shape(labelled_img.pxl_vals, 1)  # Assume label_num == 1
        ast_enc = np.packbits(ast_shape).tobytes()  # Convert to bytes for hashing
        observations.append({"timestamp": labelled_img.timestamp, "shape": ast_enc})
    return observations


def asteroid_observation_summary_by_shape(observations: List[Dict[str, int | bytes]]) -> List[Tuple[bytes, int, int, int]]:
    """
    Summarizes the observations of asteroids differentiated by their shape.
    :param observations: a list of asteroid observations. An observation is a tuple of a timestamp and a byte-representation of the asteroid shape
    :return: A summary of same shape asteroid observations consisting of asteroid shape, first observation timestamp, last observation timestamp,
    and count of observation.
    """
    # Sort occurrences by shape and then by timestamp
    observations.sort(key=lambda x: (x["shape"], x["timestamp"]))
    shape_to_data = {}  # Dictionary to hold the data for each shape
    for timestamp, shape in observations:
        if shape in shape_to_data:
            # Update existing shape data
            shape_data = shape_to_data[shape]
            shape_data['last'] = timestamp
            shape_data['count'] += 1
        else:
            # Create new shape data
            shape_to_data[shape] = {'first': timestamp, 'last': timestamp, 'count': 1}

    # Convert the dictionary to a list and sort by first occurrence
    aggregated_data = [(shape, data['first'], data['last'], data['count']) for shape, data in shape_to_data.items()]
    aggregated_data.sort(key=lambda x: x[1])  # x[1] is the first occurrence timestamp
    return aggregated_data


def asteroid_occurrence_summary_by_orbit(observations: List[Dict[str, int | bytes]], min_observations: int) -> List[Tuple[bytes, int, int, int]]:
    """
    Summarizes the observations of asteroids differentiated by their orbit.
    :param observations: a list of asteroid observations. An observation is a tuple of a timestamp and a byte-representation of the asteroid shape
    :param min_observations: minimum number of reoccurring regular observations for an asteroid shape to be considered a unique asteroid
    :return: A summary of unique asteroid observations consisting of asteroid shape, first observation timestamp, last observation timestamp,
    and count of observation.
    """
    # Sort occurrences by timestamp
    observations.sort(key=lambda x: (x["shape"], x["timestamp"]))
    aggregated_data = []

    for shape, group in groupby(observations, key=lambda x: x["shape"]):
        shape_occurrences = list(group)
        orbiting_asteroids = find_orbiting_asteroids(shape_occurrences, min_observations)  # Find image subsets

        for ast_occurrences in orbiting_asteroids:
            first_timestamp = ast_occurrences[0]
            last_timestamp = ast_occurrences[-1]
            count = len(ast_occurrences)
            aggregated_data.append((shape, first_timestamp, last_timestamp, count))

    # Sort the aggregated data by the first occurrence
    aggregated_data.sort(key=lambda x: x[1])

    return aggregated_data


def find_orbiting_asteroids(shape_observations: List[Dict[str, int | bytes]], min_observations: int) -> List[List[int]]:
    """
    Identifies and groups all observations of asteroids of a given shape into a list, if the shape reoccurred in regular intervals at least
    min_observations times.
    :param shape_observations: observations of asteroids with the same shape sorted by timestamp
    :param min_observations: minimum number of reoccurring regular observations for an asteroid shape to be considered a unique asteroid
    :return: a list of distinct asteroid observation-lists by shape and orbit
    """
    time_interval = shape_observations[-1]["timestamp"] - shape_observations[0]["timestamp"]
    max_d = time_interval // (min_observations - 1)
    distinct_asteroids = []
    processed_timestamps = set()  # Keep track of processed timestamps

    # Convert shape_occurrences to a set of timestamps for faster lookup
    timestamps = {obs["timestamp"] for obs in shape_observations}

    for d in range(1, max_d + 1):
        for observation in shape_observations:
            current_ts = observation["timestamp"]
            # Skip if this timestamp has already been processed
            if current_ts in processed_timestamps:
                continue

            # Initialize a subset with the current observation
            subset = [current_ts]
            next_prospect_ts = current_ts + d

            # Look for subsequent occurrences at regular intervals d
            while next_prospect_ts in timestamps:
                subset.append(next_prospect_ts)
                next_prospect_ts += d

            # If we have at least 4 occurrences, it's a valid subset
            if len(subset) >= 4:
                # Save the timestamps of the subset
                distinct_asteroids.append(subset)
                # Add all timestamps in this subset to the processed set
                processed_timestamps.update(subset)

    return distinct_asteroids


def solve_lvl1(file: str):
    meta, images = readFile(file)
    filtered_images = filter_for_observations(images)
    with open(f"output/{file}.out", 'w') as f:
        for img in filtered_images:
            f.write(f"{img.timestamp}\n")


def solve_lvl2(file: str):
    meta, images = readFile(file)
    filtered_images = filter_for_observations(images)
    labelled_images = [label_image(img)[0] for img in filtered_images]
    occurrences = get_asteroid_observations(labelled_images)
    aggregated_occurrences = asteroid_observation_summary_by_shape(occurrences)
    write_aggregated_occurrences_to_file(aggregated_occurrences, file)


def solve_lvl3(file: str):
    meta, images = readFile(file)
    filtered_images = filter_for_observations(images)
    labelled_images = [label_image(img)[0] for img in filtered_images]
    occurrences = get_asteroid_observations(labelled_images)
    aggregated_occurrences = asteroid_occurrence_summary_by_orbit(occurrences, min_observations=4)
    write_aggregated_occurrences_to_file(aggregated_occurrences, file)


if __name__ == "__main__":
    solve_lvl1("lvl1-0")
    solve_lvl1("lvl1-1")
    solve_lvl1("lvl1-2")
    solve_lvl1("lvl1-3")
    solve_lvl1("lvl1-4")
    solve_lvl2("lvl2-0")
    solve_lvl2("lvl2-1")
    solve_lvl2("lvl2-2")
    solve_lvl2("lvl2-3")
    solve_lvl2("lvl2-4")
    solve_lvl3("lvl3-0")
    solve_lvl3("lvl3-1")
    solve_lvl3("lvl3-2")
    solve_lvl3("lvl3-3")
    solve_lvl3("lvl3-4")
    solve_lvl3("lvl3-5")
