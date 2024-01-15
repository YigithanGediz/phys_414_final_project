import pandas as pd
import numpy as np
from scipy.stats import linregress

G = 6.6743*1e-8 # cm3 g-1 s-2
M_solar = 1.989*1e33 # g
R_earth = 6.378*1e8 # cm


"""
Helper functions we use throughout the project
"""

def load_data(path):
    """
    :param path: path to csv file
    :return: load data and add radius to it
    """

    star_data = pd.read_csv(path)
    # convert mass to CGS units
    star_data["cgs_mass"] = M_solar*star_data["mass"]

    # Radius from basic newton gravity.
    star_data["r"] = np.sqrt(G * star_data["cgs_mass"] / pow(10, star_data["logg"]))

    # After calculating r in CGS units, we divide by custom units
    star_data["r"] /= R_earth
    star_data = star_data.sort_values(by="r")
    star_data["id"] = np.arange(0, len(star_data))

    return star_data


import random
def np_histogram_distribution(data, bins=10):
    """
    Calculate the distribution of the given data using np.histogram.

    :param data: The input data array.
    :param bins: Number of bins or the bin edges.
    :return: Tuple of histogram values and bin edges.
    """
    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges

def sample_from_distribution(hist, bin_edges, num_samples):
    """
    Sample from the distribution defined by histogram and bin edges.

    :param hist: Histogram values (frequencies).
    :param bin_edges: Edges of the bins.
    :param num_samples: Number of samples to generate.
    :return: Array of samples.
    """
    # Normalize the histogram to create a probability distribution
    pdf = hist / hist.sum()
    # Create a cumulative distribution function (CDF)
    cdf = np.cumsum(pdf)
    # Generate random values
    random_values = np.random.rand(num_samples)
    # Find bins corresponding to random values using CDF
    sample_indices = np.searchsorted(cdf, random_values)
    # Generate samples by choosing a random point within each bin
    sampled_data = [np.random.uniform(bin_edges[i], bin_edges[i + 1]) for i in sample_indices]

    return np.array(sampled_data)

