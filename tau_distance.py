import scipy.stats as stats
import numpy as np


def tau_distance(r1, r2):
    """
    Tau distance
    Values close to 1 indicate strong agreement,
    and values close to -1 indicate strong disagreement.
    :param r1: list1
    :param r2: list2
    :return: tau distance between two lists
    """
    tau, p_value = stats.kendalltau(r1, r2)
    return tau

def normalised_kendall_tau_distance(r1, r2):
    """
    Compute the normalized Kendall tau distance.
    :param r1: list1
    :param r2: list2
    :return: normalized tau distance between two lists
    """
    n = len(r1)
    assert len(r2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(r1)
    b = np.argsort(r2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))
