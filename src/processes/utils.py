"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

import os
import sys
import pickle
from tqdm import tqdm
from copy import deepcopy
from typing import Optional

import numpy as np
from numpy.linalg import matrix_rank

from math import comb
import scipy.io as sio
from scipy.stats import zscore
from scipy.stats import pearsonr

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


import networkx as nx

from  PIL import Image

def save(pickle_filename:str, anything:Optional[np.ndarray]):
    """
    Pickle array

    Parameters
    ----------
    pickle_filename : str
        The filename to save the pickled array to
    anything : Optional[np.ndarray]
        The array to pickle

    Returns
    -------
    None

    """
    with open(pickle_filename, "wb") as handle:
        pickle.dump(anything, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(pickle_filename:str):
    """
    Loads a pickled array from a file.

    Parameters
    ----------
    pickle_filename : str
        The path to the pickled file to load.

    Returns
    -------
    b : Any
        The unpickled object loaded from the file.
    """
    with open(pickle_filename, "rb") as handle:
        b = pickle.load(handle)
    return b

def normalize(a:np.ndarray):
    tmp = a - np.mean(a)
    return tmp / np.std(a)

def standardize(a:np.ndarray):
    tmp = a - a.min()
    return tmp / tmp.max()

def demodulate(a:np.ndarray):
    return a / a.max()

def nodecimal(array:np.ndarray, tol:float=1e-10):
    """
    Sets array values below a tolerance to zero.

    Maps real and imaginary parts of complex array separately.
    Keeps complex array structure.

    Parameters
    ----------
    array : np.ndarray
        Input array, can be real or complex valued.

    tol : float, optional
        Tolerance below which values are set to zero.

    Returns
    -------
    array : np.ndarray
        Output with small values set to zero.
    """

    ret_real = deepcopy(array.real)
    ret_image = deepcopy(array.imag)
    ret_real[np.abs(ret_real) < tol] = 0.0
    ret_image[np.abs(ret_image) < tol] = 0.0

    return ret_real + 1j * ret_image

def signed_amplitude(complexarr: np.ndarray):
    """
    Compute signed amplitude of a complex array.

    Takes the absolute value and multiplies by the sign of the 
    real part to compute a signed amplitude.

    Prints a warning if the real part contains any zeros,
    as taking the sign of zero is undefined.

    Parameters
    ----------
    complexarr : np.ndarray
        Input complex array

    Returns
    -------
    ret : np.ndarray 
        Signed amplitude of complexarr
    """
    if np.sum(complexarr.real == 0) > 0:
        print("CAREFUL: there are 0 real valued")
    ret = np.abs(complexarr) * np.sign(complexarr.real)
    return ret

def smooth1d(y: np.ndarray, box_pts: int):
    """
    Applies a 1D smoothing filter to the input array `y` using a box filter of size `box_pts`.

    Parameters
    ----------
    y (np.ndarray): The input array to be smoothed.
    box_pts (int): The size of the box filter to use for smoothing.

    Returns
    -------
    y_smooth (np.ndarray): The smoothed version of the input array `y`.
    """

    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth
