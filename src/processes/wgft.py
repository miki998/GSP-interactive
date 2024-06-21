"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from .utils import *
from .operations import *
from .graph_utils import *

def ndft_basis_matrix(x:np.ndarray, freqspace:Optional[np.ndarray]=None):
    """
    Compute the basis matrix for the non-uniform discrete Fourier transform (NUDFT).

    Parameters:
    ----------
    x : np.ndarray
        The input array of samples.
    freqspace : np.ndarray
        The frequency space to compute the basis for.

    Returns:
    -------
    basis: np.ndarray
        The basis matrix.
    """
    assert x.ndim == 1

    if freqspace is None:
        N = x.size
        assert N % 2 == 0
        freqspace = -(N // 2) + np.arange(N)

    basis = np.exp(-2j * np.pi * x * freqspace[:, None])
    basis = basis / np.linalg.norm(basis, axis=0)
    # reorder in a graph eigenvectors-like matrix form
    basis = basis[np.argsort(np.abs(freqspace))].T
    return basis

def ndft_invbasis_matrix(x:np.ndarray, freqspace:Optional[np.ndarray]=None):
    """
    Compute the inverse basis matrix for the non-uniform discrete Fourier transform (NUDFT).

    Parameters:
    ----------
    x : np.ndarray
        The input array of samples.
    freqspace : np.ndarray
        The frequency space to compute the inverse basis for.

    Returns:
    -------
    basis: np.ndarray
        The inverse basis matrix.
    """
    assert x.ndim == 1

    if freqspace is None:
        N = x.size
        assert N % 2 == 0
        freqspace = -(N // 2) + np.arange(N)

    basis = np.exp(2j * np.pi * x * freqspace[:, None])
    basis = basis / np.linalg.norm(basis, axis=0)
    # reorder in a graph eigenvectors-like matrix form
    basis = basis[np.argsort(np.abs(freqspace))].T
    return basis


def weighted_cyclic(N:int, wval:list, widx:list):
    """
    Compute Basis for weighted chain graph

    Parameters:
    -----------
    N (int): The size of the graph
    wval (list): The weights to apply to the edges of the graph
    widx (list): The indices of the edges to apply the weights to

    Returns:
    --------
    U, V, Uu, Vu, L: The eigenvectors and Laplacian matrix of the weighted graph
    """

    cycle = make_graph(N, graph_type="cycle")

    # Weight-modifying
    for k in range(len(widx)):
        node_l, node_r = widx[k]
        cycle[node_l, node_r] = wval[k]

    # Only return the eigenvectors
    undirected_c = cycle.T + cycle

    L = compute_directed_laplacian(cycle.T)
    U, V = compute_basis(L, verbose=False)

    Lu = compute_directed_laplacian(undirected_c)
    Uu, Vu = compute_basis(Lu, verbose=False)
    return U, V, Uu, Vu, L
