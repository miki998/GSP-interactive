"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from .utils import *

from mpl_toolkits.mplot3d import Axes3D, art3d


def signal2face(signal: np.ndarray, faces_idx: np.ndarray, region_idx: np.ndarray, vlabels: np.ndarray):
    """
    Convert signal on vertices to signal on triangulated faces.

    Parameters:
    -----------
        signal (np.ndarray): Signal values on vertices.
        faces_idx (np.ndarray): Indices of vertices that form each face.
        region_idx (np.ndarray): Region indices for each vertex.
        vlabels (np.ndarray): Labels for each vertex.

    Returns:
    --------
        faces_signal (np.ndarray): Signal values on each face.
    """

    d = {r: ridx for ridx, r in enumerate(region_idx)}
    d[0] = -1 # background value
    vertex_signal = signal[np.array([d[k] for k in vlabels])]
    faces_signal = vertex_signal[faces_idx].mean(axis=1)
    return faces_signal

def plot_mesh(vertices: np.ndarray, f_index: np.ndarray, f_colors: np.ndarray, 
              title:str="The Macaque Brain", view_init:tuple=(0,-136), eps:float=0.1,
                ax:matplotlib.axes=None, fig:matplotlib.figure.Figure=None, cmap:str='viridis'):
    """
    Plot a 3D mesh given the vertices, face indices, and face colors.

    Parameters:
    -----------
        vertices (np.ndarray): Array of vertex coordinates.
        f_index (np.ndarray): Array of face indices.
        f_colors (np.ndarray): Array of face colors.
        title (str, optional): Title of the plot, defaults to "The Macaque Brain".
        view_init (tuple, optional): Initial view angles for the 3D plot, defaults to (0, -136).
        eps (float, optional): Epsilon value for setting plot limits, defaults to 0.1.
        ax (matplotlib.axes, optional): Existing 3D axes to plot on.
        fig (matplotlib.figure.Figure, optional): Existing figure to plot on.

    Returns:
    --------
        None
    """

    if ax is None or fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    norm = plt.Normalize(f_colors.min(), f_colors.max())
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(norm(f_colors))

    pc = art3d.Poly3DCollection(vertices[f_index],
                                facecolors=colors)

    ax.add_collection(pc)
    ax.set_xlim(vertices[:,0].min() + vertices[:,0].min() * eps , vertices[:,0].max() + vertices[:,0].max() * eps)
    ax.set_ylim(vertices[:,1].min() + vertices[:,1].min() * eps , vertices[:,1].max() + vertices[:,1].max() * eps)
    ax.set_zlim(vertices[:,2].min() + vertices[:,2].min() * eps , vertices[:,2].max() + vertices[:,2].max() * eps)

    ax.set_title(title)
    ax.axis('off')

    ax.view_init(view_init[0], view_init[1])
    if ax is None or fig is None:
        plt.show()