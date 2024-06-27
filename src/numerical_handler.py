from .processes import *

def graph_analysis(A:np.ndarray, signal:np.ndarray, type:str='GFT-Laplacian'):
    """
    Generate a signal through various projections.
    
    Parameters
    ----------
    A : np.ndarray
        The input graph adjacency matrix.
    signal : np.ndarray
        The input signal.
    type : str, optional
        The type of projection to use, by default 'GFT-Laplacian'.
    
    Returns
    -------
    U : np.ndarray
        The projection basis.
    coefs : np.ndarray
        The signal coefficients in the projection basis.
    V : np.ndarray
        The projection basis for the dual space.
    """
    if type == 'GFT-Laplacian':
        L, U, V, Uinv, _, _, _ = graph_utils.prep_transform(A, gso="laplacian")
        if len(signal.shape) > 1:
            coefs = np.zeros_like(signal)
            for n in range(signal.shape[0]):
                coefs[n] = operations.GFT(signal[n], U, Uinv=Uinv)
        else:
            coefs = operations.GFT(signal, U, Uinv=Uinv)
        return U, coefs, V
    

    elif type == 'GFT-Adjacency':
        L, U, V, Uinv, _, _, _ = graph_utils.prep_transform(A, gso="adj")
        if len(signal.shape) > 1:
            coefs = np.zeros_like(signal)
            for n in range(signal.shape[0]):
                coefs[n] = operations.GFT(signal[n], U, Uinv=Uinv)
        else:
            coefs = operations.GFT(signal, U, Uinv=Uinv)
        return U, coefs, V


    elif type == 'MyBasis':
        L, U, V, Uinv, S, J, Sinv = graph_utils.prep_transform(A, gso="laplacian")
        if len(signal.shape) > 1:
            coefs = np.zeros_like(signal)
            for n in range(signal.shape[0]):
                coefs[n] = operations.hermitian(S) @ operations.GFT(signal[n], U, Uinv=Uinv)
        else:
            coefs = operations.hermitian(S) @ operations.GFT(signal, U, Uinv=Uinv)
        return S @ U, coefs, V
    

    elif type == 'Polar-Decomposition_in':
        Q,F,P = operations.polar_decomposition(A)
        Up, Vp = operations.compute_basis(P, verbose=False)
        if len(signal.shape) > 1:
            coefs = np.zeros_like(signal)
            for n in range(signal.shape[0]):
                coefs[n] = operations.GFT(signal[n], Up)
        else:
            coefs = operations.GFT(signal, Up)
        return Up, coefs, Vp


    elif type == 'Polar-Decomposition_out':
        Q,F,P = operations.polar_decomposition(A)
        Uf, Vf = operations.compute_basis(F, verbose=False)
        if len(signal.shape) > 1:
            coefs = np.zeros_like(signal)
            for n in range(signal.shape[0]):
                coefs[n] = operations.GFT(signal[n], Uf)
        else:
            coefs = operations.GFT(signal, Uf)
        return Uf, coefs, Vf


    elif type == 'Polar-Decomposition_inflow':
        Q,F,P = operations.polar_decomposition(A)
        Uq, Vq = operations.compute_basis(Q, verbose=False)
        if len(signal.shape) > 1:
            coefs = np.zeros_like(signal)
            for n in range(signal.shape[0]):
                coefs[n] = operations.GFT(signal[n], Uq)
        else:
            coefs = operations.GFT(signal, Uq)
        return Uq, coefs, Vq
    
    
    else:
        raise ValueError()