from .processes import *

def graph_analysis(A, signal, type='GFT-Laplacian'):

    if type == 'GFT-Laplacian':
        L, U, V, Uinv, _, _, _ = graph_utils.prep_transform(A, gso="laplacian")
        coefs = operations.GFT(signal, U, Uinv=Uinv)
        return U, coefs, V
    
    elif type == 'GFT-Adjacency':
        L, U, V, Uinv, _, _, _ = graph_utils.prep_transform(A, gso="adj")
        coefs = operations.GFT(signal, U, Uinv=Uinv)
        return U, coefs, V

    elif type == 'MyBasis':
        L, U, V, Uinv, S, J, Sinv = graph_utils.prep_transform(A, gso="laplacian")
        coefs = operations.hermitian(S) @ operations.GFT(signal, U, Uinv=Uinv)
        return S @ U, coefs, V
    
    elif type == 'Polar-Decomposition_in':
        Q,F,P = operations.polar_decomposition(A)
        Up, Vp = operations.compute_basis(P, verbose=False)
        coefs = operations.GFT(signal, Up)
        return Up, coefs, Vp

    elif type == 'Polar-Decomposition_out':
        Q,F,P = operations.polar_decomposition(A)
        Uf, Vf = operations.compute_basis(F, verbose=False)
        coefs = operations.GFT(signal, Uf)
        return Uf, coefs, Vf

    elif type == 'Polar-Decomposition_inflow':
        Q,F,P = operations.polar_decomposition(A)
        Uq, Vq = operations.compute_basis(Q, verbose=False)
        coefs = operations.GFT(signal, Uq)
        return Uq, coefs, Vq
    else:
        raise ValueError()