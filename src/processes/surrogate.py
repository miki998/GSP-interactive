"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from .utils import *
from .operations import *

from joblib import Parallel, delayed


def p_value(null_distrib: np.ndarray, statistic: float, two_tail: bool = False):
    """
    Calculates the p-value for a given test statistic and null distribution.

    Parameters:
    -----------
    null_distrib : np.ndarray
        The null distribution to compare the statistic against.
    statistic : float 
        The test statistic value.
    two_tail : bool, optional
        Whether to calculate a two-tailed p-value, by default False.

    Returns:
    --------
    score : float
        The calculated p-value.
    """

    rc = null_distrib > statistic
    lc = null_distrib < statistic

    score_r = np.mean(rc)
    score_l = np.mean(lc)
    score = min(score_r, score_l)

    if two_tail:
        score *= 2
        score = min(score, 1)

    return score

def randomizer_phase(tasks:np.ndarray, N:int, seed:int, 
                     conj:bool=False, onlysign:bool=False):
    """
    Compute randomizing vector

    Parameters:
    -----------
    tasks : np.ndarray
        The tasks to compute the randomizing vector for.
    N : int
        The size of the randomizing vector.
    seed : int
        The seed to use for the random number generator.
    conj : bool, optional
        Whether to use the conjugate of the random shift, by default False.
    onlysign : bool, optional
        Whether to only randomize the sign, by default False.

    Returns:
    --------
    randomizer_vector : np.ndarray
        The computed randomizing vector.
    """

    np.random.seed(seed)
    randomizer_vector = np.zeros((N), dtype=complex)
    for t in tasks:
        if len(t) == 1:
            # Flip Sign with uniform probability
            randomval = np.random.random()
            mask = -1 * (randomval > 0.5) + 1 * (randomval <= 0.5)
            randomizer_vector[t[0]] = mask
        elif len(t) == 2:
            # Rotate complex value by uniform proba angle
            if onlysign:
                random_shift1 = np.random.random()
                random_shift2 = np.random.random()
                s1 = -1 * (random_shift1 > 0.5) + 1 * (random_shift1 <= 0.5)
                if conj:
                    s2 = -s1
                else:
                    s2 = -1 * (random_shift2 > 0.5) + 1 * (random_shift2 <= 0.5)
            else:
                random_shift1 = np.random.random() * 2 * np.pi
                random_shift2 = np.random.random() * 2 * np.pi
                s1 = np.exp(1j * random_shift1)
                if conj:
                    s2 = np.exp(-1j * random_shift1)
                else:
                    s2 = np.exp(1j * random_shift2)

            randomizer_vector[t[0]] = s1
            randomizer_vector[t[1]] = s2
        else:
            print("Weird grouping, stop")
    randomizer_vector = np.diag(randomizer_vector)
    return randomizer_vector

#######################
#### GRADIENT DESC ####
#######################
def randomizer_solver(sig:torch.tensor, R:torch.tensor, A:torch.tensor, B:torch.tensor, 
            Lc:torch.tensor, mask_diag:torch.tensor, step_size:float=1e-3, n_iter:int=1000, verbose:bool=False,
            cuda:bool=False):
    """
    Performs gradient descent on a convex loss function to find a 
    randomizer matrix R that satisfies certain constraints.

    Parameters:
    -----------
        sig (np.ndarray): The input signal.
        R (np.ndarray): The randomizer matrix to be optimized.
        A (np.ndarray): A matrix used in the loss function.
        B (np.ndarray): Another matrix used in the loss function.
        Lc (np.ndarray): A matrix used in the loss function.
        step_size (float, optional): The step size for the gradient descent, defaults to 1e-3.
        n_iter (int, optional): The number of iterations to perform, defaults to 1000.
        verbose (bool, optional): Whether to print progress during the optimization, defaults to False.

    Returns:
    --------
        nR (np.ndarray): The optimized randomizer matrix R.
        loss (float): The final value of the loss function.
    """

    initial_norm = torch.linalg.norm(sig)
    initial_smooth = torch.linalg.norm(Lc @ sig)
    optimizer = torch.optim.Adam([R], lr=step_size)

    # Pre-compute
    coef = B @ sig
    for i in range(n_iter):
        # Compute the randomized value
        x_r = A @ R @ coef

        smoothness = (torch.linalg.norm(Lc @ x_r) - initial_smooth) ** 2
        energy = (torch.linalg.norm(x_r) - initial_norm) ** 2
        real_term = torch.linalg.norm(x_r.imag)

        loss = smoothness + energy + 100 * real_term

        loss.backward()

        # Forcing diagonality
        R.grad.data = R.grad.data * mask_diag
        optimizer.step()
        optimizer.zero_grad()
        # R.data = R.data - step_size * R.grad.data

        # R.grad.data.zero_()

        if verbose:
            if (i % (n_iter // 10)) == 0:
                print("{}, \t{}".format(i, loss.item()))

    if cuda:
        nR = R.cpu().detach().numpy()
        nR.imag = 0
        x_r = A @ R.detach() @ coef
        smoothness = torch.abs(torch.linalg.norm(Lc @ x_r) - initial_smooth) / initial_smooth
        energy = torch.abs(torch.linalg.norm(x_r) - initial_norm) / initial_norm
        smoothness = smoothness.cpu()
        energy = energy.cpu()
    else:
        nR = R.detach().numpy()
        nR.imag = 0
        x_r = A @ R.detach() @ coef
        smoothness = torch.abs(torch.linalg.norm(Lc @ x_r) - initial_smooth) / initial_smooth
        energy = torch.abs(torch.linalg.norm(x_r) - initial_norm) / initial_norm
    
    return nR, smoothness, energy


def optimized_random_surrogate_GD(signal:np.ndarray, L:np.ndarray, U:np.ndarray, Uinv:np.ndarray, 
                                  V:np.ndarray, S:np.ndarray, Sinv:np.ndarray, 
                                  N:int, nrands:int=99, n_iter:int=1000, ssize:float=1e-2,
                                  cuda:bool=False, parrallel:bool=True, verbose:bool=False):
    """
    Generate null distribution from optimized (GD) random surrogates

    Parameters:
    ----------
    signal: np.ndarray
        The input signal.
    L: np.ndarray
        The Laplacian matrix.
    U: np.ndarray
        The matrix of eigenvectors.
    Uinv: np.ndarray
        The inverse of the matrix of eigenvectors.
    V: np.ndarray
        The matrix of eigenvectors.
    S: np.ndarray
        The diagonal matrix of singular values.
    Sinv: np.ndarray
        The inverse of the diagonal matrix of singular values.
    N: int
        The number of eigenvectors to use.
    nrands: int, optional
        The number of random surrogates to generate. Defaults to 99.
    n_iter: int, optional
        The number of iterations for the gradient descent optimization. Defaults to 1000.
    ssize: float, optional
        The step size for the gradient descent optimization. Defaults to 1e-2.

    Returns:
    --------
    ret: np.ndarray
        The array of optimized random surrogates.
    losses: np.ndarray
        The array of losses from the gradient descent optimization.
    """

    if parrallel:
        return optimized_random_surrogate_GD_parrallel(signal, L, U, Uinv, V, 
                                                       S, Sinv, N, nrands, n_iter, 
                                                       ssize, cuda)

    tasks = eigvalues_pairs(V)
    if cuda: 
        A = torch.tensor(U @ S).cuda()
        B = torch.tensor(Sinv @ Uinv).cuda()
        Lc = torch.tensor(L, dtype=torch.complex128).cuda()
        ssignal = GFT(signal, U, Uinv=Uinv)
        signal = torch.tensor(signal, dtype=torch.complex128).cuda()
        mask_diag = torch.tensor(np.eye(len(signal)), dtype=torch.complex128).cuda()
        
    else:
        A = torch.tensor(U @ S)
        B = torch.tensor(Sinv @ Uinv)
        Lc = torch.tensor(L, dtype=torch.complex128)
        ssignal = GFT(signal, U, Uinv=Uinv)
        signal = torch.tensor(signal, dtype=torch.complex128)
        mask_diag = torch.tensor(np.eye(len(signal)), dtype=torch.complex128)

    ret = []
    losses = []

    for k in tqdm(range(nrands), disable=not verbose):
        R = randomizer_phase(tasks, N, k, conj=False, onlysign=False)
        if cuda:
            R = torch.tensor(R).cuda()
        else:
            R = torch.tensor(R)
        R.requires_grad = True
        opt_R, smooth_loss, energy_loss = randomizer_solver(signal, R, A, B, Lc, mask_diag, 
                                                           n_iter=n_iter, step_size=ssize, 
                                                           cuda=cuda)

        randomized = inverseGFT(S @ opt_R @ Sinv @ ssignal, U)

        losses.append([smooth_loss, energy_loss])
        ret.append(randomized)

    losses = np.asarray(losses)
    ret = np.asarray(ret).real

    return ret, losses

def optimized_random_surrogate_GD_parrallel(signal:np.ndarray, L:np.ndarray, U:np.ndarray, Uinv:np.ndarray, 
                                  V:np.ndarray, S:np.ndarray, Sinv:np.ndarray, 
                                  N:int, nrands:int=99, n_iter:int=1000, ssize:float=1e-2,
                                  cuda:bool=False):
    """
    Generate null distribution from optimized (GD) random surrogates

    Parameters:
    ----------
    signal: np.ndarray
        The input signal.
    L: np.ndarray
        The Laplacian matrix.
    U: np.ndarray
        The matrix of eigenvectors.
    Uinv: np.ndarray
        The inverse of the matrix of eigenvectors.
    V: np.ndarray
        The matrix of eigenvectors.
    S: np.ndarray
        The diagonal matrix of singular values.
    Sinv: np.ndarray
        The inverse of the diagonal matrix of singular values.
    N: int
        The number of eigenvectors to use.
    nrands: int, optional
        The number of random surrogates to generate. Defaults to 99.
    n_iter: int, optional
        The number of iterations for the gradient descent optimization. Defaults to 1000.
    ssize: float, optional
        The step size for the gradient descent optimization. Defaults to 1e-2.

    Returns:
    --------
    ret: np.ndarray
        The array of optimized random surrogates.
    losses: np.ndarray
        The array of losses from the gradient descent optimization.
    """

    tasks = eigvalues_pairs(V)
    if cuda: 
        A = torch.tensor(U @ S).cuda()
        B = torch.tensor(Sinv @ Uinv).cuda()
        Lc = torch.tensor(L, dtype=torch.complex128).cuda()
        ssignal = GFT(signal, U, Uinv=Uinv)
        signal = torch.tensor(signal, dtype=torch.complex128).cuda()
        mask_diag = torch.tensor(np.eye(len(signal)), dtype=torch.complex128).cuda()
        
    else:
        A = torch.tensor(U @ S)
        B = torch.tensor(Sinv @ Uinv)
        Lc = torch.tensor(L, dtype=torch.complex128)
        ssignal = GFT(signal, U, Uinv=Uinv)
        signal = torch.tensor(signal, dtype=torch.complex128)
        mask_diag = torch.tensor(np.eye(len(signal)), dtype=torch.complex128)

    def single_surrogate(k):
        R = randomizer_phase(tasks, N, k, conj=False, onlysign=False)
        if cuda:
            R = torch.tensor(R).cuda()
        else:
            R = torch.tensor(R)
        R.requires_grad = True
        opt_R, smooth_loss, energy_loss = randomizer_solver(signal, R, A, B, Lc, mask_diag, 
                                                           n_iter=n_iter, step_size=ssize, 
                                                           cuda=cuda)

        randomized = inverseGFT(S @ opt_R @ Sinv @ ssignal, U)
        return randomized, smooth_loss, energy_loss

    results = Parallel(n_jobs=8)(delayed(single_surrogate)(sidx) for sidx in range(nrands))

    ret = np.array([results[k][0] for k in range(nrands)]).real
    losses = np.asarray([[results[k][1], results[k][2]] for k in range(nrands)])
    return ret, losses

def composite_random_surrogate(signal:np.ndarray, U:np.ndarray, V:np.ndarray,
                               M:np.ndarray, N:np.ndarray, niter:int=99):
    """
    Generate null distribution from super basis phase randomization surrogates.

    Parameters:
    -----------
    signal: np.ndarray
        The input signal.
    U: np.ndarray
        The matrix of eigenvectors.
    V: np.ndarray
        The matrix of eigenvectors.
    M: np.ndarray
        The matrix.
    N: np.ndarray
        The matrix.
    niter: int, optional
        The number of iterations for generating the surrogates. Defaults to 99.

    Returns:
    --------
    real_randomized_signals: np.ndarray
        The array of optimized random surrogates.
    """

    gft_signal = GFT(signal, U)

    tasks = eigvalues_pairs(V)
    real_randomized_signals = []
    for k in tqdm(range(niter)):
        Re = randomizer_phase(tasks, N, k, conj=False, onlysign=False)
        part_rand = Re @ hermitian(M) @ gft_signal

        randomized = inverseGFT(M @ part_rand, U)
        # randomized_signals.append(randomized)
        real_randomized_signals.append(np.sign(randomized.real) * np.abs(randomized))
        if np.sum(np.sign(randomized.real) == 0) > 0:
            print("CAREFUL 0 MULTIPLIED")

    real_randomized_signals = np.asarray(real_randomized_signals)
    return real_randomized_signals


########################
#### POCS ITERATION ####
########################
def generate_orthogonal_vector(u:np.ndarray, v:np.ndarray, 
                               hcutp:float=0.9999, lcutoff:float=1e-8, 
                               verbose:bool=False):
    """
    Generate an orthogonal vector to the given vector `v` starting from a given vector `u`.

    Parameters:
    ----------
        u (np.ndarray): The starting vector.
        v (np.ndarray): The vector to generate an orthogonal vector to.
        hcutp (float, optional): The high cutoff threshold for determining if the vectors are parallel. Defaults to 0.9999.
        lcutoff (float, optional): The low cutoff threshold for determining if the vectors are already orthogonal. Defaults to 1e-8.
        verbose (bool, optional): Whether to print debug messages. Defaults to False.

    Returns:
    --------
        orthogonal_vector (np.ndarray): The generated orthogonal vector.
    """

    vv = np.dot(v, v)
    uv = np.dot(u, v)

    hcutoff = hcutp * vv  # for parralel vectors
    if np.abs(uv) > hcutoff:
        if verbose:
            print("Vectors are parralel")
        return u

    if np.abs(uv) < lcutoff:
        if verbose:
            print("Vectors are already orthogonal")
        return u

    # Compute the orthogonal vector
    orthogonal_vector = u - uv / vv * v
    orthogonal_vector = orthogonal_vector.astype(complex)
    return orthogonal_vector

def pocs_randomizer(signal:np.ndarray, L:np.ndarray, U:np.ndarray, Uinv:np.ndarray, 
                    S:np.ndarray, J:np.ndarray, rscale:float, rseed:int, 
                    n_iter:int=100, cutoff:float=1e-4, signfix:Optional[np.ndarray]=None, 
                    spectreflag:bool=False, complexflag:bool=False, 
                    verbose:bool=False, logs:bool=False):
    """
    Find a randomizer satisfiying smoothness preservation and energy preservation of a given signal.
    Allow for complex signal output.
    Use two-convex sets consecutive projection to find the randomizer.

    Parameters:
    ----------
    signal (np.ndarray): The input signal.
    L (np.ndarray): Laplacian.
    U (np.ndarray): Eigenvectors of Laplacian.
    Uinv (np.ndarray): The inverse of matrix U.
    S (np.ndarray): Composie basis.
    J (np.ndarray): Composite eigenvalues.
    rscale (float): The scale factor for the random starting randomizer vector.
    rseed (int): The random seed to use.
    n_iter (int, optional): The number of iterations to perform. Defaults to 100.
    cutoff (float, optional): The cutoff threshold for smoothness and energy preservation. Defaults to 1e-4.
    signfix (Optional[np.ndarray], optional): The fixed sign constraints on the randomizer. Defaults to None.
    spectreflag (bool, optional): Whether to use the GFT spectrum energy preserving hyperplane. Defaults to False.
    complexflag (bool, optional): Whether to allow for complex signal output. Defaults to False.
    verbose (bool, optional): Whether to print progress information. Defaults to False.
    logs (bool, optional): Whether to return logs of the optimization process. Defaults to False.

    Returns:
    --------
    rsignal (np.ndarray): The randomized signal.
    newr (np.ndarray): The final randomizer vector.
    tv_logs (List[float]): The logs of the total variation difference (if logs is True).
    nr_logs (List[float]): The logs of the energy difference (if logs is True).
    """

    if rseed != -1: np.random.seed(rseed)

    tv_logs = []
    nr_logs = []

    # 1. Prepare the transforms
    ssignal = Uinv @ signal
    hsignal = hermitian(S) @ ssignal
    initial_norm = np.linalg.norm(signal)
    initial_smoothness = TV3(signal, L)
    T = U @ S

    # NOTE: if the signal is already smooth, then we can use a smaller cutoff
    starting_smoothness = TV3(signal/initial_norm, L)
    if starting_smoothness < 1e-3:
        cutoff = 0.25

    # 2. Smoothness preserving hyperplane
    n1 = nodecimal(J @ np.abs(hsignal) ** 2).real
    # GFT Spectrum energy preserving hyperplane
    n2 = np.abs(hsignal) ** 2

    # 3. Take a random starting randomizer vector
    x = (2 * (np.random.random(len(U)) - 0.5) + 2j * (np.random.random(len(U)) - 0.5)) * rscale

    # 4. Set a fixed sign constraints on the randomizer
    if signfix is None:
        signfix = np.sign(np.diag(np.random.random(len(x)) - 0.5).astype(float))
        # signfix = np.diag(np.ones_like(x))
    if complexflag:
        phasefix = np.angle(x)

    # 5. Now iteratively project and see whether we end up satisfying both projection sets
    nrj_track = 1e10
    for k in range(n_iter):
        
        # Projection on P1 - P2
        x = generate_orthogonal_vector(x, n1)

        if spectreflag:
            x = generate_orthogonal_vector(x, n2)

        if complexflag:
            newr = np.sqrt(x + 1)
            newr = newr * np.exp(1j * phasefix)
        else:
            newr = np.sqrt(x + 1)
            newr = signfix @ newr

        if (k % (n_iter//10) == 0) and verbose:
            rsignal_verbose = T @ np.diag(newr) @ hsignal

            dotprod_n1 = np.abs(np.dot(x, n1))
            tmp_smooth = np.abs((initial_smoothness - TV3(rsignal_verbose, L)) / initial_smoothness)
            nrj = np.abs((initial_norm - np.linalg.norm(rsignal_verbose)) / initial_norm)
            print(f"P1: Iteration {k+1}:")
            print(f"Energy conservation {nrj} | Smoothness {tmp_smooth} | Orthogonality n1 {dotprod_n1}")

        rsignal = T @ np.diag(newr) @ hsignal
        new_norm = np.linalg.norm(rsignal)
        ratio_newnorm = np.abs((new_norm - initial_norm) / initial_norm)
        if np.abs(ratio_newnorm - nrj_track) < 1e-7 :
            if verbose: print("Early Stop")
            break
        else:
            nrj_track = ratio_newnorm

        if np.isnan(new_norm):
            raise ValueError('Anomaly in norm division, intermediate solution too close to 0')

        # Projection on P3
        newr = newr / new_norm * initial_norm
        if (k % (n_iter//10) == 0) and verbose:

            rsignal_verbose = T @ np.diag(newr) @ hsignal
            tmp_smooth = np.abs((initial_smoothness - TV3(rsignal_verbose, L)) / initial_smoothness)
            nrj = np.abs((initial_norm - np.linalg.norm(rsignal_verbose)) / initial_norm)
            print(f"P3: Iteration {k+1}:")
            print(f"Energy conservation {nrj} | Smoothness {tmp_smooth}")
            print("----------------------")

        x = np.abs(newr) ** 2 - 1
        
        if logs:
            tv_logs.append(np.abs((initial_smoothness - TV3(rsignal, L)) / initial_smoothness))
            nr_logs.append(np.abs((initial_norm - np.linalg.norm(rsignal)) / initial_norm))

    rsignal = T @ np.diag(newr) @ hsignal
    if np.abs((initial_smoothness - TV3(rsignal, L)) / initial_smoothness) > cutoff:
        if verbose:
            print(f"Smoothness optimality not reach: {np.round(np.abs((TV3(signal, L) - TV3(rsignal, L)) / TV3(signal, L)),3)}")
        if logs:
            return rsignal, None, signfix, tv_logs, nr_logs
        return rsignal, None, signfix
    if np.abs((initial_norm - np.linalg.norm(rsignal)) / initial_norm) > cutoff:
        if verbose:
            print(f"Energy optimality not reach: {np.round(np.abs((initial_norm - np.linalg.norm(rsignal)) / initial_norm),3)}")
        if logs:
            return rsignal, None, signfix, tv_logs, nr_logs
        return rsignal, None, signfix

    if logs:
        return rsignal, newr, signfix, tv_logs, nr_logs
    return rsignal, newr, signfix

def optimized_random_surrogate_POCS(signal:np.ndarray, L:np.ndarray, U:np.ndarray, Uinv:np.ndarray, 
                                    S:np.ndarray, J:np.ndarray, nrands:int=99, 
                                    n_iter:int=100, countoff:int=10000, signflag:Optional[np.ndarray]=False, 
                                    seed:int=99, verbose:bool=True, spectreflag:bool=False, cutoff:float=1e-4,
                                    strict:bool=True, ret_signfix:bool=False):
    """
    Generate null distribution from optimized (POCS) random surrogates

    Parameters:
    ----------
    signal: np.ndarray
        The input signal array
    L: np.ndarray
        The graph Laplacian matrix
    U: np.ndarray
        The graph Fourier basis
    Uinv: np.ndarray
        The inverse graph Fourier basis
    S: np.ndarray
        The graph Fourier coefficients of the signal
    J: np.ndarray
        The graph Fourier coefficients of the signal's Hilbert transform
    nrands: int, optional
        The number of random surrogates to generate (default 99)
    n_iter: int, optional
        The number of iterations for the POCS algorithm (default 100)
    countoff: int, optional
        The number of random seeds to try before giving up (default 10000)
    signflag: Optional[np.ndarray], optional
        Whether to fix the sign of the randomized surrogates (default None)
    seed: int, optional
        The random seed (default 99)
    verbose: bool, optional
        Whether to print progress (default True)
    spectreflag: bool, optional
        Whether to use the spectral randomizer (default False)
    cutoff: float, optional
        The cutoff value for the POCS algorithm (default 1e-4)
    strict: bool
        Only return randomized signal below cutoff
    ret_signfix: bool
        Return sign of randomizer

    Returns:
    --------
    ret: np.ndarray
        An array of shape (nrands, len(signal)) containing the optimized random surrogates
    """
    
    ret = [signal]
    
    # Use the initial random sign fixing properties of the randomizers
    if signflag is None:
        signflag = [None] * nrands
    
    signfixes = deepcopy(signflag)
        
    for sidx in tqdm(range(nrands), disable=not verbose):
        rseed = sidx * countoff
        for k in range(countoff):
            ssignal, converge_flag, signfix = pocs_randomizer(signal, L, U, Uinv, S, J, 
                                            1, rseed + k, n_iter=n_iter, 
                                            spectreflag=spectreflag, signfix=signflag[sidx],
                                            complexflag=False, verbose=False, cutoff=cutoff)
            if not (converge_flag is None):
                signfixes[sidx] = signfix
                break

        if converge_flag is None and strict:
            print("#####Perf####")
            print(f"randomized nrj: {np.linalg.norm(ssignal)} | nrj: {np.linalg.norm(signal)}")
            print(f"randomized TV: {TV3(ssignal, L)} | TV: {TV3(signal, L)}")
            raise TypeError('POCS algorithm failed to converge after maximum iterations - increase countoff')

        ret.append(ssignal)

    ret = np.array(ret)
    ret = ret[1:].real

    if ret_signfix:
        return ret, signfixes
    return ret

def pocs_randomizer_new(signal:np.ndarray, L:np.ndarray, U:np.ndarray, Uinv:np.ndarray, 
                    S:np.ndarray, J:np.ndarray, rscale:float, rseed:int, 
                    n_iter:int=100, cutoff:float=1e-4, signfix:Optional[np.ndarray]=None, 
                    spectreflag:bool=False, 
                    verbose:bool=False, logs:bool=False):
    """
    Find a randomizer satisfiying smoothness preservation and energy preservation of a given signal.
    Allow for complex signal output.
    Use two-convex sets consecutive projection to find the randomizer.

    Parameters:
    ----------
    signal (np.ndarray): The input signal.
    L (np.ndarray): Laplacian.
    U (np.ndarray): Eigenvectors of Laplacian.
    Uinv (np.ndarray): The inverse of matrix U.
    S (np.ndarray): Composie basis.
    J (np.ndarray): Composite eigenvalues.
    rscale (float): The scale factor for the random starting randomizer vector.
    rseed (int): The random seed to use.
    n_iter (int, optional): The number of iterations to perform. Defaults to 100.
    cutoff (float, optional): The cutoff threshold for smoothness and energy preservation. Defaults to 1e-4.
    signfix (Optional[np.ndarray], optional): The fixed sign constraints on the randomizer. Defaults to None.
    spectreflag (bool, optional): Whether to use the GFT spectrum energy preserving hyperplane. Defaults to False.
    verbose (bool, optional): Whether to print progress information. Defaults to False.
    logs (bool, optional): Whether to return logs of the optimization process. Defaults to False.

    Returns:
    --------
    rsignal (np.ndarray): The randomized signal.
    newr (np.ndarray): The final randomizer vector.
    tv_logs (List[float]): The logs of the total variation difference (if logs is True).
    nr_logs (List[float]): The logs of the energy difference (if logs is True).
    """

    if rseed != -1: np.random.seed(rseed)

    tv_logs = []
    nr_logs = []

    # 1. Prepare the transforms
    ssignal = Uinv @ signal
    hsignal = hermitian(S) @ ssignal
    initial_norm = np.linalg.norm(signal)
    initial_smoothness = TV3(signal, L)
    T = U @ S

    M = hermitian(np.diag(hsignal)) @ hermitian(T) @ T @ np.diag(hsignal)
    Ma, Vp = compute_basis(M, verbose=False)
    Ma, Vp = Ma.real, Vp.real
    vdir = Vp * initial_norm**2/np.linalg.norm(Vp)**2

    # NOTE: if the signal is already smooth, then we can use a smaller cutoff
    starting_smoothness = TV3(signal/initial_norm, L)
    if starting_smoothness < 1e-3:
        cutoff = 0.25

    # 2. Smoothness preserving hyperplane
    n1 = nodecimal(J @ np.abs(hsignal) ** 2).real
    # GFT Spectrum energy preserving hyperplane
    n2 = np.abs(hsignal) ** 2
    n3 = Vp

    # 3. Take a random starting randomizer vector
    # x = (2 * (np.random.random(len(U)) - 0.5) + 2j * (np.random.random(len(U)) - 0.5)) * rscale
    x = 2 * (np.random.random(len(U)) - 0.5)

    # 4. Set a fixed sign constraints on the randomizer
    if signfix is None:
        signfix = np.sign(np.diag(np.random.random(len(x)) - 0.5).astype(float))

    # 5. Now iteratively project and see whether we end up satisfying both projection sets
    nrj_track = 1e10
    for k in range(n_iter):
        
        # Projection set P1 - P2 - Smoothness and Spectral conservation
        x = generate_orthogonal_vector(x, n1)

        if spectreflag:
            x = generate_orthogonal_vector(x, n2)

        # print(x)
        newr = np.sqrt(x + 1)
        newr = signfix @ newr

        if (k % (n_iter//10) == 0) and verbose:
            dotprod_n1 = np.abs(np.dot(x, n1))
            # dotprod_n2 = np.abs(np.dot(x, n2))

            rsignal = T @ np.diag(newr) @ hsignal
            tmp_smooth = np.abs((initial_smoothness - TV3(rsignal, L)) / initial_smoothness)
            nrj = np.abs((initial_norm - np.linalg.norm(rsignal)) / initial_norm)
            print(f"P1: Iteration {k+1}:")
            print(f"Orthogonality n1 {dotprod_n1} | energy conservation {nrj} | Smoothness {tmp_smooth}")
            # print("----------------------")

        # Projection set P3 - Energy Conservation
        xp = np.abs(hermitian(Ma) @ newr) ** 2 - vdir # Forward

        # normr = np.linalg.norm(newr)
        # normv = np.linalg.norm(vdir)
        # factor = ((alpha * normr) ** 2 - normv) / ((normr) ** 2 - normv)
        # print(factor)
        xp = generate_orthogonal_vector(xp, n3)
        newr = Ma @ np.abs(np.sqrt(xp + vdir)) # Backward
        newr = signfix @ newr

        rsignal = T @ np.diag(newr) @ hsignal
        new_norm = np.linalg.norm(rsignal)
        ratio_newnorm = np.abs((new_norm - initial_norm) / initial_norm)
        if np.abs(ratio_newnorm - nrj_track) < 1e-7:
            break
        else:
            nrj_track = ratio_newnorm

        alpha = (initial_norm / new_norm)
        newr = alpha * newr

        if np.isnan(new_norm):
            raise ValueError('Anomaly in norm division, intermediate solution too close to 0')
        if (k % (n_iter//10) == 0) and verbose:
            dotprod_n3 = np.abs(np.dot(xp, n3))

            rsignal = T @ np.diag(newr) @ hsignal
            tmp_smooth = np.abs((initial_smoothness - TV3(rsignal, L)) / initial_smoothness)
            nrj = np.abs((initial_norm - np.linalg.norm(rsignal)) / initial_norm)
            print(f"P3: Iteration {k+1}:")
            print(f"Orthogonality n3 {dotprod_n3} | energy conservation {nrj} | Smoothness {tmp_smooth}")
            print("----------------------")

        x = np.abs(newr) ** 2 - 1
        if logs:
            tv_logs.append(np.abs((initial_smoothness - TV3(rsignal, L)) / initial_smoothness))
            nr_logs.append(np.abs((initial_norm - np.linalg.norm(rsignal)) / initial_norm))

    # newr = signfix @ newr
    rsignal = T @ np.diag(newr) @ hsignal
    if np.abs((initial_smoothness - TV3(rsignal, L)) / initial_smoothness) > cutoff:
        if verbose:
            print(f"Smoothness optimality not reach: {np.round(np.abs((TV3(signal, L) - TV3(rsignal, L)) / TV3(signal, L)),3)}")
        if logs:
            return rsignal, None, signfix, tv_logs, nr_logs
        return rsignal, None, signfix
    if np.abs((initial_norm - np.linalg.norm(rsignal)) / initial_norm) > cutoff:
        if verbose:
            print(f"Energy optimality not reach: {np.round(np.abs((initial_norm - np.linalg.norm(rsignal)) / initial_norm),3)}")
        if logs:
            return rsignal, None, signfix, tv_logs, nr_logs
        return rsignal, None, signfix

    if logs:
        return rsignal, newr, signfix, tv_logs, nr_logs
    return rsignal, newr, signfix


def optimized_random_surrogate_POCS_new(signal:np.ndarray, L:np.ndarray, U:np.ndarray, Uinv:np.ndarray, 
                                    S:np.ndarray, J:np.ndarray, nrands:int=99, 
                                    n_iter:int=100, countoff:int=10000, signflag:Optional[np.ndarray]=False, 
                                    seed:int=99, verbose:bool=True, spectreflag:bool=False, cutoff:float=1e-4,
                                    strict:bool=True, ret_signfix:bool=False):
    """
    Generate null distribution from optimized (POCS) random surrogates

    Parameters:
    ----------
    signal: np.ndarray
        The input signal array
    L: np.ndarray
        The graph Laplacian matrix
    U: np.ndarray
        The graph Fourier basis
    Uinv: np.ndarray
        The inverse graph Fourier basis
    S: np.ndarray
        The graph Fourier coefficients of the signal
    J: np.ndarray
        The graph Fourier coefficients of the signal's Hilbert transform
    nrands: int, optional
        The number of random surrogates to generate (default 99)
    n_iter: int, optional
        The number of iterations for the POCS algorithm (default 100)
    countoff: int, optional
        The number of random seeds to try before giving up (default 10000)
    signflag: Optional[np.ndarray], optional
        Whether to fix the sign of the randomized surrogates (default None)
    seed: int, optional
        The random seed (default 99)
    verbose: bool, optional
        Whether to print progress (default True)
    spectreflag: bool, optional
        Whether to use the spectral randomizer (default False)
    cutoff: float, optional
        The cutoff value for the POCS algorithm (default 1e-4)
    strict: bool
        Only return randomized signal below cutoff
    ret_signfix: bool
        Return sign of randomizer

    Returns:
    --------
    ret: np.ndarray
        An array of shape (nrands, len(signal)) containing the optimized random surrogates
    """
    
    ret = [signal]
    
    # Use the initial random sign fixing properties of the randomizers
    if signflag is None:
        signflag = [None] * nrands
    
    signfixes = deepcopy(signflag)
        
    for sidx in tqdm(range(nrands), disable=not verbose):
        rseed = sidx * countoff
        for k in range(countoff):
            ssignal, converge_flag, signfix = pocs_randomizer_new(signal, L, U, Uinv, S, J, 
                                            1, rseed + k, n_iter=n_iter, 
                                            spectreflag=spectreflag, signfix=signflag[sidx],
                                            verbose=False, cutoff=cutoff)
            if not (converge_flag is None):
                signfixes[sidx] = signfix
                break

        if converge_flag is None and strict:
            print("#####Perf####")
            print(f"randomized nrj: {np.linalg.norm(ssignal)} | nrj: {np.linalg.norm(signal)}")
            print(f"randomized TV: {TV3(ssignal, L)} | TV: {TV3(signal, L)}")
            raise TypeError('POCS algorithm failed to converge after maximum iterations - increase countoff')

        ret.append(ssignal)

    ret = np.array(ret)
    ret = ret[1:].real

    if ret_signfix:
        return ret, signfixes
    return ret


##################################
#### NON DIRECT RANDOMIZATION ####
##################################
def naive_random_surrogate(arr: np.ndarray, nrands: int = 99, seed: int = 99):
    """
    Generate nrands number of naive random surrogates for the input array arr.

    The surrogates are generated by randomly permuting the input array arr.

    Parameters
    ----------
    arr : np.ndarray
        Input array to generate surrogates for
    nrands : int, optional
        Number of random surrogates to generate. Default is 99. 
    seed : int, optional 
        Random seed for reproducibility. Default is 99.

    Returns
    -------
    ret : np.ndarray
        Array of shape (nrands, len(arr)) containing the random surrogates.
    """
    np.random.seed(seed)

    ret = np.zeros((nrands, len(arr)))
    for i in range(nrands):
        ret[i] = arr[np.random.permutation(len(arr))]
    return ret


def undir_random_surrogate(signal: np.ndarray, U: np.ndarray, Uinv: np.ndarray,
                            nrands: int = 99, rseed: int = 99):
    """
    Undirected informed generation of surrogate signals

    Parameters
    ----------
    signal : np.ndarray
        The input signal array
    U : np.ndarray 
        The graph Fourier basis
    Uinv : np.ndarray
        The inverse graph Fourier basis  
    nrands : int, optional
        The number of random surrogates to generate (default 99)
    rseed : int, optional 
        The random seed (default 99)

    Returns
    -------
    ret : list
        A list containing the randomized surrogate signals
    """

    np.random.seed(rseed)
    ssignal = GFT(signal, U, Uinv=Uinv)

    ret = []
    for _ in tqdm(range(nrands), disable=True):
        # Initialize the randomizer
        R = np.diag(np.sign(np.random.random(len(U)) - 0.5))

        rand = R @ ssignal
        randomized = inverseGFT(rand, U)
        ret.append(randomized)

    return ret