"""
Functions for computing the total photon number distribution of the Borealis experiment
using the positive P phase space representation.
"""

from timeit import default_timer
import numpy as np
from numba import jit, prange


@jit(nopython=True)
def recursive_exp(n, x):
    """Function for recursive computation of probabilities

    Args:
        n (int): order of the function
        x (complex): argument of the exponent
    """
    if n == 0:
        return x
    return recursive_exp(n - 1, x) - np.log(x) + np.log(n)


@jit(nopython=True, parallel=True)
def recursive_total_photon_probabilities(
    phn, chn, t_matrix, mphotons, det_block, num_samples, seed=1990
):
    """Computes total photon number probabilities for input states sent into a lossy interferometer

    Args:
        phn (array): mean photon numbers of input modes
        chn (array): coherences of input modes
        t_matrix (array): transfer matrix
        mphotons (int): Maximum number of photons to compute probs
        det_block (list): Modes where the photons are detcted.
        num_samples (int): number of samples
        seed (int): seed of the random number generator

    Returns:
        (array): total photon number probabilities

    """
    np.random.seed(seed)
    num_modes = max(t_matrix.shape)
    num_input = min(t_matrix.shape)

    mask_det = np.array([1 if k in det_block else 0 for k in range(num_modes)], dtype=np.complex128)
    mask_ndet = np.array(
        [0 if k in det_block else 1 for k in range(num_modes)], dtype=np.complex128
    )

    drp = np.array([(0.5 * np.complex128(phn[i] + chn[i])) ** 0.5 for i in range(num_input)])
    drm = np.array([(0.5 * np.complex128(phn[i] - chn[i])) ** 0.5 for i in range(num_input)])

    res = np.zeros(mphotons + 1, dtype=np.complex128)
    for _ in prange(num_samples):
        wrp = np.array([np.random.normal() for _ in range(num_input)])
        wrm = np.array([np.random.normal() for _ in range(num_input)])
        alpha = t_matrix @ (drp * wrp + 1j * drm * wrm)
        beta = t_matrix.conj() @ (drp * wrp - 1j * drm * wrm)

        nobs = alpha * beta
        nb1 = mask_det @ nobs
        nb2 = mask_ndet @ nobs

        expo = np.array([recursive_exp(m, nb1) for m in range(mphotons + 1)])
        res = res + np.exp(-expo) * np.exp(-nb2)

    return res.real / num_samples


bor_sqz_par = np.load("./r.npy")
bor_transmission_mat = np.load("./T.npy")

num_modes = bor_transmission_mat.shape[0]
max_photons = 220
blocks = list(range(0, num_modes))

phn = np.sinh(bor_sqz_par) ** 2
chn = 0.5 * np.sinh(2 * bor_sqz_par)

tic = default_timer()
ps_tp_dist = recursive_total_photon_probabilities(
    phn, chn, bor_transmission_mat, 220, blocks, int(2.4e6)
)
toc = default_timer()

np.save("./borealis_prob_ps.npy", ps_tp_dist)
print(toc - tic)
