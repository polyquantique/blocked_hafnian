"""
Functions for computing total photon number probabilities using the
positive P phase space representation.
"""

import argparse
from timeit import default_timer
import numpy as np
from scipy.stats import unitary_group
from numba import jit


@jit(nopython=True)
def recursive_exp(n, x):
    """Recursive total photon number observable for the computation of probabilities

    Args:
        n (int): order of the function
        x (complex): argument of the exponent
    """
    if n == 0:
        return x

    return recursive_exp(n - 1, x) - np.log(x) + np.log(n)


@jit(nopython=True)
def recursive_total_photon_probabilities(
    phn, chn, t_matrix, mphotons, det_block, num_samples, seed=1990
):
    """Computes total photon number probabilities for input states sent into a lossy interferometer

    Args:
        phn (array): mean photon numbers of input modes
        chn (array): coherences of input modes
        t_matrix (array): transfer matrix
        mphotons (int): Maximum number of photons to compute probs
        det_block (list): Modes where the photons are detected.
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
    for _ in range(num_samples):
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


parser = argparse.ArgumentParser(description="Computes total photon number probs.")
parser.add_argument("--sqz", type=float, help="Squeezing parameter")
parser.add_argument("--eta", type=float, help="Transmission efficiency")
parser.add_argument("--M", type=int, help="Number of modes")
parser.add_argument("--N", type=int, help="Maximum number of photons")
args = parser.parse_args()

input_sqz = np.array([args.sqz] * args.M)
phn = np.sinh(input_sqz) ** 2
chn = 0.5 * np.sinh(2 * input_sqz)
t_mat = args.eta * unitary_group(dim=args.M, seed=1990).rvs()

block = list(np.arange(args.M))

tic = default_timer()
probs = recursive_total_photon_probabilities(phn, chn, t_mat, args.N, block, int(2.4e6))
toc = default_timer()

np.save(f"probs_ps_M_{args.M}_N_{args.N}", probs)
np.save(f"t_mat_M_{args.M}", t_mat)
print(args.M, toc - tic)
