"""
Functions for computing the approximate GKP density matrix using the blocked
blocked loop hafnian method.
"""

import argparse
from timeit import default_timer
import numpy as np
from scipy.linalg import block_diag
from thewalrus.decompositions import williamson, blochmessiah
from thewalrus.symplectic import passive_transformation
from thewalrus.internal_modes.fock_density_matrices import density_matrix_single_mode


def perm_matrix(M, K):
    """Computes a permutation matrix for the covariance matrix.
    Its purpose is to make internal modes of the same external mode
    adjacent to each other.

    Args:
        M (int): Number of external modes.
        K (int): Number of internal modes per external mode.

    Returns:
        (array): Permutation matrix of shape (2 * M * K, 2 * M * K).
    """
    first_indices = np.arange(K * M)
    first_indices = np.reshape(first_indices, (K, M))
    first_indices = first_indices.flatten(order="F")
    second_indices = np.arange(K * M, 2 * K * M)
    second_indices = np.reshape(second_indices, (K, M))
    second_indices = second_indices.flatten(order="F")
    indices = np.arange(2 * K * M)
    pindices = np.concatenate((first_indices, second_indices))

    perm_mat = np.zeros((2 * M * K, 2 * M * K))

    for k, l in zip(indices, pindices):
        perm_mat[k, l] = 1

    return perm_mat


def second_sqz_par(ovl, sqz_par):
    """Computes the squeezing parameters of the second internal mode

    Args:
        ovl (float): 'Overlap' parameter. It is the probability of obtaining the one
        photon Fock state when it is heralded from a two-mode squeezed state with
        two internal modes.
        sqz_par (array): Array of the squeezing parameters of the first internal mode.

    Returns:
        (array): Squeezing parameters of the second internal mode.
    """
    return np.arctanh(np.sqrt((1 - ovl) / ovl) * np.tanh(sqz_par))


def modified_gkp_cov(sqz_par, U, eff, ovl, hbar=2):
    """Computes the modified gkp covariance matrix. This is a system of single mode
    squeezed states with squeezing parameters sqz_par that are sent into a lossy
    interferometer. This interferometer is obtained by applying a finite transmission
    efficiency eff over a lossless interferometer U. Each squeezed state has two
    internal modes per external mode. The overlap between the internal modes is ovl.

    Args:
        sqz_par (array): Squeezing parameters of the first internal modes (length M).
        U (array): M x M unitary matrix representing a lossless interferometer.
        eff (array): Transmission efficiencies of the external modes (length M).
        ovl (float): 'Overlap' parameter. It is the probability of obtaining the one
        photon Fock state when it is heralded from a two-mode squeezed state with
        two internal modes.
        hbar (float): Value of hbar in the commutation relation [q, p] = i hbar.

    Returns:
        (array): Modified covariance matrix of size 4M x 4M.
    """
    K = 2
    M = len(U)
    tsqz_par = np.concatenate((sqz_par, second_sqz_par(ovl, sqz_par)))
    tcov_0 = np.concatenate((np.exp(2 * tsqz_par), np.exp(-2 * tsqz_par)))
    tcov_0 = (hbar / 2) * np.diag(tcov_0)
    tmu = np.zeros(len(tcov_0))

    tU = block_diag(U, U)
    _, tcov = passive_transformation(tmu, tcov_0, tU)

    L = np.array([])
    for i in range(M):
        L = np.append(L, np.array(K * [np.sqrt(eff[i])]))
    L = np.diag(np.append(L, L))
    ttcov = L @ tcov @ L.T + (hbar / 2) * (np.eye(tcov.shape[0]) - L @ L.T)

    return ttcov


parser = argparse.ArgumentParser(
    description="Compute density matrix of GKP states and measure time"
)
parser.add_argument("--n", type=int, help="Cutoff")
parser.add_argument("--ovl", type=float, help="Overlap of internal modes")
parser.add_argument("--eta", type=float, help="Transmission efficiency")
args = parser.parse_args()

ovl, eta = args.ovl, args.eta
cov_gkp = np.load("./cov_gkp.npy")
M, K = cov_gkp.shape[0] // 2, 2
eff = eta * np.ones(M)
cutoff = args.n

_, wV = williamson(cov_gkp)
V, sval, W = blochmessiah(wV)
sval = np.diag(sval)

U = V[M:, M:] - 1j * V[:M, M:]
sqz_par = np.array([np.log(sval[k]) for k in range(M)])

perm = perm_matrix(M, K)
mod_cov = modified_gkp_cov(sqz_par, U, eff, ovl)
pmod_cov = perm @ mod_cov @ perm.T
pmod_cov = np.array(pmod_cov, dtype=np.float64)


tic = default_timer()
rho = density_matrix_single_mode(pmod_cov, pattern={0: 5, 1: 7}, cutoff=cutoff)
toc = default_timer()
np.save(f"./gkp_rho_cutoff_{cutoff}_ovl_{int(100 * ovl)}_eta_{int(10 * eta)}", rho)
print(cutoff, toc - tic)
