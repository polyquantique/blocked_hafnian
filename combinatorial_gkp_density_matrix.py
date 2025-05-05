"""
Functions for computing the approximate GKP density matrix using the
combinatorial method.
"""

import argparse
from timeit import default_timer
from itertools import product
import numpy as np
from scipy.linalg import block_diag
from sympy.utilities.iterables import partitions, multiset_permutations
from thewalrus.decompositions import williamson, blochmessiah
from thewalrus.symplectic import passive_transformation
from thewalrus.quantum import density_matrix_element, reduced_gaussian
import dask


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


def to_first_mode_perm(M, K):
    """Permutation matrix that moves the last external mode to the first position.
    The internal modes are moved accordingly.

    Args:
        M (int): Number of external modes.
        K (int): Number of internal modes per external mode

    Returns:
        (array): Permutation matrix
    """
    ext_indices = list(range(M))
    swap_ext_indices = ext_indices[M - 1 :] + ext_indices[: M - 1]
    all_indices = []
    for g in ext_indices:
        all_indices = all_indices + list(range(g * K, (g + 1) * K))
    swap_all_indices = []
    for g in swap_ext_indices:
        swap_all_indices = swap_all_indices + list(range(g * K, (g + 1) * K))

    ext_indices_2 = list(range(M, 2 * M))
    swap_ext_indices_2 = ext_indices_2[M - 1 :] + ext_indices_2[: M - 1]
    all_indices_2 = []
    for g in ext_indices_2:
        all_indices_2 = all_indices_2 + list(range(g * K, (g + 1) * K))
    swap_all_indices_2 = []
    for g in swap_ext_indices_2:
        swap_all_indices_2 = swap_all_indices_2 + list(range(g * K, (g + 1) * K))

    perm_mat = np.zeros((2 * M * K, 2 * M * K))

    for k, l in zip(all_indices + all_indices_2, swap_all_indices + swap_all_indices_2):
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


def compatible_patterns(num_photons, block_size):
    """Computes all the patterns of length block_size that are compatible
    with a given number of photons, i.e., the total number of photons in
    the patterns is num_photons.
    WARNING: The size of the output increases combinatorially with the input.

    Args:
        num_photons (int): Total number of photons in the detction patterns.
        block_size (int): Size of the detection pattern.

    Returns:
        (list): Array of detection pattern compatible with the total photon number.
    """

    all_patterns = []
    for k in partitions(num_photons, m=block_size):
        cles = list(k.keys())
        vals = list(k.values())
        pattern = []
        for l in range(len(k)):
            pattern = pattern + ([cles[l]] * vals[l])

        if len(pattern) < block_size:
            pattern = pattern + ([0] * (block_size - len(pattern)))

        for ppart in multiset_permutations(pattern):
            all_patterns.append(ppart)

    return all_patterns


def combinatorial_gkp_density_matrix_element(m, n, mu, cov):
    """Computes the density matrix elements of a GKP state in the presence
    of two internal modes. The state is hrealded by measuring p1=5 photons
    in the external mode 1, and p2=7 photons in the external mode 2. The second
    internal mode of the external mode 0 is projected to the vacuum. The resulting
    element has the form (m|rho_gkp|n). This uses the combinatorial (or naive) method.

    Args:
        m (int): First element of the Fock state basis.
        n (int): Second element of the Fock state basis.
        mu (array): Vector of first moments of the heralding Gaussian state.
        it has length 2KM.
        cov (array): Covariance matrix of the of the heralding Gaussian state.
        it has size (2KM) x (2KM).

    Returns:
        (complex): Density matrix element of the unnormalized heralded state.
    """

    K = 2
    M = len(cov) // (2 * K)
    p1, p2 = 5, 7
    patterns_p1 = compatible_patterns(p1, K)
    patterns_p2 = compatible_patterns(p2, K)
    perm = to_first_mode_perm(M, K)
    remaining_modes = [0] + list(range(2, K * M))

    cov = perm @ cov @ perm.T
    rmu, rcov = reduced_gaussian(mu, cov, modes=remaining_modes)

    elem_gkp = 0
    for patt1, patt2 in product(patterns_p1, patterns_p2):
        mv = [m] + patt1 + patt2
        nv = [n] + patt1 + patt2
        elem_gkp += density_matrix_element(rmu, rcov, mv, nv)

    return elem_gkp


def combinatorial_gkp_density_matrix(mu, cov, cutoff):
    """Computes the density matrix of a GKP state in the presence of two internal modes.
    The state is hrealded by measuring p1=5 photons in the external mode 1, and p2=7
    photons in the external mode 2. The second internal mode of the external mode 0 is
    projected to the vacuum. The computation of the matrix elements is not optimal.

    Args:
        mu (array): Vector of first moments of the heralding Gaussian state.
        it has length 2KM.
        cov (array): Covariance matrix of the of the heralding Gaussian state.
        it has size (2KM) x (2KM).
        cutoff (int): Cutoff of the Fock states basis.

    Returns:
        (array): Density matrix of the unnormalized heralded state.
    """

    compute_list = []
    list_modes = [[m, n] for m in range(cutoff) for n in range(m, cutoff) if ((m + n) % 2) == 0]
    rho = np.zeros((cutoff, cutoff), dtype=np.complex128)

    for modes in list_modes:
        compute_list.append(
            dask.delayed(combinatorial_gkp_density_matrix_element)(modes[1], modes[0], mu, cov)
        )

    elements = dask.compute(*compute_list, num_workers=10)

    for k in range(len(elements)):
        m, n = list_modes[k]
        rho[m, n] = elements[k]
        rho[n, m] = (elements[k]).conj()

    return rho


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
pmod_mu = np.zeros(len(pmod_cov))


tic = default_timer()
rho = combinatorial_gkp_density_matrix(pmod_mu, pmod_cov, cutoff=cutoff)
toc = default_timer()
np.save(f"./slow_gkp_rho_cutoff_{cutoff}_ovl_{int(100 * ovl)}_eta_{int(10 * eta)}", rho)
print(cutoff, ovl, eta, toc - tic)
