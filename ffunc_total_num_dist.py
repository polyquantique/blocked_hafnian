"""
Functions for computing the total photon number distribution using the
f-function method.
"""

import argparse
from timeit import default_timer
import numpy as np
from scipy.stats import unitary_group
from thewalrus.symplectic import squeezing, passive_transformation
from thewalrus.quantum import Qmat, Xmat
from thewalrus._hafnian import f_from_matrix


def blocked_total_photon_probabilities(cov, max_photons, det_block):
    """Computes total photon number probabilities for input states sent into a lossy interferometer

    Args:
        cov (array): covariance matrix of the output state
        max_photons (int): maximum number of photons to compute probs
        det_block (list): Modes where the photons are detcted.


    Returns:
        (array): total photon number probabilities

    """
    num_modes = len(cov) // 2
    mX, mQ = Xmat(num_modes), Qmat(cov)
    mA = mX @ (np.eye(2 * num_modes) - np.linalg.inv(mQ))
    vacuum_prob = np.sqrt(np.linalg.det(np.linalg.inv(mQ)))
    add_block = np.array(det_block) + num_modes
    reps = det_block + add_block.tolist()
    rX = mX[np.ix_(reps, reps)]
    rA = mA[np.ix_(reps, reps)]

    return np.real_if_close(vacuum_prob * f_from_matrix(rX @ rA, 2 * max_photons))


parser = argparse.ArgumentParser(description="Computes total photon number probs.")
parser.add_argument("--sqz", type=float, help="Squeezing parameter")
parser.add_argument("--eta", type=float, help="Transmission efficiency")
parser.add_argument("--M", type=int, help="Number of modes")
parser.add_argument("--N", type=int, help="Maximum number of photons")
args = parser.parse_args()

input_sqz = np.array([args.sqz] * args.M)
t_mat = args.eta * unitary_group(dim=args.M, seed=1990).rvs()

block = list(np.arange(args.M))

cov_init = squeezing(input_sqz) @ squeezing(input_sqz)
_, cov_final = passive_transformation(np.zeros(len(cov_init)), cov_init, t_mat)

tic = default_timer()
probs = blocked_total_photon_probabilities(cov_final, args.N, block)
toc = default_timer()

np.save(f"probs_blk_M_{args.M}_N_{args.N}", probs)
np.save(f"t_mat_M_{args.M}", t_mat)
print(args.M, toc - tic)
