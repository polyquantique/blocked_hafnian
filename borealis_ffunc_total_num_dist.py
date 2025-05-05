"""
Functions for computing the total photon number distribution of the
Borealis experiment using the f-function.
"""

from timeit import default_timer
import numpy as np
from thewalrus.symplectic import squeezing, passive_transformation
from thewalrus.quantum import Qmat, Xmat
from thewalrus._hafnian import f_from_matrix


def blocked_total_photon_probabilities(cov, max_photons, det_block):
    """Computes total photon number probabilities for a Gaussian state

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


bor_sqz_par = np.load("./r.npy")
bor_transmission_mat = np.load("./T.npy")

num_modes = bor_transmission_mat.shape[0]
max_photons = 220
blocks = list(range(0, num_modes))

hbar = 2
cov_init = (hbar / 2) * squeezing(bor_sqz_par) @ squeezing(bor_sqz_par)
_, cov_final = passive_transformation(np.zeros(len(cov_init)), cov_init, bor_transmission_mat)

tic = default_timer()
blk_tp_dist = blocked_total_photon_probabilities(cov_final, max_photons, blocks)
toc = default_timer()

np.save("./borealis_prob_blk.npy", blk_tp_dist)
print(toc - tic)
