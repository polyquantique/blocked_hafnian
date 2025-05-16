## Code used to obtain the results in ["Simulating lossy and partially distinguishable quantum optical circuits: theory, algorithms and applications to experiment validation and state preparation"](https://arxiv.org/abs/2412.17742).

* `blhaf_gkp_density_matrix.py` computes the density matrix of an approximate GKP state using the blocked loop Hafnian technique, and measures the time of computation. This code was used to obtain Figures 5, 6 and 7 of the manuscript.

* `combinatorial_gkp_density_matrix.py` computes the density matrix of an approximate GKP state using the combinatorial method, and measures the time of computation. This code was used to obtain Figure 5 of the manuscript.

* `ffunc_total_num_dist.py` computes the total photon-number distribution of a GBS setup with uniform squeezing and uniform transmission losses using the f-function technique. This code was used to obtain Figures 8 and 9 of the manuscript.

* `phase_space_total_num_dist.py` computes the total photon-number distribution of a GBS setup with uniform squeezing and uniform transmission losses using the positive $P$-distribution technique.

* `borealis_ffunc_total_num_dist.py` computes the total photon-number distribution of the Borealis experiment using the f-function technique, and measures the time of computation. This code was used to obtain Figure 10 of the manuscript. This code makes use of the transmission matrix and squeezing parameters of the Borealis experiment, which are available [here](https://github.com/XanaduAI/xanadu-qca-data).

* `borealis_phase_space_total_num_dist.py` computes the total photon-number distribution of the Borealis experiment using the positive $P$-distribution technique. This code makes use of the transmission matrix and squeezing parameters of the Borealis experiment, which are available [here](https://github.com/XanaduAI/xanadu-qca-data).

* `cov_gkp.npy` covariance matrix of the Gaussian state used for generating the approximate GKP states.
