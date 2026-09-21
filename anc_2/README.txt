===========================================================================
  ANCILLARY DATA: symmetric seesaw for the depolarizing and amplitude
  damping channels
===========================================================================

Contents
--------
  depolarizing_fidelity_n1-20.npz                fidelity vs p, n = 1 ... 20  (figure data)
  amplitude_damping_fidelity_n1-20.npz           fidelity vs gamma, n = 1 ... 20 (figure data)
  verify_results.py                              checks the two files above

  depolarizing_operators_n1-20.npz     optimal operators, one parameter value
  amplitude_damping_operators_n1-20.npz
  verify_all_n.py                                checks the two operator files, for every n

Dependencies: numpy, scipy.

---------------------------------------------------------------------------
FIDELITY SWEEPS  (*_fidelity_n1-20.npz)
---------------------------------------------------------------------------
These carry the curves plotted in the paper: a (20, 50) array fid_sym of entanglement
fidelities, n = 1 ... 20 against 50 values of the channel parameter, plus the analytic
single-use and reference-code benchmarks.  See verify_results.py for the key list and the
consistency checks.

---------------------------------------------------------------------------
OPERATOR ARCHIVES  (*_operators_n1-20.npz)
---------------------------------------------------------------------------
The encoder/decoder pair achieving the fidelity at one parameter value per channel
(depolarizing p = 0.151020, amplitude damping gamma = 0.510204; both are grid points of the
sweeps above, so the values can be compared directly with fid_sym).

The encoder is an exact isometry V : C^2 -> Sym^n(C^2), stored as its (n+1) x 2 matrix of Dicke
amplitudes -- see the discussion in the superactivation ancillary README.  These channels are
unflagged, so there is a single sector k = 0 and the decoder is indexed by one Young diagram.

Keys per n
  n<N>_fidelity, n<N>_encoder_isometry_dicke,
  n<N>_decoder_0_labels, n<N>_decoder_0_dims, n<N>_decoder_0_kept,
  n<N>_decoder_0_block_<j>, n<N>_alice_pov_0_block_<j>

---------------------------------------------------------------------------
HOW TO VERIFY
---------------------------------------------------------------------------
    python verify_results.py     # the fidelity sweeps
    python verify_all_n.py       # the operator archives, every n

verify_all_n.py checks that each encoder is an exact isometry, that every decoder block is
positive semidefinite and unital, and that the fidelity recomputed from the stored operators
matches the stored value.

---------------------------------------------------------------------------
NOTE ON THE LARGE-n DEPOLARIZING POINTS
---------------------------------------------------------------------------
At n = 18, 19, 20 the depolarizing fidelities obtained here sit slightly below the values in
depolarizing_fidelity_n1-20.npz (by 4e-4 to 2.5e-3).  This is a local-optimum effect of the
seesaw and not a consequence of restricting the encoder to be isometric: re-running those
points with the encoder completely unconstrained reproduces the same values, and the
unconstrained optimum is itself found to be an isometry into the symmetric subspace (Choi rank
one, all weight on lambda = (n)).  The depolarizing channel is highly degenerate at large n,
where the pseudo-inverse square roots inside the power iteration are poorly conditioned.
===========================================================================
