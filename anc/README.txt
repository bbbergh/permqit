===========================================================================
  ANCILLARY DATA: Optimal isometric encoders and decoders, n = 1 ... 20
===========================================================================

Contents
--------
  superactivation_encoders_n1-20.npz    encoder + fidelity, every n   (18 kB, tracked in git)
  superactivation_operators_n1-20.npz   encoder + decoders + Alice-POV + fidelity, every n
                                        (86 MB; distributed separately, see below)
  verify_all_n.py                       checks all of the above, for every n
  superactivation_operators_n17.npz               (earlier release) n = 17 only
  superactivation_operators_n17_aux.npz           (earlier release) n = 17, with Alice-POV blocks
  verify_operators.py                             (earlier release) checks the n = 17 files

Dependencies for either verifier: Python >= 3.10, numpy, scipy.  Nothing else.

The *_encoders_*.npz file holds only the encoders and the fidelities and is small enough to
version-control; it is all that is needed to reproduce the results, because for a fixed encoder
the optimal decoder is the solution of the recovery SDP and carries no extra information.  The
full *_operators_*.npz archives additionally store the decoder and Alice-POV blocks so that
verify_all_n.py can recompute every fidelity with numpy alone; they are too large for git and
are distributed as a release asset / Zenodo deposit.

---------------------------------------------------------------------------
CHANNEL MODEL
---------------------------------------------------------------------------
Each channel use N : M_2 -> M_2 otimes M_2 (output = Z otimes B) is a flagged mixture

  N(rho) = (1/2) (|0><0|_Z otimes id(rho)_B) + (1/2) (|1><1|_Z otimes P_p(rho)_B),

with P_p the Pauli channel ((1-p)/2, p/2, p/2, (1-p)/2) -- equivalently the binary symmetric
channel with crossover p = 1/(1+sqrt(2)) ~ 0.41421 -- and the classical flag Z delivered to the
receiver.  For n uses the flag string reveals the number k of "Pauli" events, so the n-use
channel splits into n+1 sectors N_k = P_p^{otimes k} otimes id^{otimes (n-k)}, each decoded
separately:

  F = sum_{k=0}^{n} C(n,k) (1/2)^n F_D[k].

Non-asymptotic superactivation is established when F > 0.75.

---------------------------------------------------------------------------
THE ENCODER IS AN EXACT ISOMETRY
---------------------------------------------------------------------------
For every n the encoder is E(rho) = V rho V^dagger with V : C^2 -> Sym^n(C^2) an isometry, so
its Choi state is pure.  A permutation-invariant encoder is an isometry precisely when its Choi
matrix has rank one, which forces a single Young diagram with multiplicity f_lambda = 1; for
n > d_A = 2 the antisymmetric diagram is absent, leaving lambda = (n) uniquely.  Hence

    isometric + permutation invariant  <=>  encodes into the symmetric subspace.

Tr_R of that Choi state is the rank-two, flat state on a two-dimensional subspace of Sym^n --
exactly the class of inputs used by Agarwal et al. [arXiv:2605.09138].  The encoder is therefore
fully described by the (n+1) x 2 matrix of Dicke amplitudes stored below.

---------------------------------------------------------------------------
FILE LAYOUT: superactivation_operators_n1-20.npz
---------------------------------------------------------------------------
Scalar keys
  channel, param (= p), q (= 1/2), d_R (= 2), n_values, fidelities

Per n
  n<N>_fidelity                  entanglement fidelity at n = N
  n<N>_encoder_isometry_dicke    (N+1) x d_R isometry V, columns V|i>, in the Dicke basis
                                 |D^N_w>, w = 0 ... N
  n<N>_decoder_<k>_labels        (mu_k | mu_{N-k}) label of each block of sector k
  n<N>_decoder_<k>_dims          dimension of each block (0 for empty blocks)
  n<N>_decoder_<k>_kept          indices of the blocks actually stored
  n<N>_decoder_<k>_block_<j>     Choi block of decoder k, ordered (R, S)
  n<N>_alice_pov_<k>_block_<j>   Choi block of N_k o E, same indexing

Only blocks that carry weight are stored.  Because the encoder is supported on lambda = (n) and
id^{otimes (n-k)} acts trivially on the surviving copies, the channel's selection rule forces

    mu_{n-k} = (n-k)

for every non-zero component; all other blocks have M_k^mu = 0, contribute nothing to the
fidelity, and are left by the optimizer at the trivial unital point I_R (x) I_m / d_R.  At
n = 17 that is 90 blocks out of 330.  verify_all_n.py reconstructs the omitted ones.

---------------------------------------------------------------------------
HOW TO VERIFY
---------------------------------------------------------------------------
    python verify_all_n.py

checks, for every n: that V is an exact isometry (V^dagger V = I); that each stored decoder
block is positive semidefinite and unital (Tr_R B = I); that the fidelity recomputed from the
stored operators,

    F = sum_k w_k (1/d_R^2) sum_mu f_mu Re Tr(M_k^mu D_k^mu),

matches the stored value (it does, to ~1e-15); and that F is non-decreasing in n.  The SYT
counts f_mu are obtained from binomials, so no representation-theory code is needed.

---------------------------------------------------------------------------
CONVENTIONS
---------------------------------------------------------------------------
* Choi convention: J^E = (id otimes E)(|Gamma><Gamma|), |Gamma> = sum_i |ii>, id on the
  reference R.
* All matrices are complex128.  Decoder and Alice-POV blocks are ordered (R, S).
* d_R = d_A = d_B = 2;  q = 1/2;  p = 1/(1+sqrt(2)).
===========================================================================
