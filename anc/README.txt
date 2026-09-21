===========================================================================
  ANCILLARY DATA
  Optimal isometric encoders and decoders for the symmetric seesaw
===========================================================================

Layout
------
  README.txt
  verify.py                       validity check for everything below, every n
  Flagged_X_Z_Independent/
      fidelities.txt              entanglement fidelity, indexed by n
      optimal_encoders/           encoder_n01.txt ... encoder_n20.txt
      optimal_decoders_n17/       decoders_n17.npz  (see "decoders" below)
  depolarizing/
      fidelities.txt
      optimal_encoders/
      fidelity_vs_p_n1-20.npz     fidelity against p for n = 1..20 (figure data)
  amplitude_damping/
      fidelities.txt
      optimal_encoders/
      fidelity_vs_gamma_n1-20.npz fidelity against gamma for n = 1..20 (figure data)

Dependencies for verify.py: Python >= 3.10, numpy, scipy.  Nothing else.

---------------------------------------------------------------------------
CHANNELS
---------------------------------------------------------------------------
Flagged_X_Z_Independent
    N(rho) = q |0><0|_Z (x) id(rho)_B + (1-q) |1><1|_Z (x) P_p(rho)_B,
    P_p the Pauli channel ((1-p)/2, p/2, p/2, (1-p)/2), i.e. the binary symmetric channel with
    crossover p = 1/(1+sqrt(2)) ~ 0.41421, and q = 1/2.  The classical flag Z is delivered to
    the receiver, so for n uses the flag string reveals the number k of "Pauli" events and the
    n-use channel splits into n+1 sectors N_k = P_p^{(x)k} (x) id^{(x)(n-k)}, each with its own
    decoder:  F = sum_k C(n,k) (1/2)^n F_D[k].  Superactivation is established when F > 3/4.

depolarizing          N_p(rho) = (1-p) rho + p I/2,  at p = 0.151020
amplitude_damping     A_gamma,                       at gamma = 0.510204

For the two unflagged channels the quantity of interest is max over n, so the fidelity need not
increase with n (and does not: the permutation-invariant ansatz gives a genuinely non-monotone
curve).  For the flagged channel it does increase with n, because the identity branch lets a
code on n uses embed into n+1.

---------------------------------------------------------------------------
ENCODERS
---------------------------------------------------------------------------
For every n the encoder is E(rho) = V rho V^dagger with V : C^2 -> Sym^n(C^2) an isometry.  A
permutation-invariant encoder is an isometry exactly when its Choi matrix has rank one, which
forces a single Young diagram with multiplicity f_lambda = 1; for n > d_A = 2 the antisymmetric
diagram is absent, leaving lambda = (n) uniquely.  So

    isometric + permutation invariant  <=>  encodes into the symmetric subspace,

and Tr_R of the Choi state is the rank-two flat state on a two-dimensional subspace of Sym^n --
the class of inputs used by Agarwal et al. [arXiv:2605.09138].

encoder_nNN.txt stores V as plain text: n+1 rows, one per Dicke basis vector |D^n_w>, and four
columns Re(V[w,0]) Im(V[w,0]) Re(V[w,1]) Im(V[w,1]).

---------------------------------------------------------------------------
DECODERS
---------------------------------------------------------------------------
Decoders are not distributed here, with one exception: n = 17 of the flagged channel, which is
the point that establishes superactivation and therefore ships so that the claim can be checked
end to end.  Everywhere else the decoder adds no information -- for a fixed encoder it is the
solution of the recovery SDP -- and the archives are large, so they are kept with the authors
and available on request.

decoders_n17.npz holds, per sector k:
  decoder_<k>_labels      (mu_k | mu_{n-k}) label of every block
  decoder_<k>_dims        dimension of every block (0 for empty blocks)
  decoder_<k>_kept        indices of the blocks actually stored
  decoder_<k>_block_<j>   Choi block of the decoder, ordered (R, S)
  alice_pov_<k>_block_<j> Choi block of N_k o E, same indexing

Only blocks carrying weight are stored.  Because the encoder is supported on lambda = (n) and
id^{(x)(n-k)} acts trivially on the surviving copies, the channel's selection rule forces
mu_{n-k} = (n-k); all other blocks have M_k^mu = 0, contribute nothing to the fidelity, and are
left by the optimizer at the trivial unital point I_R (x) I_m / d_R.  At n = 17 that is 90
blocks stored out of 330; verify.py reconstructs the remaining 240 and checks them too.

---------------------------------------------------------------------------
VERIFICATION
---------------------------------------------------------------------------
    python verify.py

checks, for every channel and every n, that each stored block is a proper quantum map:

  encoders   V^dagger V = I.  The Choi matrix |v><v|, v = sum_i |i> (x) V|i>, is rank one and
             positive, hence completely positive; Tr_{A^n}|v><v| = (V^dagger V)^T, so this one
             identity certifies trace preservation as well.  The shape (n+1 amplitudes) is
             checked too.
  decoders   every stored block is positive semidefinite and satisfies Tr_R B = I_m; the
             omitted blocks are reconstructed as I_R (x) I_m / d_R and checked.
  fidelity   for n = 17, recomputed from the stored operators and compared with fidelities.txt;
             the SYT counts come from binomials, so no representation theory is needed.

Expected output: ALL CHECKS PASSED, with the n = 17 fidelity agreeing to ~1e-15 and every
encoder isometric to ~1e-15.

---------------------------------------------------------------------------
CONVENTIONS
---------------------------------------------------------------------------
* Choi convention J^E = (id (x) E)(|Gamma><Gamma|), |Gamma> = sum_i |ii>, id acting on R.
* Decoder and Alice-POV blocks are ordered (R, S) and are complex128.
* d_R = d_A = d_B = 2.
===========================================================================
