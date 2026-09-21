"""Lay out the ancillary directory as one anc/ tree with one subfolder per channel.

    anc/
      README.txt
      verify.py
      Flagged_X_Z_Independent/   fidelities.txt, optimal_encoders/, optimal_decoders_n17/
      depolarizing/              fidelities.txt, optimal_encoders/, fidelity_vs_p_n1-20.npz
      amplitude_damping/         fidelities.txt, optimal_encoders/, fidelity_vs_gamma_n1-20.npz

Encoders are written as plain text, one file per n, so the archive is readable without any
particular library.  Decoders are kept on the local machine (local_data/, git-ignored) except
for n = 17 of the flagged channel, which ships here so that the superactivation claim can be
checked end to end.

    uv run python examples/simulations/build_anc_tree.py
"""
from __future__ import annotations

import os
import shutil

import numpy as np

SRC = {
    "Flagged_X_Z_Independent": ("local_data/superactivation_operators_n1-20.npz", None),
    "depolarizing": ("local_data/depolarizing_operators_n1-20.npz",
                     ("anc_2/depolarizing_fidelity_n1-20.npz", "fidelity_vs_p_n1-20.npz")),
    "amplitude_damping": ("local_data/amplitude_damping_operators_n1-20.npz",
                          ("anc_2/amplitude_damping_fidelity_n1-20.npz",
                           "fidelity_vs_gamma_n1-20.npz")),
}
DESCRIPTION = {
    "Flagged_X_Z_Independent":
        "flagged mixture  N(rho) = q |0><0|_Z (x) id(rho) + (1-q) |1><1|_Z (x) P_p(rho),\n"
        "#   P_p the Pauli channel ((1-p)/2, p/2, p/2, (1-p)/2) = binary symmetric channel",
    "depolarizing": "N_p(rho) = (1-p) rho + p I/2",
    "amplitude_damping": "A_gamma, Kraus [[1,0],[0,sqrt(1-gamma)]] and [[0,sqrt(gamma)],[0,0]]",
}


def write_fidelities(path, channel, param, q, d_R, ns, fids):
    with open(path, "w") as fh:
        fh.write(f"# channel: {channel}\n")
        fh.write(f"#   {DESCRIPTION[channel]}\n")
        fh.write(f"# parameter: {param!r}\n")
        if channel == "Flagged_X_Z_Independent":
            fh.write(f"# mixing q: {q!r}\n")
        fh.write(f"# reference dimension d_R: {d_R}\n")
        fh.write("# entanglement fidelity of the optimal isometric encoder/decoder pair,\n")
        fh.write("# indexed by the number of channel uses n\n")
        fh.write(f"#{'n':>4}  {'fidelity':>20}\n")
        for n, f in zip(ns, fids):
            fh.write(f"{n:5d}  {f:20.16f}\n")


def write_encoder(path, channel, n, param, V):
    with open(path, "w") as fh:
        fh.write(f"# optimal encoder for {channel}, n = {n}, parameter {param!r}\n")
        fh.write("# isometry V : C^d_R -> Sym^n(C^2), so that E(rho) = V rho V^dagger.\n")
        fh.write(f"# {V.shape[0]} rows = Dicke basis |D^{n}_w>, w = 0 .. {n};"
                 f"  {V.shape[1]} columns = V|0>, V|1>.\n")
        fh.write("# columns below: Re(V[w,0]) Im(V[w,0]) Re(V[w,1]) Im(V[w,1])\n")
        fh.write("# V^dagger V = I to machine precision; check with verify.py\n")
        for w in range(V.shape[0]):
            fh.write("  ".join(f"{V[w, c].real:+.17e}  {V[w, c].imag:+.17e}"
                               for c in range(V.shape[1])) + "\n")


def main():
    os.makedirs("local_data", exist_ok=True)
    # move the bulky operator archives out of anc/ and onto the local machine
    for src, dst in [("anc/superactivation_operators_n1-20.npz",
                      "local_data/superactivation_operators_n1-20.npz"),
                     ("anc_2/depolarizing_operators_n1-20.npz",
                      "local_data/depolarizing_operators_n1-20.npz"),
                     ("anc_2/amplitude_damping_operators_n1-20.npz",
                      "local_data/amplitude_damping_operators_n1-20.npz")]:
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  moved {src} -> {dst}")

    for channel, (src, sweep) in SRC.items():
        d = np.load(src, allow_pickle=True)
        root = os.path.join("anc", channel)
        enc_dir = os.path.join(root, "optimal_encoders")
        os.makedirs(enc_dir, exist_ok=True)
        ns = [int(x) for x in d["n_values"]]
        fids = [float(d[f"n{n}_fidelity"]) for n in ns]
        param = float(d["param"])
        q = float(d["q"]) if "q" in d.files else 0.5
        d_R = int(d["d_R"])

        write_fidelities(os.path.join(root, "fidelities.txt"), channel, param, q, d_R, ns, fids)
        for n in ns:
            write_encoder(os.path.join(enc_dir, f"encoder_n{n:02d}.txt"),
                          channel, n, param, d[f"n{n}_encoder_isometry_dicke"])

        if sweep is not None and os.path.exists(sweep[0]):
            shutil.copy(sweep[0], os.path.join(root, sweep[1]))

        if channel == "Flagged_X_Z_Independent":
            dec_dir = os.path.join(root, "optimal_decoders_n17")
            os.makedirs(dec_dir, exist_ok=True)
            payload = {k[len("n17_"):]: d[k] for k in d.files if k.startswith("n17_")}
            payload["n"] = np.int64(17)
            payload["param"] = np.float64(param)
            payload["q"] = np.float64(q)
            payload["d_R"] = np.int64(d_R)
            payload["fidelity"] = np.float64(d["n17_fidelity"])
            out = os.path.join(dec_dir, "decoders_n17.npz")
            np.savez_compressed(out, **payload)
            print(f"  {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
        print(f"  {root}: {len(ns)} encoders, fidelities.txt")


if __name__ == "__main__":
    main()
