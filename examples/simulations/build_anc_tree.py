"""Lay out the ancillary directory as one anc/ tree with one subfolder per channel.

    anc/
      verify.py
      flagged_x_z_independent/   fidelities.txt, optimal_encoders/, optimal_decoders/
      depolarizing/              fidelities.txt, optimal_encoders/, optimal_decoders/,
                                 fidelity_vs_p_n1-20.npz
      amplitude_damping/         fidelities.txt, optimal_encoders/, optimal_decoders/,
                                 fidelity_vs_gamma_n1-20.npz

Every operator is stored the same way: one compressed NumPy archive per n, with the same key
names throughout, so that encoders and decoders are read back identically.  (Plain text was the
earlier choice for encoders, but the flagged decoders at n = 17 run to 67 MB as text against
9.7 MB compressed, so a single uniform binary format is the only consistent option.)
For the two single-channel examples the optimal decoders ship for every n; for the flagged
channel only n = 17 does, the blocklength at which superactivation is established.

    uv run python -m examples.simulations.build_anc_tree
    uv run python -m examples.simulations.build_anc_tree --decoders-only
"""
from __future__ import annotations

import glob
import os
import shutil

import numpy as np

SRC = {
    "flagged_x_z_independent": ("local_data/superactivation_operators_n1-20.npz", None),
    "depolarizing": ("local_data/depolarizing_operators_n1-20.npz",
                     ("anc_2/depolarizing_fidelity_n1-20.npz", "fidelity_vs_p_n1-20.npz")),
    "amplitude_damping": ("local_data/amplitude_damping_operators_n1-20.npz",
                          ("anc_2/amplitude_damping_fidelity_n1-20.npz",
                           "fidelity_vs_gamma_n1-20.npz")),
}
DESCRIPTION = {
    "flagged_x_z_independent":
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
        if channel == "flagged_x_z_independent":
            fh.write(f"# mixing q: {q!r}\n")
        fh.write(f"# reference dimension d_R: {d_R}\n")
        fh.write("# entanglement fidelity of the optimal isometric encoder/decoder pair,\n")
        fh.write("# indexed by the number of channel uses n\n")
        fh.write(f"#{'n':>4}  {'fidelity':>20}\n")
        for n, f in zip(ns, fids):
            fh.write(f"{n:5d}  {f:20.16f}\n")


def write_encoder(path, channel, n, param, d_R, fidelity, V):
    """Store one n's encoder: the isometry V : C^d_R -> Sym^n(A) in the Dicke basis.

    Same container and key conventions as the decoder archives, so both are read back the
    same way.
    """
    np.savez_compressed(
        path,
        encoder_isometry_dicke=np.asarray(V, dtype=np.complex128),
        n=np.int64(n),
        param=np.float64(param),
        d_R=np.int64(d_R),
        fidelity=np.float64(fidelity),
        channel=str(channel),
    )


def write_decoders(path, d, channel, n, param, q, d_R):
    """Store one n's decoder blocks, in the Schur-Weyl basis, with the blocks of M it decodes.

    ``alice_pov`` is kept alongside the decoder so that the fidelity can be recomputed from the
    archive alone: rebuilding it from the encoder would need the block-diagonalising
    isomorphism, which the standalone verifier deliberately does not depend on.
    """
    payload = {k[len(f"n{n}_"):]: d[k] for k in d.files if k.startswith(f"n{n}_")}
    payload["n"] = np.int64(n)
    payload["param"] = np.float64(param)
    payload["q"] = np.float64(q)
    payload["d_R"] = np.int64(d_R)
    payload["channel"] = str(channel)
    np.savez_compressed(path, **payload)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decoders-only", action="store_true",
                        help="write only the per-n decoder archives, leaving the rest in place")
    parser.add_argument("--encoders-only", action="store_true",
                        help="rewrite only the per-n encoder archives")
    args = parser.parse_args()

    if args.encoders_only:
        for channel, (src, _) in SRC.items():
            d = np.load(src, allow_pickle=True)
            enc_dir = os.path.join("anc", channel, "optimal_encoders")
            os.makedirs(enc_dir, exist_ok=True)
            param, d_R = float(d["param"]), int(d["d_R"])
            ns = [int(x) for x in d["n_values"]]
            for n in ns:
                write_encoder(os.path.join(enc_dir, f"encoder_n{n:02d}.npz"), channel, n,
                              param, d_R, float(d[f"n{n}_fidelity"]),
                              d[f"n{n}_encoder_isometry_dicke"])
            for stale in glob.glob(os.path.join(enc_dir, "*.txt")):
                os.remove(stale)
            print(f"  {enc_dir}: {len(ns)} encoder archives (removed the .txt versions)")
        return

    if args.decoders_only:
        for channel, (src, _) in SRC.items():
            if channel == "flagged_x_z_independent":
                continue
            d = np.load(src, allow_pickle=True)
            dec_dir = os.path.join("anc", channel, "optimal_decoders")
            os.makedirs(dec_dir, exist_ok=True)
            param = float(d["param"])
            q = float(d["q"]) if "q" in d.files else 0.5
            d_R = int(d["d_R"])
            total = 0
            for n in [int(x) for x in d["n_values"]]:
                out = os.path.join(dec_dir, f"decoders_n{n:02d}.npz")
                write_decoders(out, d, channel, n, param, q, d_R)
                total += os.path.getsize(out)
            print(f"  {dec_dir}: {len(d['n_values'])} files, {total / 1e6:.1f} MB")
        return

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
        for n, fid in zip(ns, fids):
            write_encoder(os.path.join(enc_dir, f"encoder_n{n:02d}.npz"),
                          channel, n, param, d_R, fid, d[f"n{n}_encoder_isometry_dicke"])

        if sweep is not None and os.path.exists(sweep[0]):
            shutil.copy(sweep[0], os.path.join(root, sweep[1]))

        if channel != "flagged_x_z_independent":
            dec_dir = os.path.join(root, "optimal_decoders")
            os.makedirs(dec_dir, exist_ok=True)
            for n in ns:
                write_decoders(os.path.join(dec_dir, f"decoders_n{n:02d}.npz"),
                               d, channel, n, param, q, d_R)

        if channel == "flagged_x_z_independent":
            dec_dir = os.path.join(root, "optimal_decoders")
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
