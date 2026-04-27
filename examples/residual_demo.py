"""Sign2nd-diff residual encoding demo.

Takes a smooth 7-joint sine-wave signal (500 steps + small noise), computes
the sign-of-second-difference residual, packs it to bits, unpacks, and checks
the round-trip.

Note on round-trip semantics: pack_residual_bits maps {+1 -> 1, 0/-1 -> 0}.
unpack_residual_bits maps {1 -> +1, 0 -> -1}. Zeros in the residual therefore
become -1 after the round-trip. The round-trip is exact for +1/-1 values; zeros
become -1 (1-bit encoding has no zero symbol). This is expected and noted below.

Run:
    python examples/residual_demo.py
"""

from __future__ import annotations

import math

import numpy as np

from wshard import (
    compute_sign2nd_diff_multidim,
    pack_multidim_residual_bits,
    unpack_multidim_residual_bits,
)


def main() -> None:
    rng = np.random.default_rng(7)
    T, D = 500, 7

    # Smooth sine-wave signal on 7 joints with small noise
    t = np.linspace(0, 4 * math.pi, T)
    freqs = np.linspace(0.5, 2.0, D)
    signal = np.sin(np.outer(t, freqs)).astype(np.float32)
    signal += rng.standard_normal((T, D)).astype(np.float32) * 0.05

    # Compute sign2nd-diff residual: values in {-1, 0, +1}
    residual = compute_sign2nd_diff_multidim(signal)  # [T, D] int8

    # Pack to bits: +1 -> 1, 0/-1 -> 0  (1 bit per timestep per dimension)
    packed = pack_multidim_residual_bits(residual)

    # Unpack: 1 -> +1, 0 -> -1  (zeros become -1; this is the expected lossy map)
    unpacked = unpack_multidim_residual_bits(packed, T, D)

    # Verify: where residual != 0, the sign is preserved exactly
    nonzero_mask = residual != 0
    signs_preserved = np.all(unpacked[nonzero_mask] == residual[nonzero_mask])

    # Original raw signal size vs packed residual size
    orig_bytes = T * D * 4       # float32
    packed_bytes = len(packed)   # ceil(T/8) bytes per dimension
    ratio = orig_bytes / packed_bytes

    print("Residual encoding demo")
    print(f"  signal shape   : {T} steps x {D} joints  (float32 sine + noise)")
    print(f"  original size  : {orig_bytes:,} bytes  ({orig_bytes / 1024:.1f} KiB)")
    print(f"  packed size    : {packed_bytes:,} bytes  ({packed_bytes / 1024:.1f} KiB)")
    print(f"  ratio          : {ratio:.1f}x")
    print(f"  non-zero steps : {nonzero_mask.sum()} / {T * D}")
    print(f"  signs preserved: {signs_preserved}  (non-zero residuals are bit-identical)")
    print(f"  zero->-1 map   : expected (1-bit encoding has no zero symbol)")


if __name__ == "__main__":
    main()
