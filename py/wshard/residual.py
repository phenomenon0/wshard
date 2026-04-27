"""
Sign2ndDiff residual encoding for W-SHARD.

The 1-bit curvature-sign residual is highly decorrelating under local linear prediction:
    residual[i] = sign(2*x[i] - x[i-1] - x[i+1])

This captures whether each point is:
- +1: Above the line between neighbors (convex/peak)
- -1: Below the line between neighbors (concave/dip)
- 0: On the line (linear)

This module is fully vectorized with NumPy for high performance.
"""

from typing import List, Optional, Tuple
import numpy as np

# Optional cowrie integration for BITMASK encoding
try:
    from cowrie.gen2 import Value, BitmaskData

    HAS_COWRIE = True
except ImportError:
    HAS_COWRIE = False


def sign2nd_diff(prev: float, curr: float, next_val: float) -> int:
    """
    Compute the sign of the second difference (scalar version).

    Returns:
        +1 if curr is above the line between prev and next (convex/peak)
        -1 if curr is below the line (concave/dip)
        0 if curr is on the line (linear)
    """
    diff = 2.0 * curr - prev - next_val
    if diff > 0:
        return 1
    elif diff < 0:
        return -1
    return 0


def compute_sign2nd_diff(signal: np.ndarray, edge_rule: str = "pad") -> np.ndarray:
    """
    Compute sign2nd_diff residuals for a 1D signal (vectorized).

    Args:
        signal: 1D float array of shape [T]
        edge_rule: "pad" (set edges to 0) or "overlap" (deprecated)

    Returns:
        int8 array of shape [T] with values in {-1, 0, +1}
    """
    T = len(signal)
    if T < 3:
        return np.zeros(T, dtype=np.int8)

    # Vectorized computation: diff = 2*x[i] - x[i-1] - x[i+1]
    # For interior points [1:T-1]
    diff = 2.0 * signal[1:-1] - signal[:-2] - signal[2:]

    # Vectorized sign: +1 if > 0, -1 if < 0, 0 if == 0
    interior = np.sign(diff).astype(np.int8)

    # Construct full array with padded edges
    residuals = np.zeros(T, dtype=np.int8)
    residuals[1:-1] = interior

    # Edge rule: pad mode sets edges to 0 (already initialized to 0)
    # Edge rule: overlap mode would require overlap data

    return residuals


def compute_sign2nd_diff_multidim(
    signal: np.ndarray, edge_rule: str = "pad"
) -> np.ndarray:
    """
    Compute sign2nd_diff residuals for a multi-dimensional signal (vectorized).

    Args:
        signal: 2D float array of shape [T, D]
        edge_rule: "pad" (set edges to 0)

    Returns:
        int8 array of shape [T, D] with values in {-1, 0, +1}
    """
    T, D = signal.shape
    if T < 3:
        return np.zeros((T, D), dtype=np.int8)

    # Vectorized computation for all dimensions at once
    # diff[i, d] = 2*signal[i, d] - signal[i-1, d] - signal[i+1, d]
    diff = 2.0 * signal[1:-1, :] - signal[:-2, :] - signal[2:, :]

    # Vectorized sign
    interior = np.sign(diff).astype(np.int8)

    # Construct full array with padded edges
    residuals = np.zeros((T, D), dtype=np.int8)
    residuals[1:-1, :] = interior

    return residuals


def compute_error_residual(
    signal: np.ndarray, omen: np.ndarray, edge_rule: str = "pad"
) -> np.ndarray:
    """
    Compute sign2nd_diff residual on the prediction error (signal - omen).

    Args:
        signal: Ground truth [T] or [T, D]
        omen: Prediction [T] or [T, D]
        edge_rule: "pad" or "overlap"

    Returns:
        int8 array with residuals
    """
    error = signal - omen
    if error.ndim == 1:
        return compute_sign2nd_diff(error, edge_rule)
    return compute_sign2nd_diff_multidim(error, edge_rule)


def pack_residual_bits(residuals: np.ndarray) -> bytes:
    """
    Pack residuals into bits (LSB-first, vectorized).

    +1 maps to 1, all else maps to 0.

    Args:
        residuals: 1D int8 array of shape [T]

    Returns:
        Packed bytes, ceil(T/8) bytes
    """
    T = len(residuals)
    num_bytes = (T + 7) // 8

    # Convert to bits: +1 -> 1, else -> 0
    bits = (residuals > 0).astype(np.uint8)

    # Pad to multiple of 8
    if T % 8 != 0:
        bits = np.pad(bits, (0, 8 - T % 8), mode="constant", constant_values=0)

    # Reshape to [num_bytes, 8] and pack using bit shifts
    bits = bits.reshape(-1, 8)

    # Pack bits LSB-first: bit[i] contributes (1 << i)
    powers = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    packed = np.sum(bits * powers, axis=1, dtype=np.uint8)

    return bytes(packed)


def unpack_residual_bits(packed: bytes, T: int) -> np.ndarray:
    """
    Unpack residual bits to int8 array (vectorized).

    Bit 1 → +1, Bit 0 → -1

    Args:
        packed: Packed bytes
        T: Number of timesteps

    Returns:
        int8 array of shape [T] with values in {-1, +1}
    """
    # Convert bytes to numpy array
    packed_arr = np.frombuffer(packed, dtype=np.uint8)

    # Expand each byte to 8 bits using bit masks
    # Shape: [num_bytes, 8]
    bit_masks = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    bits = ((packed_arr[:, np.newaxis] & bit_masks) > 0).astype(np.int8)

    # Flatten and truncate to T
    bits = bits.ravel()[:T]

    # Map: 1 -> +1, 0 -> -1
    residuals = np.where(bits == 1, np.int8(1), np.int8(-1))

    return residuals


def pack_multidim_residual_bits(residuals: np.ndarray) -> bytes:
    """
    Pack multi-dimensional residuals (concatenated per-dimension, vectorized).

    Args:
        residuals: 2D int8 array of shape [T, D]

    Returns:
        Packed bytes, D * ceil(T/8) bytes
    """
    T, D = residuals.shape
    num_bytes_per_dim = (T + 7) // 8

    # Convert to bits: +1 -> 1, else -> 0
    bits = (residuals > 0).astype(np.uint8)

    # Pad T to multiple of 8
    if T % 8 != 0:
        pad_size = 8 - T % 8
        bits = np.pad(bits, ((0, pad_size), (0, 0)), mode="constant", constant_values=0)

    # Reshape to [num_bytes_per_dim, 8, D]
    bits = bits.reshape(-1, 8, D)

    # Pack bits LSB-first for each dimension
    powers = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8).reshape(1, 8, 1)
    packed = np.sum(bits * powers, axis=1, dtype=np.uint8)  # [num_bytes_per_dim, D]

    # Transpose to [D, num_bytes_per_dim] then flatten (dims concatenated)
    packed = packed.T.ravel()

    return bytes(packed)


def unpack_multidim_residual_bits(packed: bytes, T: int, D: int) -> np.ndarray:
    """
    Unpack multi-dimensional residual bits (vectorized).

    Args:
        packed: Packed bytes
        T: Number of timesteps
        D: Number of dimensions

    Returns:
        int8 array of shape [T, D]
    """
    bytes_per_dim = (T + 7) // 8

    # Convert bytes to numpy array and reshape to [D, bytes_per_dim]
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    packed_arr = packed_arr.reshape(D, bytes_per_dim)

    # Expand each byte to 8 bits
    bit_masks = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    # Shape: [D, bytes_per_dim, 8]
    bits = ((packed_arr[:, :, np.newaxis] & bit_masks) > 0).astype(np.int8)

    # Reshape to [D, bytes_per_dim * 8] then truncate and transpose
    bits = bits.reshape(D, -1)[:, :T].T  # [T, D]

    # Map: 1 -> +1, 0 -> -1
    residuals = np.where(bits == 1, np.int8(1), np.int8(-1))

    return residuals


def quantize_delta(
    signal: np.ndarray, omen: np.ndarray, window_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize signal-omen delta to int8 with per-window scales (vectorized).

    Args:
        signal: Ground truth [T]
        omen: Prediction [T]
        window_size: Window size for scale computation

    Returns:
        Tuple of (quantized int8 deltas, float32 scales)
    """
    delta = signal - omen
    T = len(delta)

    num_windows = (T + window_size - 1) // window_size

    # Pad delta to multiple of window_size
    pad_size = num_windows * window_size - T
    if pad_size > 0:
        delta_padded = np.pad(delta, (0, pad_size), mode="constant", constant_values=0)
    else:
        delta_padded = delta

    # Reshape to [num_windows, window_size]
    delta_windows = delta_padded.reshape(num_windows, window_size)

    # Compute max absolute value per window
    max_abs = np.max(np.abs(delta_windows), axis=1)

    # Compute scales (avoid division by zero)
    scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)

    # Quantize: broadcast scales to match window shape
    scales_expanded = scales[:, np.newaxis]  # [num_windows, 1]
    quantized_windows = np.clip(
        np.round(delta_windows / scales_expanded), -127, 127
    ).astype(np.int8)

    # Flatten and truncate to original size
    quantized = quantized_windows.ravel()[:T]

    return quantized, scales


def dequantize_delta(
    quantized: np.ndarray, scales: np.ndarray, window_size: int = 256
) -> np.ndarray:
    """
    Dequantize int8 deltas using per-window scales (vectorized).

    Args:
        quantized: int8 array [T]
        scales: float32 scales per window
        window_size: Window size

    Returns:
        float32 delta array [T]
    """
    T = len(quantized)
    num_windows = len(scales)

    # Pad quantized to multiple of window_size
    pad_size = num_windows * window_size - T
    if pad_size > 0:
        quantized_padded = np.pad(
            quantized, (0, pad_size), mode="constant", constant_values=0
        )
    else:
        quantized_padded = quantized

    # Reshape to [num_windows, window_size]
    quantized_windows = quantized_padded.reshape(num_windows, window_size).astype(
        np.float32
    )

    # Broadcast multiply by scales
    scales_expanded = scales[:, np.newaxis]  # [num_windows, 1]
    delta_windows = quantized_windows * scales_expanded

    # Flatten and truncate
    delta = delta_windows.ravel()[:T]

    return delta


def reconstruct_from_omen_and_delta(
    omen: np.ndarray, quantized: np.ndarray, scales: np.ndarray, window_size: int = 256
) -> np.ndarray:
    """
    Reconstruct signal from omen + quantized delta.

    Args:
        omen: Prediction [T]
        quantized: int8 deltas [T]
        scales: Per-window scales
        window_size: Window size

    Returns:
        Reconstructed signal [T]
    """
    delta = dequantize_delta(quantized, scales, window_size)
    return omen + delta


def pack_bool_mask(mask: np.ndarray) -> bytes:
    """Pack boolean mask to bits (LSB-first)."""
    return pack_residual_bits(mask.astype(np.int8))


def unpack_bool_mask(packed: bytes, T: int) -> np.ndarray:
    """Unpack boolean mask from bits."""
    residuals = unpack_residual_bits(packed, T)
    return residuals > 0


# ============================================================
# Cowrie BITMASK integration (Gap 3)
# ============================================================

def pack_residual_bitmask(residuals: np.ndarray) -> bytes:
    """
    Pack residuals using cowrie BITMASK API.

    Uses cowrie's BitmaskData for validation and bit packing if available,
    but stores only the raw packed bits (identical wire format to
    pack_residual_bits). This ensures backward compatibility — old readers
    can still decode the block as raw LSB-first bits.

    Args:
        residuals: 1D int8 array of shape [T] with values in {-1, 0, +1}

    Returns:
        Raw packed bytes, ceil(T/8) bytes (identical format to pack_residual_bits)
    """
    if HAS_COWRIE:
        # Use cowrie's BitmaskData for validation, extract raw bits
        bools = [bool(r > 0) for r in residuals]
        bm = Value.bitmask_from_bools(bools)
        return bytes(bm.data.bits)

    # Fallback: raw LSB-first packing
    return pack_residual_bits(residuals)


def unpack_residual_bitmask(data: bytes, T: int) -> np.ndarray:
    """
    Unpack residuals using cowrie BITMASK API.

    Interprets raw packed bits via cowrie's BitmaskData if available,
    otherwise falls back to raw bit unpacking. The on-disk format
    is the same either way (LSB-first packed bytes).

    Args:
        data: Raw packed bytes
        T: Number of timesteps

    Returns:
        int8 array of shape [T] with values in {-1, +1}
    """
    if HAS_COWRIE and len(data) > 0:
        bm = BitmaskData(count=T, bits=data)
        bools = bm.to_bools()
        return np.array([1 if b else -1 for b in bools], dtype=np.int8)

    # Fallback: raw LSB-first unpacking
    return unpack_residual_bits(data, T)


def pack_multidim_residual_bitmask(residuals: np.ndarray) -> bytes:
    """
    Pack multi-dimensional residuals using cowrie BITMASK per dimension.

    Args:
        residuals: 2D int8 array of shape [T, D]

    Returns:
        Concatenated cowrie BITMASK bytes with length-prefixed segments
    """
    import struct

    T, D = residuals.shape
    segments = []
    for d in range(D):
        segment = pack_residual_bitmask(residuals[:, d])
        # Length-prefix each segment for parsing
        segments.append(struct.pack("<I", len(segment)))
        segments.append(segment)
    return b"".join(segments)


def unpack_multidim_residual_bitmask(data: bytes, T: int, D: int) -> np.ndarray:
    """
    Unpack multi-dimensional residuals from cowrie BITMASK segments.

    Args:
        data: Concatenated length-prefixed BITMASK segments
        T: Number of timesteps
        D: Number of dimensions

    Returns:
        int8 array of shape [T, D]
    """
    import struct

    result = np.zeros((T, D), dtype=np.int8)
    offset = 0
    for d in range(D):
        seg_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        segment = data[offset:offset + seg_len]
        offset += seg_len
        result[:, d] = unpack_residual_bitmask(segment, T)
    return result
