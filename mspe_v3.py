#!/usr/bin/env python3
"""
MSPE-V3 image encrypt/decrypt

Usage:
    python mspe_v3.py encrypt input.png output.png "your-secret-key"
    python mspe_v3.py decrypt encrypted.png restored.png "your-secret-key"

Dependencies:
    pip install pillow numpy scipy

Algorithm:
- Convert RGB -> YCbCr
- Process Y / Cb / Cr independently
- Full-frame 2D DCT
- Key-derived +/-1 spectral mask
- Preserve DC coefficient
- Inverse DCT
- Apply channel gain
- Convert back to RGB

Prototype gains:
    Y  = 0.42
    Cb = 0.28
    Cr = 0.28

Important:
This is experimental perceptual scrambling and is NOT a replacement for
standard cryptographic encryption such as AES-GCM.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn


GAINS = (0.42, 0.28, 0.28)
CHANNEL_LABELS = (b"Y", b"Cb", b"Cr")


def normalize_key(key: str) -> bytes:
    """Normalize an arbitrary UTF-8 key string to 32 bytes."""
    return hashlib.sha256(key.encode("utf-8")).digest()


def keyed_sign_mask(
    height: int,
    width: int,
    master_key: bytes,
    label: bytes,
) -> np.ndarray:
    """
    Deterministically generate a +/-1 spectral mask from:
      key + channel label + frequency coordinates.
    """
    ky = np.arange(height, dtype=np.uint64)[:, None]
    kx = np.arange(width, dtype=np.uint64)[None, :]

    seed = np.uint64(
        int.from_bytes(
            hashlib.sha256(master_key + b"|" + label).digest()[:8],
            "little",
        )
    )

    z = (
        kx * np.uint64(0x9E3779B185EBCA87)
        ^ ky * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ seed
    )

    # SplitMix64-style mixing.
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)

    mask = np.where(
        (z & np.uint64(1)) == 0,
        1.0,
        -1.0,
    ).astype(np.float32)

    # Preserve DC / average channel level.
    mask[0, 0] = 1.0
    return mask


def encrypt_image(
    image: Image.Image,
    master_key: bytes,
) -> Image.Image:
    ycc = np.asarray(
        image.convert("YCbCr"),
        dtype=np.float32,
    )

    height, width, _ = ycc.shape
    output = np.empty_like(ycc)

    for channel, (label, gain) in enumerate(
        zip(CHANNEL_LABELS, GAINS)
    ):
        centered = ycc[..., channel] - 128.0

        coeff = dctn(
            centered,
            type=2,
            norm="ortho",
        )

        mask = keyed_sign_mask(
            height,
            width,
            master_key,
            label,
        )

        scrambled = idctn(
            coeff * mask,
            type=2,
            norm="ortho",
        )

        encrypted_channel = 128.0 + gain * scrambled

        output[..., channel] = np.clip(
            np.rint(encrypted_channel),
            0,
            255,
        )

    encrypted_ycc = Image.fromarray(
        output.astype(np.uint8),
        mode="YCbCr",
    )

    return encrypted_ycc.convert("RGB")


def decrypt_image(
    image: Image.Image,
    master_key: bytes,
) -> Image.Image:
    ycc = np.asarray(
        image.convert("YCbCr"),
        dtype=np.float32,
    )

    height, width, _ = ycc.shape
    output = np.empty_like(ycc)

    for channel, (label, gain) in enumerate(
        zip(CHANNEL_LABELS, GAINS)
    ):
        normalized = (
            ycc[..., channel] - 128.0
        ) / gain

        coeff = dctn(
            normalized,
            type=2,
            norm="ortho",
        )

        mask = keyed_sign_mask(
            height,
            width,
            master_key,
            label,
        )

        recovered = idctn(
            coeff * mask,
            type=2,
            norm="ortho",
        )

        recovered_channel = recovered + 128.0

        output[..., channel] = np.clip(
            np.rint(recovered_channel),
            0,
            255,
        )

    restored_ycc = Image.fromarray(
        output.astype(np.uint8),
        mode="YCbCr",
    )

    return restored_ycc.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MSPE-V3 image perceptual encryption/decryption"
    )

    parser.add_argument(
        "mode",
        choices=("encrypt", "decrypt"),
        help="Operation mode",
    )

    parser.add_argument(
        "input",
        help="Input image path",
    )

    parser.add_argument(
        "output",
        help="Output image path",
    )

    parser.add_argument(
        "key",
        help="Secret key string",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(
            f"Input file does not exist: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_key = normalize_key(args.key)

    with Image.open(input_path) as image:
        image = image.convert("RGB")

        if args.mode == "encrypt":
            result = encrypt_image(
                image,
                master_key,
            )
        else:
            result = decrypt_image(
                image,
                master_key,
            )

        # PNG is strongly recommended to avoid an extra lossy image encode.
        result.save(output_path)

    print(
        f"{args.mode} complete: {output_path}"
    )


if __name__ == "__main__":
    main()
