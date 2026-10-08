#!/usr/bin/env python3
"""
MSPE-V7 Video
=============

Goal
----
A stronger version of MSPE-V5 for a ciphertext-only attacker who:
- knows the complete algorithm,
- does not know the secret key,
- has many ciphertext frames,
- can train optimization / machine-learning reconstruction attacks.

Important security statement
----------------------------
MSPE-V7 is still an experimental *perceptual* encryption transform.
It is NOT claimed to provide IND-CPA / AEAD-level cryptographic secrecy.

V7 is the lower-noise refinement of V6. V6 mixed a 128x128 low/mid-frequency
core and changed it every 100 ms. That strongly reduced fixed leakage, but it
also moved too much signal energy across frequency coordinates and reduced
inter-frame predictability, so H.264/H.265 quantization noise became more
visible after inverse gain.

V7 keeps the same time-varying keyed orthogonal construction, but uses a
smaller 32x32 semantic core, a longer 250 ms epoch, and slightly larger
headroom gains. The goal is to preserve most of V6's temporal diversity while
reducing codec-induced noise and improving direct 480p recovery.

Pipeline
--------
FFmpeg decode -> raw YUV420p
-> per-plane 2D orthonormal DCT
-> static whole-plane keyed sign scrambling
-> time-varying low-frequency orthogonal mixing
-> inverse 2D DCT
-> headroom gain
-> FFmpeg encode

Audio is deliberately ignored.

Dependencies
------------
    pip install numpy scipy
    ffmpeg / ffprobe available in PATH

Examples
--------
Encrypt:
    python mspe_v7_video.py encrypt input.mp4 encrypted.mp4 "secret-key"

Decrypt:
    python mspe_v7_video.py decrypt encrypted.mp4 restored.mp4 "secret-key"

The decrypt side must use the same --epoch-ms / --mix-core / gain parameters
as the encrypt side.

Timing note
-----------
The current prototype derives epochs from frame timestamp ~= frame_index / fps.
This tolerates ordinary spatial resizing/transcoding when frame timing is kept,
but it is NOT yet a complete synchronization layer for arbitrary frame
insertion/deletion/VFR edits. A production design should carry an authenticated
public epoch identifier using a robust watermark or trusted stream metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct, dctn, idctn


# ===========================================================================
# EXPLICIT CONSTANTS AND THEIR ORIGIN
# ===========================================================================

# 128 = 2^7, the exact midpoint used to center unsigned 8-bit samples.
# This value comes from the sample representation itself.
CENTER_8BIT = np.float32(128.0)

# Headroom gains are engineering parameters, NOT cryptographic constants.
#
# Origin of 0.50 / 0.32:
# - We swept luma gains {0.42, 0.46, 0.50, 0.54} and chroma gains
#   {0.28, 0.32, 0.36} over sampled frames from the fixed benchmark set.
# - In the sampled final-V7 sweep, measured pre-encode luma clipping was:
#       Y=0.42: 0.00034%
#       Y=0.46: 0.00373%
#       Y=0.50: 0.01737%
#       Y=0.54: 0.06179%
#   We chose 0.50 as the noise/clipping compromise rather than pushing to 0.54.
# - C=0.28 / 0.32 / 0.36 all measured zero pre-encode clipping in that
#   sampled sweep. 0.32 keeps additional codec/gamut margin; 0.36 is available
#   as a CLI tuning option when chroma fidelity matters more.
# - Larger gains reduce decryption noise amplification (roughly 1/gain),
#   which is why V7 raises them from V6's 0.42 / 0.28.
#
# Both values remain explicit CLI parameters.
DEFAULT_GAIN_Y = 0.50
DEFAULT_GAIN_C = 0.32

# 32 = 2^5 harmonics per axis.
#
# This is an engineering parameter, NOT a security constant.
#
# Origin of 32:
# - V6 used 128x128 and achieved strong temporal scrambling, but the fixed
#   color-animation benchmark fell to about 37.8 dB at 1080p / 33.2 dB at
#   direct 480p recovery.
# - A controlled sweep with the same V6 construction and gains 0.50/0.32 gave:
#       core 32, epoch 250 ms: 43.13 dB / 39.52 dB
#       core 32, epoch 500 ms: 43.04 dB / 39.49 dB
#       core 64, epoch 250 ms: 41.17 dB / 36.79 dB
#       core 64, epoch 500 ms: 41.19 dB / 36.88 dB
# - Therefore 32 was chosen as the better quality/security tradeoff among
#   those tested points. It still covers coarse scene structure while moving
#   much less low/mid-frequency energy than V6's 128x128 core.
#
# It is exposed as --mix-core and can be changed.
DEFAULT_MIX_CORE = 32

# 250 milliseconds = 4 transform epochs per second.
#
# This is an engineering tradeoff, NOT a cryptographic constant.
#
# Origin of 250 ms:
# - V6's 100 ms switching increased temporal discontinuity and codec noise.
# - The controlled 32-core benchmark showed 250 ms and 500 ms had almost the
#   same PSNR, so 250 ms was selected because it gives twice as many independent
#   temporal transforms while retaining the lower-noise behavior.
# - On a static digit-1 ciphertext test, different 250 ms epochs had only about
#   0.084 pixel correlation; averaging all 12 ciphertext frames had about
#   0.0046 correlation with the plaintext and only about 11.2 dB PSNR. These are
#   diagnostics, not a proof against ML attacks.
#
# It is exposed as --epoch-ms.
DEFAULT_EPOCH_MS = 250

LABEL_Y = b"Y"
LABEL_CB = b"Cb"
LABEL_CR = b"Cr"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    rotation: int


def normalize_key(key: str) -> bytes:
    """
    Convert an arbitrary UTF-8 key/passphrase to 32 bytes with SHA-256.

    SHA-256 is standardized; no custom internal constants are introduced here.
    """
    return hashlib.sha256(key.encode("utf-8")).digest()


def parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 30.0
    return float(Fraction(value))


def probe_video(path: str) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:stream_side_data=rotation",
        "-of", "json",
        path,
    ]
    try:
        data = json.loads(subprocess.check_output(cmd))
    except FileNotFoundError:
        raise SystemExit("ffprobe not found; install FFmpeg first.")

    streams = data.get("streams", [])
    if not streams:
        raise SystemExit("No video stream found.")

    s = streams[0]
    coded_w = int(s["width"])
    coded_h = int(s["height"])

    rotation = 0
    for side in s.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(round(float(side["rotation"]))) % 360
            break

    # ffmpeg autorotates by default.
    if rotation in (90, 270):
        width, height = coded_h, coded_w
    else:
        width, height = coded_w, coded_h

    if width % 2 or height % 2:
        raise SystemExit(
            f"YUV420p requires even display dimensions; got {width}x{height}."
        )

    return VideoInfo(
        width=width,
        height=height,
        fps=parse_rate(s.get("avg_frame_rate", "30/1")),
        rotation=rotation,
    )


# ---------------------------------------------------------------------------
# Resolution-compatible static whole-plane sign mask
# ---------------------------------------------------------------------------

def make_static_sign_mask(
    height: int,
    width: int,
    master_key: bytes,
    label: bytes,
) -> np.ndarray:
    """
    Produce +/-1 for each DCT harmonic coordinate with SHAKE-256.

    Each frequency row ky uses:
        SHAKE256(key || fixed-domain-label || channel || ky)

    SHAKE is an extendable-output function. The first W bytes are identical
    whether W=427 or W=960, so the same low-frequency coordinate receives the
    same sign after downscaling to a smaller resolution.

    There are no private mixer constants.
    """
    out = np.empty((height, width), dtype=np.int8)
    prefix = master_key + b"|MSPE-V7|STATIC-SIGN|" + label + b"|"

    for ky in range(height):
        seed = prefix + ky.to_bytes(4, "little", signed=False)
        raw = hashlib.shake_256(seed).digest(width)
        row = np.frombuffer(raw, dtype=np.uint8)
        out[ky] = np.where((row & 1) == 0, 1, -1)

    out[0, 0] = 1
    return out


# ---------------------------------------------------------------------------
# Time-varying core transform
# ---------------------------------------------------------------------------

def _derive_affine_permutation(
    n: int,
    master_key: bytes,
    label: bytes,
    epoch: int,
    stage: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive a permutation p(i) = (a*i + b) mod n.

    a and b come from SHAKE-256.
    a is incremented until gcd(a,n)=1, which guarantees invertibility.

    No unexplained numeric constants are used.
    """
    if n <= 1:
        empty = np.arange(n, dtype=np.int64)
        return empty, empty

    seed = (
        master_key
        + b"|MSPE-V7|AFFINE|"
        + label
        + b"|"
        + epoch.to_bytes(8, "little", signed=False)
        + b"|"
        + stage
    )
    raw = hashlib.shake_256(seed).digest(16)
    x = int.from_bytes(raw[:8], "little")
    y = int.from_bytes(raw[8:], "little")

    a = x % n
    if a == 0:
        a = 1
    while math.gcd(a, n) != 1:
        a += 1
        if a >= n:
            a = 1

    b = y % n

    idx = np.arange(n, dtype=np.int64)
    perm = (a * idx + b) % n

    # If encrypted_vector = original_vector[perm], then:
    # original_vector = encrypted_vector[inverse]
    inverse = np.empty(n, dtype=np.int64)
    inverse[perm] = idx
    return perm, inverse


def _derive_epoch_signs(
    n: int,
    master_key: bytes,
    label: bytes,
    epoch: int,
) -> np.ndarray:
    """
    Time-varying +/-1 vector generated directly by SHAKE-256.
    """
    seed = (
        master_key
        + b"|MSPE-V7|EPOCH-SIGN|"
        + label
        + b"|"
        + epoch.to_bytes(8, "little", signed=False)
    )
    raw = hashlib.shake_256(seed).digest(n)
    bits = np.frombuffer(raw, dtype=np.uint8)
    return np.where((bits & 1) == 0, 1.0, -1.0).astype(np.float32)


@dataclass
class EpochTransform:
    perm1: np.ndarray
    invperm1: np.ndarray
    perm2: np.ndarray
    invperm2: np.ndarray
    signs: np.ndarray


class MSPEV7:
    def __init__(
        self,
        width: int,
        height: int,
        master_key: bytes,
        fps: float,
        workers: int = -1,
        gain_y: float = DEFAULT_GAIN_Y,
        gain_c: float = DEFAULT_GAIN_C,
        mix_core: int = DEFAULT_MIX_CORE,
        epoch_ms: int = DEFAULT_EPOCH_MS,
    ) -> None:
        self.width = width
        self.height = height
        self.cw = width // 2
        self.ch = height // 2

        self.master_key = master_key
        self.fps = fps
        self.workers = workers
        self.epoch_ms = epoch_ms

        self.gain_y = np.float32(gain_y)
        self.gain_c = np.float32(gain_c)

        self.core_y = max(2, min(mix_core, height, width))
        self.core_c = max(2, min(mix_core, self.ch, self.cw))

        self.mask_y = make_static_sign_mask(
            height, width, master_key, LABEL_Y
        )
        self.mask_cb = make_static_sign_mask(
            self.ch, self.cw, master_key, LABEL_CB
        )
        self.mask_cr = make_static_sign_mask(
            self.ch, self.cw, master_key, LABEL_CR
        )

        self.y_size = width * height
        self.c_size = self.cw * self.ch
        self.frame_size = self.y_size + 2 * self.c_size

        # Cache epoch transforms because several adjacent frames intentionally
        # share one epoch.
        self.epoch_cache: dict[tuple[bytes, int, int], EpochTransform] = {}

    def epoch_for_frame(self, frame_index: int) -> int:
        # Approximate media timestamp in milliseconds.
        # A tiny 1e-9 is not a security parameter; it only avoids accidental
        # floating-point boundary underflow at exact decimal epoch boundaries.
        t_ms = (frame_index * 1000.0) / self.fps
        return int(math.floor((t_ms + 1e-9) / self.epoch_ms))

    def _epoch_transform(
        self,
        label: bytes,
        epoch: int,
        n: int,
    ) -> EpochTransform:
        cache_key = (label, epoch, n)
        cached = self.epoch_cache.get(cache_key)
        if cached is not None:
            return cached

        p1, ip1 = _derive_affine_permutation(
            n, self.master_key, label, epoch, b"P1"
        )
        p2, ip2 = _derive_affine_permutation(
            n, self.master_key, label, epoch, b"P2"
        )
        signs = _derive_epoch_signs(
            n, self.master_key, label, epoch
        )

        value = EpochTransform(
            perm1=p1,
            invperm1=ip1,
            perm2=p2,
            invperm2=ip2,
            signs=signs,
        )

        # Keep cache bounded. With 100 ms epochs, 32 epochs/channel is about
        # 3.2 seconds of history and already far more than needed for streaming.
        # 32 is an engineering cache-size choice, not a security parameter.
        if len(self.epoch_cache) > 96:  # 32 epochs x 3 channels
            self.epoch_cache.clear()

        self.epoch_cache[cache_key] = value
        return value

    def _mix_core_encrypt(
        self,
        coeff: np.ndarray,
        core: int,
        label: bytes,
        epoch: int,
    ) -> None:
        block = coeff[:core, :core].copy().reshape(-1)

        # Keep only global DC fixed. Mix all other low/mid-frequency
        # coefficients together.
        ac = block[1:].copy()
        tr = self._epoch_transform(label, epoch, ac.size)

        ac = ac[tr.perm1]

        # Orthonormal 1D DCT is a fixed, publicly known orthogonal mixing
        # matrix. Key dependence comes from the two epoch permutations and
        # the epoch sign vector.
        ac = dct(
            ac,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        ac *= tr.signs
        ac = ac[tr.perm2]

        block[1:] = ac
        coeff[:core, :core] = block.reshape(core, core)

    def _mix_core_decrypt(
        self,
        coeff: np.ndarray,
        core: int,
        label: bytes,
        epoch: int,
    ) -> None:
        block = coeff[:core, :core].copy().reshape(-1)
        ac = block[1:].copy()

        tr = self._epoch_transform(label, epoch, ac.size)

        ac = ac[tr.invperm2]
        ac *= tr.signs

        ac = idct(
            ac,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        ac = ac[tr.invperm1]

        block[1:] = ac
        coeff[:core, :core] = block.reshape(core, core)

    def _encrypt_plane(
        self,
        plane: np.ndarray,
        static_mask: np.ndarray,
        gain: np.float32,
        core: int,
        label: bytes,
        epoch: int,
    ) -> np.ndarray:
        x = plane.astype(np.float32)
        x -= CENTER_8BIT

        coeff = dctn(
            x,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        # Whole-plane phase/sign obscuration.
        coeff *= static_mask

        # New V6 operation: time-varying magnitude + phase mixing in the
        # semantically important low/mid-frequency core.
        self._mix_core_encrypt(
            coeff, core, label, epoch
        )

        out = idctn(
            coeff,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        out *= gain
        out += CENTER_8BIT
        np.rint(out, out=out)
        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8)

    def _decrypt_plane(
        self,
        plane: np.ndarray,
        static_mask: np.ndarray,
        gain: np.float32,
        core: int,
        label: bytes,
        epoch: int,
    ) -> np.ndarray:
        x = plane.astype(np.float32)
        x -= CENTER_8BIT
        x /= gain

        coeff = dctn(
            x,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        # Reverse V6 core mixing first...
        self._mix_core_decrypt(
            coeff, core, label, epoch
        )

        # ...then reverse the static sign mask.
        coeff *= static_mask

        out = idctn(
            coeff,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        out += CENTER_8BIT
        np.rint(out, out=out)
        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8)

    def transform_frame(
        self,
        frame: bytes,
        frame_index: int,
        decrypt: bool,
    ) -> bytes:
        if len(frame) != self.frame_size:
            raise ValueError(
                f"Expected {self.frame_size} raw bytes, got {len(frame)}"
            )

        raw = np.frombuffer(frame, dtype=np.uint8)

        cb0 = self.y_size
        cr0 = self.y_size + self.c_size

        y = raw[:cb0].reshape(self.height, self.width)
        cb = raw[cb0:cr0].reshape(self.ch, self.cw)
        cr = raw[cr0:].reshape(self.ch, self.cw)

        epoch = self.epoch_for_frame(frame_index)

        if decrypt:
            oy = self._decrypt_plane(
                y, self.mask_y, self.gain_y,
                self.core_y, LABEL_Y, epoch,
            )
            ocb = self._decrypt_plane(
                cb, self.mask_cb, self.gain_c,
                self.core_c, LABEL_CB, epoch,
            )
            ocr = self._decrypt_plane(
                cr, self.mask_cr, self.gain_c,
                self.core_c, LABEL_CR, epoch,
            )
        else:
            oy = self._encrypt_plane(
                y, self.mask_y, self.gain_y,
                self.core_y, LABEL_Y, epoch,
            )
            ocb = self._encrypt_plane(
                cb, self.mask_cb, self.gain_c,
                self.core_c, LABEL_CB, epoch,
            )
            ocr = self._encrypt_plane(
                cr, self.mask_cr, self.gain_c,
                self.core_c, LABEL_CR, epoch,
            )

        return b"".join(
            (oy.tobytes(), ocb.tobytes(), ocr.tobytes())
        )


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return b""
            raise EOFError("Unexpected EOF inside raw frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def build_decoder(path: str) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-nostdin",
        "-i", path,
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-pix_fmt", "yuv420p",
        "-f", "rawvideo",
        "pipe:1",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )


def build_encoder(
    path: str,
    info: VideoInfo,
    codec: str,
    crf: int,
    preset: str,
) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-nostdin",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-s:v", f"{info.width}x{info.height}",
        "-r", f"{info.fps:.12g}",
        "-i", "pipe:0",
        "-an",
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
    ]

    if Path(path).suffix.lower() in {".mp4", ".mov", ".m4v"}:
        cmd += ["-movflags", "+faststart"]

    cmd.append(path)

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )


def process_video(
    mode: str,
    input_path: str,
    output_path: str,
    key: str,
    workers: int,
    codec: str,
    crf: int,
    preset: str,
    gain_y: float,
    gain_c: float,
    mix_core: int,
    epoch_ms: int,
    progress_every: int,
) -> None:
    info = probe_video(input_path)

    transform = MSPEV7(
        width=info.width,
        height=info.height,
        master_key=normalize_key(key),
        fps=info.fps,
        workers=workers,
        gain_y=gain_y,
        gain_c=gain_c,
        mix_core=mix_core,
        epoch_ms=epoch_ms,
    )

    decoder = build_decoder(input_path)
    encoder = build_encoder(
        output_path, info, codec, crf, preset
    )

    assert decoder.stdout is not None
    assert encoder.stdin is not None

    decrypt = mode == "decrypt"
    frame_index = 0
    transform_seconds = 0.0
    wall_start = time.perf_counter()

    try:
        while True:
            frame = read_exact(
                decoder.stdout,
                transform.frame_size,
            )
            if not frame:
                break

            t0 = time.perf_counter()
            out = transform.transform_frame(
                frame,
                frame_index=frame_index,
                decrypt=decrypt,
            )
            transform_seconds += time.perf_counter() - t0

            encoder.stdin.write(out)
            frame_index += 1

            if (
                progress_every > 0
                and frame_index % progress_every == 0
            ):
                elapsed = time.perf_counter() - wall_start
                print(
                    f"\rframes={frame_index} "
                    f"end-to-end={frame_index/elapsed:.2f} fps",
                    end="",
                    flush=True,
                )
    finally:
        try:
            encoder.stdin.close()
        except Exception:
            pass

    dec_rc = decoder.wait()
    enc_rc = encoder.wait()

    dec_err = (
        decoder.stderr.read().decode("utf-8", errors="replace")
        if decoder.stderr else ""
    )
    enc_err = (
        encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.stderr else ""
    )

    if dec_rc != 0:
        raise SystemExit("FFmpeg decoder failed:\n" + dec_err)
    if enc_rc != 0:
        raise SystemExit("FFmpeg encoder failed:\n" + enc_err)

    wall = time.perf_counter() - wall_start
    transform_fps = (
        frame_index / transform_seconds
        if transform_seconds else 0.0
    )
    total_fps = frame_index / wall if wall else 0.0

    if progress_every:
        print()

    print(f"MSPE-V7 {mode}")
    print(f"resolution        : {info.width}x{info.height}")
    print(f"input rotation    : {info.rotation} degrees")
    print(f"source fps        : {info.fps:.6f}")
    print(f"frames            : {frame_index}")
    print(f"mix core          : {mix_core}x{mix_core}")
    print(f"epoch             : {epoch_ms} ms")
    print(f"gain Y / C        : {gain_y} / {gain_c}")
    print(f"transform speed   : {transform_fps:.2f} fps")
    print(f"end-to-end speed  : {total_fps:.2f} fps")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MSPE-V7 time-varying perceptual video scrambling"
    )

    parser.add_argument(
        "mode",
        choices=("encrypt", "decrypt"),
    )
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("key")

    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="scipy.fft workers; -1 = all CPUs",
    )
    parser.add_argument(
        "--mix-core",
        type=int,
        default=DEFAULT_MIX_CORE,
        help=(
            "low/mid-frequency square side in harmonic coordinates. "
            f"Default: {DEFAULT_MIX_CORE}"
        ),
    )
    parser.add_argument(
        "--epoch-ms",
        type=int,
        default=DEFAULT_EPOCH_MS,
        help=(
            "time-varying transform epoch in milliseconds. "
            f"Default: {DEFAULT_EPOCH_MS}"
        ),
    )
    parser.add_argument(
        "--gain-y",
        type=float,
        default=DEFAULT_GAIN_Y,
    )
    parser.add_argument(
        "--gain-c",
        type=float,
        default=DEFAULT_GAIN_C,
    )
    parser.add_argument(
        "--codec",
        default="libx264",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help=(
            "FFmpeg test/output quality setting; not a security constant. "
            "Default: 18"
        ),
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help=(
            "FFmpeg speed/compression setting; not a security constant. "
            "Default: veryfast"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    process_video(
        mode=args.mode,
        input_path=str(inp),
        output_path=str(out),
        key=args.key,
        workers=args.workers,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        gain_y=args.gain_y,
        gain_c=args.gain_c,
        mix_core=args.mix_core,
        epoch_ms=args.epoch_ms,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
