#!/usr/bin/env python3
"""
MSPE-V8 Video
=============

Design priority
---------------
V8 deliberately prioritizes:
    1. lower decoded noise after ordinary lossy video compression,
    2. direct decryption after spatial downscaling (e.g. 1080p -> 480p),
    3. hiding obvious coarse spatial structure in ciphertext.

Unlike V6/V7, V8 does NOT change the low-frequency transform rapidly.
It uses a slow 1-second segment rekey and aligns the initial encrypted encode
with a keyframe at the same boundary. This preserves inter-frame correlation
inside almost the whole GOP, which lets H.264/H.265/AV1 prediction work much
better and therefore reduces codec noise after decryption. The slow rekey is
a modest hedge for the ciphertext-only threat model, not a special
known-plaintext-attack mechanism.

V8 does NOT go back to V4's weak "sign only" design.  It uses:

    YUV420p
      -> full-frame orthonormal 2D DCT
      -> key-derived whole-plane +/-1 sign mask
      -> key-derived ORTHOGONAL mixing inside DYADIC LOW-FREQUENCY BANDS
      -> inverse 2D DCT
      -> headroom gain
      -> ordinary FFmpeg encoding

The dyadic bands only mix coefficients with broadly similar spatial
frequencies. This preserves compression friendliness much better than V6/V7's
single large low-frequency mixing core.

Audio is intentionally ignored.

Dependencies
------------
    pip install numpy scipy
    ffmpeg / ffprobe available in PATH

Examples
--------
    python mspe_v8_video.py encrypt input.mp4 encrypted.mp4 "secret-key"
    python mspe_v8_video.py decrypt encrypted.mp4 restored.mp4 "secret-key"

Security note
-------------
This is experimental perceptual scrambling. It is NOT a replacement for
standard authenticated encryption such as AES-GCM / ChaCha20-Poly1305.
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

# 128 = 2^7, exactly the midpoint of unsigned 8-bit samples [0, 255].
# It is dictated by the sample representation, not chosen as a secret value.
CENTER_8BIT = np.float32(128.0)

# Default gains are engineering parameters, NOT cryptographic constants.
#
# V7 used Y=0.50 / C=0.32. V8 removes rapid temporal transform changes and
# constrains low-frequency movement to same-octave bands, so we can use more
# of the legal 8-bit range. A larger gain reduces codec/quantization noise
# amplification during decryption (approximately proportional to 1/gain).
#
# Luma gain sweep on sampled frames from the fixed benchmark set measured
# pre-encode clipping approximately as follows:
#     Y=0.50 -> 0.00495% average
#     Y=0.54 -> 0.03333% average
#     Y=0.58 -> 0.13581% average
#     Y=0.60 -> 0.22085% average, worst sampled frame 0.88349%
#     Y=0.62 -> 0.32962% average
# Color-animation H.264/CRF18 quality continued to improve slightly from
# Y=0.58 to 0.60, but the high-contrast digit benchmark lost about 1.4 dB
# because clipping became the dominant error. Therefore Y=0.58 is retained as
# the better cross-content compromise.
#
# For Cb/Cr, sampled fixed benchmarks and the uploaded real video showed zero
# pre-encode clipping through at least C=0.62. Raising chroma from V7's 0.32
# to C=0.60 materially reduces chroma noise without observed clipping in the
# sampled benchmark. This is empirical tuning, not a security number. Both
# gains are exposed as CLI parameters.
DEFAULT_GAIN_Y = 0.58
DEFAULT_GAIN_C = 0.60

# 32 = 2^5 harmonic indices per axis.
#
# This comes from the V7 parameter sweep: 32x32 preserved materially more
# quality than 64x64 while still covering coarse scene structure.  V8 keeps
# the same maximum semantic region but changes *how* it is mixed.
DEFAULT_MAX_HARMONIC = 32

# Dyadic band boundaries: 4, 8, 16, 32 = 2^2, 2^3, 2^4, 2^5.
#
# These are "nothing-up-my-sleeve" engineering boundaries based on powers of
# two (octave-like frequency bands), not arbitrary hidden constants.
# With max harmonic 32, V8 groups DCT coordinates by m=max(kx,ky):
#     band 0: 1 <= m < 4
#     band 1: 4 <= m < 8
#     band 2: 8 <= m < 16
#     band 3: 16 <= m < 32
# DC (0,0) is never mixed.
DEFAULT_BAND_EDGES = (4, 8, 16, 32)

# 1000 milliseconds = exactly 1 second.
#
# This is NOT a cryptographic constant. V7 changed its transform every 250 ms,
# which damaged inter-frame prediction and visibly increased codec noise. V8
# changes only once per second and asks the initial FFmpeg encoder to put a
# keyframe at the same boundary. This gives the encoder long stretches of
# temporally stable ciphertext while avoiding one fixed low-frequency mapping
# for an entire long video.
#
# 1000 ms is used because it is the exact SI-derived one-second boundary (not
# an opaque tuned number), and the fixed color-animation benchmark showed
# 45.71 dB / 43.33 dB at 1080p / direct 480p with a rekey actually occurring,
# essentially the same quality as the fully static V8 experiment.
#
# This slow rekeying is retained for the original ciphertext-only threat model;
# it is not intended as special known-plaintext-attack protection.
DEFAULT_EPOCH_MS = 1000

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
    """Normalize arbitrary UTF-8 key text to 32 bytes using standard SHA-256."""
    return hashlib.sha256(key.encode("utf-8")).digest()


def parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 30.0
    return float(Fraction(value))


def probe_video(path: str) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:stream_side_data=rotation",
        "-of", "json", path,
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

    # FFmpeg autorotates by default.
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
# Whole-plane sign mask: same harmonic coordinate -> same sign at 1080p/480p
# ---------------------------------------------------------------------------

def make_static_sign_mask(
    height: int,
    width: int,
    master_key: bytes,
    label: bytes,
) -> np.ndarray:
    """
    Generate +/-1 DCT signs using SHAKE-256.

    For each ky we request `width` SHAKE bytes. SHAKE output has the prefix
    property, so the first 427 values at 480p are identical to the first 427
    values at 1080p. Therefore identical low harmonic coordinates use the
    same key sign across resolutions.
    """
    out = np.empty((height, width), dtype=np.int8)
    prefix = master_key + b"|MSPE-V8|STATIC-SIGN|" + label + b"|"

    for ky in range(height):
        seed = prefix + ky.to_bytes(4, "little", signed=False)
        raw = hashlib.shake_256(seed).digest(width)
        row = np.frombuffer(raw, dtype=np.uint8)
        out[ky] = np.where((row & 1) == 0, 1, -1)

    out[0, 0] = 1
    return out


# ---------------------------------------------------------------------------
# Static band-local orthogonal transform
# ---------------------------------------------------------------------------

@dataclass
class BandTransform:
    coords_y: np.ndarray
    coords_x: np.ndarray
    perm1: np.ndarray
    invperm1: np.ndarray
    perm2: np.ndarray
    invperm2: np.ndarray
    signs: np.ndarray


def _permutation_from_shake(
    n: int,
    seed: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic keyed permutation from standardized SHAKE-256 output."""
    if n <= 1:
        p = np.arange(n, dtype=np.int64)
        return p, p.copy()

    raw = hashlib.shake_256(seed).digest(n * 8)
    priorities = np.frombuffer(raw, dtype="<u8")
    p = np.argsort(priorities, kind="stable").astype(np.int64)
    inv = np.empty_like(p)
    inv[p] = np.arange(n, dtype=np.int64)
    return p, inv


def build_band_transforms(
    max_harmonic: int,
    master_key: bytes,
    label: bytes,
    epoch: int,
) -> list[BandTransform]:
    """
    Build static key-dependent transforms for dyadic frequency bands.

    A coefficient belongs to a band according to m=max(kx,ky). Mixing within
    a band keeps energy at a broadly similar spatial scale, which is the main
    V8 noise-reduction mechanism.
    """
    max_harmonic = max(2, int(max_harmonic))

    # Power-of-two boundaries up to max_harmonic. The first band deliberately
    # combines m=1..3 so the very-low-frequency directional coefficients are
    # not left almost untouched in tiny 3-coefficient rings.
    edges = [e for e in DEFAULT_BAND_EDGES if e <= max_harmonic]
    if not edges or edges[-1] != max_harmonic:
        edges.append(max_harmonic)

    bands: list[BandTransform] = []
    low = 1
    for high in edges:
        if high <= low:
            continue

        ys, xs = [], []
        for ky in range(max_harmonic):
            for kx in range(max_harmonic):
                m = max(kx, ky)
                if low <= m < high:
                    ys.append(ky)
                    xs.append(kx)

        if not ys:
            low = high
            continue

        n = len(ys)
        domain = (
            master_key + b"|MSPE-V8|BAND|" + label + b"|"
            + epoch.to_bytes(8, "little", signed=False) + b"|"
            + low.to_bytes(2, "little") + high.to_bytes(2, "little")
        )

        p1, ip1 = _permutation_from_shake(n, domain + b"|P1")
        p2, ip2 = _permutation_from_shake(n, domain + b"|P2")

        raw = hashlib.shake_256(domain + b"|SIGN").digest(n)
        bits = np.frombuffer(raw, dtype=np.uint8)
        signs = np.where((bits & 1) == 0, 1.0, -1.0).astype(np.float32)

        bands.append(BandTransform(
            coords_y=np.asarray(ys, dtype=np.intp),
            coords_x=np.asarray(xs, dtype=np.intp),
            perm1=p1,
            invperm1=ip1,
            perm2=p2,
            invperm2=ip2,
            signs=signs,
        ))
        low = high

    return bands


class MSPEV8:
    def __init__(
        self,
        width: int,
        height: int,
        master_key: bytes,
        fps: float,
        workers: int = -1,
        gain_y: float = DEFAULT_GAIN_Y,
        gain_c: float = DEFAULT_GAIN_C,
        max_harmonic: int = DEFAULT_MAX_HARMONIC,
        epoch_ms: int = DEFAULT_EPOCH_MS,
    ) -> None:
        self.width = width
        self.height = height
        self.cw = width // 2
        self.ch = height // 2
        self.workers = workers
        self.master_key = master_key
        self.fps = fps
        self.epoch_ms = epoch_ms

        self.gain_y = np.float32(gain_y)
        self.gain_c = np.float32(gain_c)

        self.harm_y = max(2, min(max_harmonic, height, width))
        self.harm_c = max(2, min(max_harmonic, self.ch, self.cw))

        self.mask_y = make_static_sign_mask(height, width, master_key, LABEL_Y)
        self.mask_cb = make_static_sign_mask(self.ch, self.cw, master_key, LABEL_CB)
        self.mask_cr = make_static_sign_mask(self.ch, self.cw, master_key, LABEL_CR)

        # Low-frequency band transforms are cached per slow time epoch.
        self.band_cache: dict[tuple[bytes, int, int], list[BandTransform]] = {}

        self.y_size = width * height
        self.c_size = self.cw * self.ch
        self.frame_size = self.y_size + 2 * self.c_size

    def epoch_for_frame(self, frame_index: int) -> int:
        # Approximate media timestamp from CFR frame index. The tiny epsilon is
        # only a floating-point boundary guard, not a security parameter.
        t_ms = (frame_index * 1000.0) / self.fps
        return int(math.floor((t_ms + 1e-9) / self.epoch_ms))

    def _bands(self, label: bytes, harmonic: int, epoch: int) -> list[BandTransform]:
        key = (label, harmonic, epoch)
        value = self.band_cache.get(key)
        if value is None:
            value = build_band_transforms(
                harmonic, self.master_key, label, epoch
            )
            # A bounded cache is enough for streaming; 12 entries ~= four
            # epochs x three channels. Cache size is a memory choice only.
            if len(self.band_cache) >= 12:
                self.band_cache.clear()
            self.band_cache[key] = value
        return value

    def _band_mix_encrypt(
        self,
        coeff: np.ndarray,
        bands: list[BandTransform],
    ) -> None:
        for b in bands:
            v = coeff[b.coords_y, b.coords_x].copy()
            v = v[b.perm1]
            # Orthonormal DCT: energy preserving, no hidden scale factor.
            v = dct(v, type=2, norm="ortho", workers=self.workers, overwrite_x=True)
            v *= b.signs
            v = v[b.perm2]
            coeff[b.coords_y, b.coords_x] = v

    def _band_mix_decrypt(
        self,
        coeff: np.ndarray,
        bands: list[BandTransform],
    ) -> None:
        for b in bands:
            v = coeff[b.coords_y, b.coords_x].copy()
            v = v[b.invperm2]
            v *= b.signs
            v = idct(v, type=2, norm="ortho", workers=self.workers, overwrite_x=True)
            v = v[b.invperm1]
            coeff[b.coords_y, b.coords_x] = v

    def _encrypt_plane(
        self,
        plane: np.ndarray,
        mask: np.ndarray,
        gain: np.float32,
        label: bytes,
        harmonic: int,
        epoch: int,
    ) -> np.ndarray:
        x = plane.astype(np.float32)
        x -= CENTER_8BIT

        coeff = dctn(
            x, type=2, norm="ortho",
            workers=self.workers, overwrite_x=True,
        )

        coeff *= mask
        self._band_mix_encrypt(coeff, self._bands(label, harmonic, epoch))

        out = idctn(
            coeff, type=2, norm="ortho",
            workers=self.workers, overwrite_x=True,
        )
        out *= gain
        out += CENTER_8BIT
        np.rint(out, out=out)
        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8)

    def _decrypt_plane(
        self,
        plane: np.ndarray,
        mask: np.ndarray,
        gain: np.float32,
        label: bytes,
        harmonic: int,
        epoch: int,
    ) -> np.ndarray:
        x = plane.astype(np.float32)
        x -= CENTER_8BIT
        x /= gain

        coeff = dctn(
            x, type=2, norm="ortho",
            workers=self.workers, overwrite_x=True,
        )

        self._band_mix_decrypt(coeff, self._bands(label, harmonic, epoch))
        coeff *= mask

        out = idctn(
            coeff, type=2, norm="ortho",
            workers=self.workers, overwrite_x=True,
        )
        out += CENTER_8BIT
        np.rint(out, out=out)
        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8)

    def transform_frame(
        self, frame: bytes, frame_index: int, decrypt: bool
    ) -> bytes:
        if len(frame) != self.frame_size:
            raise ValueError(
                f"Expected {self.frame_size} bytes, got {len(frame)}"
            )

        raw = np.frombuffer(frame, dtype=np.uint8)
        cb0 = self.y_size
        cr0 = self.y_size + self.c_size

        y = raw[:cb0].reshape(self.height, self.width)
        cb = raw[cb0:cr0].reshape(self.ch, self.cw)
        cr = raw[cr0:].reshape(self.ch, self.cw)

        epoch = self.epoch_for_frame(frame_index)

        if decrypt:
            oy = self._decrypt_plane(y, self.mask_y, self.gain_y, LABEL_Y, self.harm_y, epoch)
            ocb = self._decrypt_plane(cb, self.mask_cb, self.gain_c, LABEL_CB, self.harm_c, epoch)
            ocr = self._decrypt_plane(cr, self.mask_cr, self.gain_c, LABEL_CR, self.harm_c, epoch)
        else:
            oy = self._encrypt_plane(y, self.mask_y, self.gain_y, LABEL_Y, self.harm_y, epoch)
            ocb = self._encrypt_plane(cb, self.mask_cb, self.gain_c, LABEL_CB, self.harm_c, epoch)
            ocr = self._encrypt_plane(cr, self.mask_cr, self.gain_c, LABEL_CR, self.harm_c, epoch)

        return b"".join((oy.tobytes(), ocb.tobytes(), ocr.tobytes()))


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
    return subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-i", path,
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
        ],
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
    force_epoch_ms: int | None = None,
) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s:v", f"{info.width}x{info.height}",
        "-r", f"{info.fps:.12g}", "-i", "pipe:0",
        "-an", "-c:v", codec, "-preset", preset,
        "-crf", str(crf), "-pix_fmt", "yuv420p",
    ]
    if force_epoch_ms and force_epoch_ms > 0:
        seconds = force_epoch_ms / 1000.0
        cmd += [
            "-force_key_frames",
            f"expr:gte(t,n_forced*{seconds:.6f})",
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
    max_harmonic: int,
    epoch_ms: int,
    progress_every: int,
) -> None:
    info = probe_video(input_path)
    transform = MSPEV8(
        width=info.width,
        height=info.height,
        master_key=normalize_key(key),
        fps=info.fps,
        workers=workers,
        gain_y=gain_y,
        gain_c=gain_c,
        max_harmonic=max_harmonic,
        epoch_ms=epoch_ms,
    )

    decoder = build_decoder(input_path)
    encoder = build_encoder(
        output_path, info, codec, crf, preset,
        force_epoch_ms=(epoch_ms if mode == "encrypt" else None),
    )

    assert decoder.stdout is not None
    assert encoder.stdin is not None

    decrypt = mode == "decrypt"
    frames = 0
    transform_seconds = 0.0
    wall_start = time.perf_counter()

    try:
        while True:
            frame = read_exact(decoder.stdout, transform.frame_size)
            if not frame:
                break

            t0 = time.perf_counter()
            out = transform.transform_frame(
                frame, frame_index=frames, decrypt=decrypt
            )
            transform_seconds += time.perf_counter() - t0

            encoder.stdin.write(out)
            frames += 1

            if progress_every and frames % progress_every == 0:
                elapsed = time.perf_counter() - wall_start
                print(
                    f"\rframes={frames} end-to-end={frames/elapsed:.2f} fps",
                    end="", flush=True,
                )
    finally:
        try:
            encoder.stdin.close()
        except Exception:
            pass

    dec_rc = decoder.wait()
    enc_rc = encoder.wait()

    dec_err = decoder.stderr.read().decode("utf-8", errors="replace") if decoder.stderr else ""
    enc_err = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""

    if dec_rc != 0:
        raise SystemExit("FFmpeg decoder failed:\n" + dec_err)
    if enc_rc != 0:
        raise SystemExit("FFmpeg encoder failed:\n" + enc_err)

    wall = time.perf_counter() - wall_start
    tfps = frames / transform_seconds if transform_seconds else 0.0
    efps = frames / wall if wall else 0.0

    if progress_every:
        print()

    print(f"MSPE-V8 {mode}")
    print(f"resolution        : {info.width}x{info.height}")
    print(f"input rotation    : {info.rotation} degrees")
    print(f"source fps        : {info.fps:.6f}")
    print(f"frames            : {frames}")
    print(f"max harmonic      : {max_harmonic}")
    print(f"band edges        : {DEFAULT_BAND_EDGES}")
    print(f"rekey epoch       : {epoch_ms} ms")
    print(f"gain Y / C        : {gain_y} / {gain_c}")
    print(f"transform speed   : {tfps:.2f} fps")
    print(f"end-to-end speed  : {efps:.2f} fps")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MSPE-V8 low-noise perceptual video scrambling"
    )
    parser.add_argument("mode", choices=("encrypt", "decrypt"))
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("key")

    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--max-harmonic", type=int, default=DEFAULT_MAX_HARMONIC,
        help=f"largest low-frequency harmonic region. Default: {DEFAULT_MAX_HARMONIC}",
    )
    parser.add_argument(
        "--epoch-ms", type=int, default=DEFAULT_EPOCH_MS,
        help=f"slow low-frequency rekey interval. Default: {DEFAULT_EPOCH_MS} ms",
    )
    parser.add_argument("--gain-y", type=float, default=DEFAULT_GAIN_Y)
    parser.add_argument("--gain-c", type=float, default=DEFAULT_GAIN_C)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument(
        "--crf", type=int, default=18,
        help="FFmpeg quality setting; not a security constant. Default: 18",
    )
    parser.add_argument(
        "--preset", default="veryfast",
        help="FFmpeg speed/compression setting. Default: veryfast",
    )
    parser.add_argument("--progress-every", type=int, default=30)

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
        max_harmonic=args.max_harmonic,
        epoch_ms=args.epoch_ms,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
