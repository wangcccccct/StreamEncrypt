#!/usr/bin/env python3
"""
MSPE-V11 Media
=============

V11 is a performance-oriented implementation derived from the V8 algorithm, with epoch rekeying removed and still-image support added.

Design rule
-----------
The cryptographic/perceptual core is inherited from V8, except epoch rekeying is removed:
- YUV420p input/output
- full-frame orthonormal 2D DCT
- key-derived whole-plane +/-1 sign mask
- key-derived orthogonal mixing inside dyadic low-frequency bands
- gains Y=0.58 / C=0.60 by default

V11 keeps the same core transform structure while removing epoch-based rekeying and adding still-image processing.

Performance changes vs V8
-------------------------
1. Cb and Cr are transformed as one batched 3-D array:
       shape = (2, H/2, W/2)
   with DCT axes=(1,2). This is mathematically equivalent to running the two
   chroma planes separately but reduces Python/SciPy call overhead.

2. Frame-level parallel pipeline:
   several independent frames can be transformed concurrently while FFmpeg
   decodes/encodes in parallel. Output order is preserved.

3. Thread-local reusable float32 / uint8 scratch buffers:
   avoids repeatedly allocating multi-megabyte intermediate arrays.

4. Input raw-frame bytearrays are pooled and reused.

5. Low-frequency band transforms are derived once at initialization and reused
   for every frame/image.

Audio is intentionally ignored.

Dependencies
------------
    pip install numpy scipy
    ffmpeg / ffprobe available in PATH

Examples
--------
    python mspe_v11_media.py encrypt input.mp4 encrypted.mp4 "secret-key"
    python mspe_v11_media.py decrypt encrypted.mp4 restored.mp4 "secret-key"
    python mspe_v11_media.py encrypt input.png encrypted.png "secret-key"
    python mspe_v11_media.py decrypt encrypted.png restored.png "secret-key"

Manual performance tuning
-------------------------
    --pipeline-workers 4
        Number of frames transformed concurrently.

    --workers 1
        scipy.fft workers PER transformed frame.

The default is:
    pipeline_workers = min(4, os.cpu_count())
    fft workers       = 2 on >=4-CPU hosts with >=3 frame workers,
                        otherwise 1 (or all CPUs in single-frame mode).

This deliberately favors frame-level parallelism. On the reference 5-CPU
benchmark, 4 frame workers x 2 FFT workers/frame was faster end-to-end than
one frame consuming the whole FFT worker pool. This affects scheduling only;
the frame scheduling does not change transform results.

Security note
-------------
This is experimental perceptual scrambling, not authenticated cryptography.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct, dctn, idctn


# ===========================================================================
# EXPLICIT CONSTANTS AND THEIR ORIGIN
# ===========================================================================

# 128 = 2^7, exactly the midpoint of unsigned 8-bit samples [0,255].
CENTER_8BIT = np.float32(128.0)

# These are engineering headroom parameters inherited unchanged from V8.
#
# Origin:
# V8's benchmark sweep found Y=0.58 to be the best cross-content compromise
# before clipping became dominant on the high-contrast digit test.
# Chroma remained unclipped through at least 0.62 in sampled tests, so 0.60
# was retained to reduce inverse codec-noise amplification.
#
# They are NOT cryptographic constants and remain CLI-adjustable.
DEFAULT_GAIN_Y = 0.58
DEFAULT_GAIN_C = 0.60

# 32 = 2^5, inherited unchanged from V8.
# V7/V8 testing showed 32x32 gave a much better noise/obscuration compromise
# than a larger 64x64 or 128x128 low-frequency mixing core.
DEFAULT_MAX_HARMONIC = 32

# 4,8,16,32 are exact powers of two. These are octave-like frequency bands,
# not opaque magic numbers.
DEFAULT_BAND_EDGES = (4, 8, 16, 32)


# Performance defaults only; they do NOT affect the transform itself.
#
# 4 is a conservative upper bound chosen so a normal desktop can overlap a few
# frames without exploding RAM use. The actual default is min(4,cpu_count).
DEFAULT_MAX_PIPELINE_WORKERS = 4

# Two queued frames per worker gives enough work to keep worker threads busy
# while bounding raw-frame memory. This is purely a pipeline-buffering value.
DEFAULT_INFLIGHT_MULTIPLIER = 2

LABEL_Y = b"Y"
LABEL_CB = b"Cb"
LABEL_CR = b"Cr"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    fps: float
    rotation: int


@dataclass(frozen=True)
class BandTransform:
    coords_y: np.ndarray
    coords_x: np.ndarray
    perm1: np.ndarray
    invperm1: np.ndarray
    perm2: np.ndarray
    invperm2: np.ndarray
    signs: np.ndarray


@dataclass
class Scratch:
    """Per-worker reusable float arrays. Never shared across worker threads."""
    y_f32: np.ndarray
    c_f32: np.ndarray


def normalize_key(key: str) -> bytes:
    """Normalize arbitrary UTF-8 key text to 32 bytes using standard SHA-256."""
    return hashlib.sha256(key.encode("utf-8")).digest()


def parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 30.0
    return float(Fraction(value))


def probe_video(path: str) -> MediaInfo:
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
        raise SystemExit("No decodable image/video stream found.")

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

    return MediaInfo(
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
    Generate +/-1 DCT signs using standardized SHAKE-256.

    For each frequency row ky, SHAKE output is requested as a prefix stream.
    Therefore identical low harmonic coordinates retain the same sign after
    spatial downscaling.
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


def _permutation_from_shake(
    n: int,
    seed: bytes,
) -> tuple[np.ndarray, np.ndarray]:
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
) -> list[BandTransform]:
    """
    Build the key-dependent orthogonal transform for octave-like bands.
    """
    max_harmonic = max(2, int(max_harmonic))

    edges = [e for e in DEFAULT_BAND_EDGES if e <= max_harmonic]
    if not edges or edges[-1] != max_harmonic:
        edges.append(max_harmonic)

    bands: list[BandTransform] = []
    low = 1

    for high in edges:
        if high <= low:
            continue

        ys: list[int] = []
        xs: list[int] = []

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


class MSPEV11:
    """
    V11 transform with fixed key-derived band transforms.

    The object is safe for concurrent transform_frame() calls:
    - immutable masks and band transforms are shared,
    - scratch buffers are thread-local.
    """

    def __init__(
        self,
        width: int,
        height: int,
        master_key: bytes,
        workers: int = 1,
        gain_y: float = DEFAULT_GAIN_Y,
        gain_c: float = DEFAULT_GAIN_C,
        max_harmonic: int = DEFAULT_MAX_HARMONIC,
    ) -> None:
        self.width = width
        self.height = height
        self.cw = width // 2
        self.ch = height // 2

        self.workers = workers
        self.master_key = master_key

        self.gain_y = np.float32(gain_y)
        self.gain_c = np.float32(gain_c)

        self.harm_y = max(2, min(max_harmonic, height, width))
        self.harm_c = max(2, min(max_harmonic, self.ch, self.cw))

        # Store the masks as float32. Multiplication by +/-1 is exact for
        # binary floating-point values, while a float32 mask avoids an int8 ->
        # float conversion in every full-plane multiply. This changes only
        # implementation cost, not the transform.
        self.mask_y = make_static_sign_mask(
            height, width, master_key, LABEL_Y
        ).astype(np.float32)
        self.mask_cb = make_static_sign_mask(
            self.ch, self.cw, master_key, LABEL_CB
        ).astype(np.float32)
        self.mask_cr = make_static_sign_mask(
            self.ch, self.cw, master_key, LABEL_CR
        ).astype(np.float32)

        # Stack chroma masks once so Cb/Cr can be transformed in one batch.
        self.mask_c = np.stack(
            (self.mask_cb, self.mask_cr),
            axis=0,
        )

        self.y_size = width * height
        self.c_size = self.cw * self.ch
        self.frame_size = self.y_size + 2 * self.c_size

        self.bands_y = build_band_transforms(
            self.harm_y, self.master_key, LABEL_Y
        )
        self.bands_cb = build_band_transforms(
            self.harm_c, self.master_key, LABEL_CB
        )
        self.bands_cr = build_band_transforms(
            self.harm_c, self.master_key, LABEL_CR
        )
        self._tls = threading.local()

    def _scratch(self) -> Scratch:
        s = getattr(self._tls, "scratch", None)
        if s is None:
            s = Scratch(
                y_f32=np.empty(
                    (self.height, self.width),
                    dtype=np.float32,
                ),
                c_f32=np.empty(
                    (2, self.ch, self.cw),
                    dtype=np.float32,
                ),
            )
            self._tls.scratch = s
        return s

    def _band_mix_encrypt(
        self,
        coeff: np.ndarray,
        bands: list[BandTransform],
    ) -> None:
        for b in bands:
            v = coeff[b.coords_y, b.coords_x].copy()
            v = v[b.perm1]
            v = dct(
                v,
                type=2,
                norm="ortho",
                workers=self.workers,
                overwrite_x=True,
            )
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
            v = idct(
                v,
                type=2,
                norm="ortho",
                workers=self.workers,
                overwrite_x=True,
            )
            v = v[b.invperm1]
            coeff[b.coords_y, b.coords_x] = v

    def _encrypt_luma(
        self,
        y_u8: np.ndarray,
        dst: np.ndarray,
        scratch: Scratch,
    ) -> None:
        # Exact constant-plane fast path. A constant spatial plane has only a
        # DC DCT coefficient; V8 preserves DC and does not mix it. Most frames
        # reject on the three scalar probes without scanning the full plane.
        v0 = int(y_u8[0, 0])
        if (
            int(y_u8[self.height // 2, self.width // 2]) == v0
            and int(y_u8[-1, -1]) == v0
            and np.all(y_u8 == v0)
        ):
            value = np.uint8(np.clip(np.rint(128.0 + (v0 - 128.0) * float(self.gain_y)), 0, 255))
            dst.fill(value)
            return

        # Convert directly into the reusable float32 buffer.
        np.subtract(
            y_u8,
            CENTER_8BIT,
            out=scratch.y_f32,
            casting="unsafe",
        )

        coeff = dctn(
            scratch.y_f32,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        coeff *= self.mask_y
        self._band_mix_encrypt(
            coeff,
            self.bands_y,
        )

        out = idctn(
            coeff,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        out *= self.gain_y
        out += CENTER_8BIT
        np.rint(out, out=out)
        np.clip(out, 0.0, 255.0, out=out)

        np.copyto(dst, out, casting="unsafe")

    def _decrypt_luma(
        self,
        y_u8: np.ndarray,
        dst: np.ndarray,
        scratch: Scratch,
    ) -> None:
        v0 = int(y_u8[0, 0])
        if (
            int(y_u8[self.height // 2, self.width // 2]) == v0
            and int(y_u8[-1, -1]) == v0
            and np.all(y_u8 == v0)
        ):
            value = np.uint8(np.clip(np.rint(128.0 + (v0 - 128.0) / float(self.gain_y)), 0, 255))
            dst.fill(value)
            return

        np.subtract(
            y_u8,
            CENTER_8BIT,
            out=scratch.y_f32,
            casting="unsafe",
        )
        scratch.y_f32 /= self.gain_y

        coeff = dctn(
            scratch.y_f32,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        self._band_mix_decrypt(
            coeff,
            self.bands_y,
        )
        coeff *= self.mask_y

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

        np.copyto(dst, out, casting="unsafe")

    def _encrypt_chroma(
        self,
        cb_u8: np.ndarray,
        cr_u8: np.ndarray,
        dst: np.ndarray,
        scratch: Scratch,
    ) -> None:
        cb0 = int(cb_u8[0, 0])
        cr0 = int(cr_u8[0, 0])
        if (
            int(cb_u8[self.ch // 2, self.cw // 2]) == cb0
            and int(cb_u8[-1, -1]) == cb0
            and int(cr_u8[self.ch // 2, self.cw // 2]) == cr0
            and int(cr_u8[-1, -1]) == cr0
            and np.all(cb_u8 == cb0)
            and np.all(cr_u8 == cr0)
        ):
            dst[0].fill(np.uint8(np.clip(np.rint(128.0 + (cb0 - 128.0) * float(self.gain_c)), 0, 255)))
            dst[1].fill(np.uint8(np.clip(np.rint(128.0 + (cr0 - 128.0) * float(self.gain_c)), 0, 255)))
            return

        np.subtract(
            cb_u8,
            CENTER_8BIT,
            out=scratch.c_f32[0],
            casting="unsafe",
        )
        np.subtract(
            cr_u8,
            CENTER_8BIT,
            out=scratch.c_f32[1],
            casting="unsafe",
        )

        # One batched DCT call for both independent chroma planes.
        coeff = dctn(
            scratch.c_f32,
            type=2,
            norm="ortho",
            axes=(1, 2),
            workers=self.workers,
            overwrite_x=True,
        )

        coeff *= self.mask_c

        self._band_mix_encrypt(
            coeff[0],
            self.bands_cb,
        )
        self._band_mix_encrypt(
            coeff[1],
            self.bands_cr,
        )

        out = idctn(
            coeff,
            type=2,
            norm="ortho",
            axes=(1, 2),
            workers=self.workers,
            overwrite_x=True,
        )

        out *= self.gain_c
        out += CENTER_8BIT
        np.rint(out, out=out)
        np.clip(out, 0.0, 255.0, out=out)

        np.copyto(dst, out, casting="unsafe")

    def _decrypt_chroma(
        self,
        cb_u8: np.ndarray,
        cr_u8: np.ndarray,
        dst: np.ndarray,
        scratch: Scratch,
    ) -> None:
        cb0 = int(cb_u8[0, 0])
        cr0 = int(cr_u8[0, 0])
        if (
            int(cb_u8[self.ch // 2, self.cw // 2]) == cb0
            and int(cb_u8[-1, -1]) == cb0
            and int(cr_u8[self.ch // 2, self.cw // 2]) == cr0
            and int(cr_u8[-1, -1]) == cr0
            and np.all(cb_u8 == cb0)
            and np.all(cr_u8 == cr0)
        ):
            dst[0].fill(np.uint8(np.clip(np.rint(128.0 + (cb0 - 128.0) / float(self.gain_c)), 0, 255)))
            dst[1].fill(np.uint8(np.clip(np.rint(128.0 + (cr0 - 128.0) / float(self.gain_c)), 0, 255)))
            return

        np.subtract(
            cb_u8,
            CENTER_8BIT,
            out=scratch.c_f32[0],
            casting="unsafe",
        )
        np.subtract(
            cr_u8,
            CENTER_8BIT,
            out=scratch.c_f32[1],
            casting="unsafe",
        )
        scratch.c_f32 /= self.gain_c

        coeff = dctn(
            scratch.c_f32,
            type=2,
            norm="ortho",
            axes=(1, 2),
            workers=self.workers,
            overwrite_x=True,
        )

        self._band_mix_decrypt(
            coeff[0],
            self.bands_cb,
        )
        self._band_mix_decrypt(
            coeff[1],
            self.bands_cr,
        )

        coeff *= self.mask_c

        out = idctn(
            coeff,
            type=2,
            norm="ortho",
            axes=(1, 2),
            workers=self.workers,
            overwrite_x=True,
        )

        out += CENTER_8BIT
        np.rint(out, out=out)
        np.clip(out, 0.0, 255.0, out=out)

        np.copyto(dst, out, casting="unsafe")

    def transform_frame_into(
        self,
        frame: bytes | bytearray | memoryview,
        output: bytearray | memoryview,
        decrypt: bool,
    ) -> None:
        if len(frame) != self.frame_size or len(output) != self.frame_size:
            raise ValueError(
                f"Expected {self.frame_size} input/output bytes"
            )

        raw = np.frombuffer(frame, dtype=np.uint8)
        dst_raw = np.frombuffer(output, dtype=np.uint8)

        cb0 = self.y_size
        cr0 = self.y_size + self.c_size

        y = raw[:cb0].reshape(self.height, self.width)
        cb = raw[cb0:cr0].reshape(self.ch, self.cw)
        cr = raw[cr0:].reshape(self.ch, self.cw)

        dst_y = dst_raw[:cb0].reshape(self.height, self.width)
        dst_c = dst_raw[cb0:].reshape(2, self.ch, self.cw)

        scratch = self._scratch()

        if decrypt:
            self._decrypt_luma(y, dst_y, scratch)
            self._decrypt_chroma(cb, cr, dst_c, scratch)
        else:
            self._encrypt_luma(y, dst_y, scratch)
            self._encrypt_chroma(cb, cr, dst_c, scratch)

    def transform_frame(
        self,
        frame: bytes | bytearray | memoryview,
        decrypt: bool,
    ) -> bytes:
        # Compatibility/convenience wrapper used by tests. The streaming path
        # uses transform_frame_into() with pooled output buffers to avoid this
        # allocation/copy.
        output = bytearray(self.frame_size)
        self.transform_frame_into(
            frame, output, decrypt
        )
        return bytes(output)


# ---------------------------------------------------------------------------
# FFmpeg / streaming helpers
# ---------------------------------------------------------------------------

def read_exact_into(stream, buffer: bytearray) -> bool:
    """
    Fill an existing raw-frame buffer.
    Returns False only on clean EOF before reading any byte.
    """
    mv = memoryview(buffer)
    offset = 0
    size = len(buffer)

    while offset < size:
        n = stream.readinto(mv[offset:])
        if not n:
            if offset == 0:
                return False
            raise EOFError("Unexpected EOF inside raw frame")
        offset += n

    return True


def build_decoder(path: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-i", path,
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=8 * 1024 * 1024,
    )


def build_encoder(
    path: str,
    info: MediaInfo,
    codec: str,
    crf: int,
    preset: str,
    ffmpeg_threads: int = 0,
) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s:v", f"{info.width}x{info.height}",
        "-r", f"{info.fps:.12g}", "-i", "pipe:0",
        "-an", "-c:v", codec, "-preset", preset,
        "-crf", str(crf), "-pix_fmt", "yuv420p",
    ]

    if ffmpeg_threads > 0:
        cmd += ["-threads", str(ffmpeg_threads)]


    if Path(path).suffix.lower() in {".mp4", ".mov", ".m4v"}:
        cmd += ["-movflags", "+faststart"]

    cmd.append(path)

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=8 * 1024 * 1024,
    )



def build_image_encoder(
    path: str,
    info: MediaInfo,
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s:v", f"{info.width}x{info.height}",
            "-i", "pipe:0", "-frames:v", "1", path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=8 * 1024 * 1024,
    )


def process_image(
    mode: str,
    input_path: str,
    output_path: str,
    key: str,
    workers: int,
    gain_y: float,
    gain_c: float,
    max_harmonic: int,
) -> None:
    info = probe_video(input_path)
    _, workers = resolve_parallelism(1, workers)

    if Path(output_path).suffix.lower() in {".jpg", ".jpeg", ".webp"}:
        print(
            "Warning: lossy image output can reduce later decryption quality; "
            "PNG/BMP/TIFF is recommended for encrypted images."
        )

    transform = MSPEV11(
        width=info.width,
        height=info.height,
        master_key=normalize_key(key),
        workers=workers,
        gain_y=gain_y,
        gain_c=gain_c,
        max_harmonic=max_harmonic,
    )

    decoder = build_decoder(input_path)
    assert decoder.stdout is not None
    raw = bytearray(transform.frame_size)
    if not read_exact_into(decoder.stdout, raw):
        raise SystemExit("FFmpeg decoder produced no image frame.")

    dec_rc = decoder.wait()
    dec_err = (
        decoder.stderr.read().decode("utf-8", errors="replace")
        if decoder.stderr else ""
    )
    if dec_rc != 0:
        raise SystemExit("FFmpeg image decoder failed:\n" + dec_err)

    output = bytearray(transform.frame_size)
    transform.transform_frame_into(
        raw,
        output,
        decrypt=(mode == "decrypt"),
    )

    encoder = build_image_encoder(output_path, info)
    assert encoder.stdin is not None
    encoder.stdin.write(output)
    encoder.stdin.close()
    enc_rc = encoder.wait()
    enc_err = (
        encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.stderr else ""
    )
    if enc_rc != 0:
        raise SystemExit("FFmpeg image encoder failed:\n" + enc_err)

    print(f"MSPE-V11 {mode} image")
    print(f"resolution        : {info.width}x{info.height}")
    print(f"max harmonic      : {max_harmonic}")
    print(f"band edges        : {DEFAULT_BAND_EDGES}")
    print(f"gain Y / C        : {gain_y} / {gain_c}")
    print(f"FFT workers       : {workers}")

def resolve_parallelism(
    pipeline_workers: int,
    fft_workers: int,
) -> tuple[int, int]:
    cpus = max(1, os.cpu_count() or 1)

    if pipeline_workers <= 0:
        pipeline_workers = min(
            DEFAULT_MAX_PIPELINE_WORKERS,
            cpus,
        )

    pipeline_workers = max(1, pipeline_workers)

    if fft_workers == 0:
        # Auto:
        # On the reference 5-CPU benchmark, 4 frame workers x 2 pocketfft
        # workers/frame was faster end-to-end than one giant FFT worker pool.
        # Keep the rule conservative on smaller machines. This setting changes
        # scheduling only; it does not change transform results.
        if pipeline_workers >= 3 and cpus >= 4:
            fft_workers = 2
        elif pipeline_workers > 1:
            fft_workers = 1
        else:
            fft_workers = -1

    return pipeline_workers, fft_workers


def process_video(
    mode: str,
    input_path: str,
    output_path: str,
    key: str,
    workers: int,
    pipeline_workers: int,
    max_inflight: int,
    codec: str,
    crf: int,
    preset: str,
    ffmpeg_threads: int,
    gain_y: float,
    gain_c: float,
    max_harmonic: int,
    progress_every: int,
) -> None:
    info = probe_video(input_path)

    pipeline_workers, workers = resolve_parallelism(
        pipeline_workers,
        workers,
    )

    if max_inflight <= 0:
        max_inflight = max(
            1,
            pipeline_workers * DEFAULT_INFLIGHT_MULTIPLIER,
        )

    transform = MSPEV11(
        width=info.width,
        height=info.height,
        master_key=normalize_key(key),
        workers=workers,
        gain_y=gain_y,
        gain_c=gain_c,
        max_harmonic=max_harmonic,
    )

    decoder = build_decoder(input_path)
    encoder = build_encoder(
        output_path,
        info,
        codec,
        crf,
        preset,
        ffmpeg_threads=ffmpeg_threads,
    )

    assert decoder.stdout is not None
    assert encoder.stdin is not None

    decrypt = mode == "decrypt"
    frame_count = 0
    wall_start = time.perf_counter()

    # Raw input buffers are recycled instead of reallocating ~3 MB per 1080p
    # frame.
    free_input_buffers = deque(
        bytearray(transform.frame_size)
        for _ in range(max_inflight)
    )
    free_output_buffers = deque(
        bytearray(transform.frame_size)
        for _ in range(max_inflight)
    )

    # Queue items are (future, input_buffer, output_buffer).
    pending: deque[
        tuple[Future[None], bytearray, bytearray]
    ] = deque()

    def emit_one() -> None:
        nonlocal frame_count

        fut, input_buffer, output_buffer = pending.popleft()
        fut.result()
        encoder.stdin.write(output_buffer)

        free_input_buffers.append(input_buffer)
        free_output_buffers.append(output_buffer)
        frame_count += 1

        if (
            progress_every
            and frame_count % progress_every == 0
        ):
            elapsed = time.perf_counter() - wall_start
            print(
                f"\rframes={frame_count} "
                f"end-to-end={frame_count/elapsed:.2f} fps",
                end="",
                flush=True,
            )

    try:
        if pipeline_workers == 1:
            # Avoid executor overhead in explicitly single-threaded mode.
            input_buffer = bytearray(transform.frame_size)
            output_buffer = bytearray(transform.frame_size)
            while read_exact_into(
                decoder.stdout,
                input_buffer,
            ):
                transform.transform_frame_into(
                    input_buffer,
                    output_buffer,
                    decrypt=decrypt,
                )
                encoder.stdin.write(output_buffer)

                frame_count += 1

                if (
                    progress_every
                    and frame_count % progress_every == 0
                ):
                    elapsed = time.perf_counter() - wall_start
                    print(
                        f"\rframes={frame_count} "
                        f"end-to-end={frame_count/elapsed:.2f} fps",
                        end="",
                        flush=True,
                    )
        else:
            with ThreadPoolExecutor(
                max_workers=pipeline_workers,
                thread_name_prefix="mspe",
            ) as executor:
                while True:
                    # If no reusable input buffer is available, emit the oldest
                    # completed/in-order frame first.
                    if not free_input_buffers or not free_output_buffers:
                        emit_one()

                    input_buffer = free_input_buffers.popleft()
                    output_buffer = free_output_buffers.popleft()

                    if not read_exact_into(
                        decoder.stdout,
                        input_buffer,
                    ):
                        free_input_buffers.append(input_buffer)
                        free_output_buffers.append(output_buffer)
                        break

                    fut = executor.submit(
                        transform.transform_frame_into,
                        input_buffer,
                        output_buffer,
                        decrypt,
                    )
                    pending.append((fut, input_buffer, output_buffer))

                    if len(pending) >= max_inflight:
                        emit_one()

                while pending:
                    emit_one()

    finally:
        try:
            encoder.stdin.close()
        except Exception:
            pass

    dec_rc = decoder.wait()
    enc_rc = encoder.wait()

    dec_err = (
        decoder.stderr.read().decode(
            "utf-8",
            errors="replace",
        )
        if decoder.stderr
        else ""
    )
    enc_err = (
        encoder.stderr.read().decode(
            "utf-8",
            errors="replace",
        )
        if encoder.stderr
        else ""
    )

    if dec_rc != 0:
        raise SystemExit(
            "FFmpeg decoder failed:\n" + dec_err
        )

    if enc_rc != 0:
        raise SystemExit(
            "FFmpeg encoder failed:\n" + enc_err
        )

    wall = time.perf_counter() - wall_start
    end_to_end_fps = (
        frame_count / wall
        if wall
        else 0.0
    )

    if progress_every:
        print()

    print(f"MSPE-V11 {mode}")
    print(f"resolution        : {info.width}x{info.height}")
    print(f"input rotation    : {info.rotation} degrees")
    print(f"source fps        : {info.fps:.6f}")
    print(f"frames            : {frame_count}")
    print(f"max harmonic      : {max_harmonic}")
    print(f"band edges        : {DEFAULT_BAND_EDGES}")
    print(f"gain Y / C        : {gain_y} / {gain_c}")
    print(f"pipeline workers  : {pipeline_workers}")
    print(f"FFT workers/frame : {workers}")
    print(f"max inflight      : {max_inflight}")
    print(f"end-to-end speed  : {end_to_end_fps:.2f} fps")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MSPE-V11: performance-optimized media scrambling "
            "without epoch rekeying"
        )
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
        default=0,
        help=(
            "scipy.fft workers PER frame. "
            "0=auto (default); -1=all CPUs for each frame."
        ),
    )
    parser.add_argument(
        "--pipeline-workers",
        type=int,
        default=0,
        help=(
            "number of frames transformed concurrently. "
            "0=auto (default)."
        ),
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=0,
        help=(
            "maximum queued raw frames. "
            "0=2x pipeline workers (default)."
        ),
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help=(
            "FFmpeg encoder thread count. "
            "0 lets FFmpeg decide (default)."
        ),
    )
    parser.add_argument(
        "--max-harmonic",
        type=int,
        default=DEFAULT_MAX_HARMONIC,
        help=(
            "largest low-frequency harmonic region. "
            f"Default: {DEFAULT_MAX_HARMONIC}"
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
            "FFmpeg quality setting; "
            "not a security constant. Default: 18"
        ),
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help=(
            "FFmpeg speed/compression setting. "
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
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if inp.suffix.lower() in IMAGE_EXTENSIONS:
        if out.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(
                "Image input requires an image output extension: "
                + ", ".join(sorted(IMAGE_EXTENSIONS))
            )
        process_image(
            mode=args.mode,
            input_path=str(inp),
            output_path=str(out),
            key=args.key,
            workers=args.workers,
            gain_y=args.gain_y,
            gain_c=args.gain_c,
            max_harmonic=args.max_harmonic,
        )
    else:
        process_video(
            mode=args.mode,
            input_path=str(inp),
            output_path=str(out),
            key=args.key,
            workers=args.workers,
            pipeline_workers=args.pipeline_workers,
            max_inflight=args.max_inflight,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
            ffmpeg_threads=args.ffmpeg_threads,
            gain_y=args.gain_y,
            gain_c=args.gain_c,
            max_harmonic=args.max_harmonic,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
