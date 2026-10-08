#!/usr/bin/env python3
"""
MSPE-V5 Video
=============

Purpose
-------
Perceptual video scrambling that remains approximately decryptable after
ordinary lossy transcoding and spatial downscaling.

V5 change vs V4
---------------
V4 only flipped DCT coefficient signs. That preserves |DCT coefficient|
at every frequency coordinate, so coarse low-frequency structure can still
produce a visible "shadow" of the original scene.

V5 therefore uses TWO reversible frequency-domain operations:

    1) key-derived +/-1 sign scrambling over the whole DCT plane;
    2) key-derived permutation inside a LOW-FREQUENCY CORE.

The second step moves low-frequency magnitudes away from their original
coordinates, which reduces silhouette / coarse-layout leakage without
moving low-frequency energy into high-frequency regions that a resize would
discard.

Video pipeline
--------------
FFmpeg decode -> raw YUV420p -> MSPE-V5 -> FFmpeg encode.

Audio is deliberately ignored.

Dependencies
------------
    pip install numpy scipy
    ffmpeg / ffprobe available in PATH

Examples
--------
Encrypt:
    python mspe_v5_video.py encrypt input.mp4 encrypted.mp4 "secret-key"

Decrypt:
    python mspe_v5_video.py decrypt encrypted.mp4 restored.mp4 "secret-key"

IMPORTANT
---------
This is experimental perceptual scrambling, not authenticated cryptography.
For cryptographic confidentiality/integrity, use a standard construction
such as AES-GCM or ChaCha20-Poly1305 in addition to (or instead of) this.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn


# ---------------------------------------------------------------------------
# EXPLICIT / "NOTHING-UP-MY-SLEEVE" CONSTANTS
# ---------------------------------------------------------------------------

# 128 = 2^7, the exact midpoint used when centering an unsigned 8-bit plane.
# This is determined by the 8-bit sample representation, not chosen secretly.
CENTER_8BIT = np.float32(128.0)

# These gains are NOT cryptographic constants.
#
# Origin:
#   They are the V3/V4 empirical headroom values from the preceding benchmark:
#       Y  = 0.42
#       Cb = 0.28
#       Cr = 0.28
#   They were selected to keep pre-encode clipping at or near zero on the
#   earlier digit/gradient tests while avoiding excessive inverse noise gain.
#
# They are exposed as command-line parameters so there is no hidden tuning.
DEFAULT_GAIN_Y = 0.42
DEFAULT_GAIN_C = 0.28

# 16 = 2^4.
#
# This is NOT a security constant. It is an engineering parameter controlling
# how much of the low-frequency DCT region is magnitude-scrambled.
#
# Why 16:
#   - a power of two, easy to reason about and benchmark;
#   - the first 16 horizontal/vertical harmonics carry very coarse scene
#     layout. At 1080p, H/16 is about 67.5 pixels, so this core targets
#     large silhouettes / broad brightness structure rather than fine detail;
#   - our 16/32/64 benchmark showed 16 already removed the visible coarse
#     shadow while preserving substantially more 480p recovery quality;
#   - it exists comfortably in 1080p and 854x480 YUV420p luma/chroma planes;
#   - it is exposed as --core, so it is transparent and independently tunable.
DEFAULT_CORE = 16

LABEL_Y = b"Y"
LABEL_CB = b"Cb"
LABEL_CR = b"Cr"


@dataclass(frozen=True)
class VideoInfo:
    # width/height are the dimensions AFTER FFmpeg's default autorotation.
    # This matters for phone videos that store 1920x1080 plus a +/-90 degree
    # display-rotation tag instead of physically rotating the encoded frames.
    width: int
    height: int
    fps: float
    rotation: int


def normalize_key(key: str) -> bytes:
    """
    SHA-256 is used only to convert an arbitrary UTF-8 passphrase into a
    fixed 32-byte internal key.

    SHA-256 is a standardized hash; there are no custom hash constants here.
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

    # FFmpeg applies display rotation automatically unless -noautorotate is
    # requested. Therefore the rawvideo pipe dimensions must match the
    # autorotated frame, not merely the coded width/height from the stream.
    #
    # A +/-90 degree rotation swaps width and height. 0/180 does not.
    rotation = 0
    for side in s.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(round(float(side["rotation"]))) % 360
            break

    if rotation in (90, 270):
        w, h = coded_h, coded_w
    else:
        w, h = coded_w, coded_h

    if w % 2 or h % 2:
        raise SystemExit(
            f"YUV420p needs even display dimensions; got {w}x{h}."
        )

    return VideoInfo(
        width=w,
        height=h,
        fps=parse_rate(s.get("avg_frame_rate", "30/1")),
        rotation=rotation,
    )


def make_sign_mask(
    height: int,
    width: int,
    master_key: bytes,
    label: bytes,
) -> np.ndarray:
    """
    Generate a resolution-compatible +/-1 mask with SHAKE-256.

    Why row-wise SHAKE-256?
    -----------------------
    For each frequency row ky, SHAKE-256 produces an arbitrarily long byte
    stream:

        SHAKE256(key || label || ky)

    The sign at kx is the low bit of output byte kx.

    SHAKE output has the prefix property: requesting 427 bytes produces the
    same first 427 bytes as requesting 960 bytes. Therefore the same low
    harmonic coordinate (kx, ky) gets the same sign at 1080p and 480p.

    This deliberately avoids private / unexplained integer mixer constants.
    """
    mask = np.empty((height, width), dtype=np.int8)

    prefix = master_key + b"|MSPE-V5|SIGN|" + label + b"|"

    for ky in range(height):
        row_seed = prefix + ky.to_bytes(4, "little", signed=False)
        random_bytes = hashlib.shake_256(row_seed).digest(width)
        row = np.frombuffer(random_bytes, dtype=np.uint8)
        mask[ky, :] = np.where((row & 1) == 0, 1, -1)

    # Preserve DC: DC is the frame/channel mean and is not part of the
    # silhouette-scrambling permutation.
    mask[0, 0] = 1
    return mask


def make_core_permutation(
    core: int,
    master_key: bytes,
    label: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a keyed permutation for all AC coefficients in core x core.

    DC (flat index 0) is excluded.

    SHAKE-256 emits one 64-bit sort key per AC coefficient. Sorting those keys
    gives a deterministic key-dependent permutation. This is precomputed once,
    so its cost is negligible compared with per-frame DCT/IDCT.

    The permutation depends on:
        master key, channel label, core size
    but NOT on full image resolution. Thus the same low-frequency harmonic
    core uses the same permutation at 1080p and 480p.
    """
    count = core * core - 1
    if count <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    seed = (
        master_key
        + b"|MSPE-V5|CORE-PERM|"
        + label
        + b"|"
        + core.to_bytes(2, "little", signed=False)
    )

    raw = hashlib.shake_256(seed).digest(count * 8)
    priorities = np.frombuffer(raw, dtype="<u8")

    perm = np.argsort(priorities, kind="stable")
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(count, dtype=perm.dtype)
    return perm, inverse


class MSPEV5:
    def __init__(
        self,
        width: int,
        height: int,
        master_key: bytes,
        workers: int = -1,
        gain_y: float = DEFAULT_GAIN_Y,
        gain_c: float = DEFAULT_GAIN_C,
        core: int = DEFAULT_CORE,
    ) -> None:
        self.width = width
        self.height = height
        self.cw = width // 2
        self.ch = height // 2
        self.workers = workers

        self.gain_y = np.float32(gain_y)
        self.gain_c = np.float32(gain_c)

        # The effective core is separately bounded for luma/chroma.
        self.core_y = max(1, min(core, height, width))
        self.core_c = max(1, min(core, self.ch, self.cw))

        self.mask_y = make_sign_mask(
            height, width, master_key, LABEL_Y
        )
        self.mask_cb = make_sign_mask(
            self.ch, self.cw, master_key, LABEL_CB
        )
        self.mask_cr = make_sign_mask(
            self.ch, self.cw, master_key, LABEL_CR
        )

        self.perm_y, self.invperm_y = make_core_permutation(
            self.core_y, master_key, LABEL_Y
        )
        self.perm_cb, self.invperm_cb = make_core_permutation(
            self.core_c, master_key, LABEL_CB
        )
        self.perm_cr, self.invperm_cr = make_core_permutation(
            self.core_c, master_key, LABEL_CR
        )

        self.y_size = width * height
        self.c_size = self.cw * self.ch
        self.frame_size = self.y_size + 2 * self.c_size

    @staticmethod
    def _permute_core(
        coeff: np.ndarray,
        core: int,
        permutation: np.ndarray,
    ) -> None:
        if core <= 1:
            return

        block = coeff[:core, :core].copy().reshape(-1)
        ac = block[1:].copy()
        block[1:] = ac[permutation]
        coeff[:core, :core] = block.reshape(core, core)

    def _encrypt_plane(
        self,
        plane: np.ndarray,
        mask: np.ndarray,
        gain: np.float32,
        core: int,
        permutation: np.ndarray,
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

        # V4 operation: sign/phase scrambling.
        coeff *= mask

        # V5 addition: redistribute low-frequency magnitudes.
        self._permute_core(coeff, core, permutation)

        scrambled = idctn(
            coeff,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        scrambled *= gain
        scrambled += CENTER_8BIT

        np.rint(scrambled, out=scrambled)
        np.clip(scrambled, 0.0, 255.0, out=scrambled)
        return scrambled.astype(np.uint8)

    def _decrypt_plane(
        self,
        plane: np.ndarray,
        mask: np.ndarray,
        gain: np.float32,
        core: int,
        inverse_permutation: np.ndarray,
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

        # Reverse in opposite order:
        # encrypt = sign -> permutation
        # decrypt = inverse permutation -> sign
        self._permute_core(
            coeff,
            core,
            inverse_permutation,
        )

        coeff *= mask

        restored = idctn(
            coeff,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        restored += CENTER_8BIT
        np.rint(restored, out=restored)
        np.clip(restored, 0.0, 255.0, out=restored)
        return restored.astype(np.uint8)

    def transform_frame(self, frame: bytes, decrypt: bool) -> bytes:
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

        if decrypt:
            oy = self._decrypt_plane(
                y, self.mask_y, self.gain_y,
                self.core_y, self.invperm_y,
            )
            ocb = self._decrypt_plane(
                cb, self.mask_cb, self.gain_c,
                self.core_c, self.invperm_cb,
            )
            ocr = self._decrypt_plane(
                cr, self.mask_cr, self.gain_c,
                self.core_c, self.invperm_cr,
            )
        else:
            oy = self._encrypt_plane(
                y, self.mask_y, self.gain_y,
                self.core_y, self.perm_y,
            )
            ocb = self._encrypt_plane(
                cb, self.mask_cb, self.gain_c,
                self.core_c, self.perm_cb,
            )
            ocr = self._encrypt_plane(
                cr, self.mask_cr, self.gain_c,
                self.core_c, self.perm_cr,
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
            raise EOFError("EOF inside raw frame")
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
    core: int,
    progress_every: int,
) -> None:
    info = probe_video(input_path)

    transform = MSPEV5(
        width=info.width,
        height=info.height,
        master_key=normalize_key(key),
        workers=workers,
        gain_y=gain_y,
        gain_c=gain_c,
        core=core,
    )

    decoder = build_decoder(input_path)
    encoder = build_encoder(
        output_path,
        info,
        codec,
        crf,
        preset,
    )

    assert decoder.stdout is not None
    assert encoder.stdin is not None

    decrypt = mode == "decrypt"
    frames = 0
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
                decrypt=decrypt,
            )
            transform_seconds += time.perf_counter() - t0

            encoder.stdin.write(out)
            frames += 1

            if progress_every and frames % progress_every == 0:
                elapsed = time.perf_counter() - wall_start
                speed = frames / elapsed
                print(
                    f"\rframes={frames} end-to-end={speed:.2f} fps",
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
        frames / transform_seconds if transform_seconds else 0.0
    )
    total_fps = frames / wall if wall else 0.0

    if progress_every:
        print()

    print(f"MSPE-V5 {mode}")
    print(f"resolution        : {info.width}x{info.height}")
    print(f"input rotation    : {info.rotation} degrees")
    print(f"frames            : {frames}")
    print(f"low-frequency core: {core}x{core}")
    print(f"gain Y / C        : {gain_y} / {gain_c}")
    print(f"transform speed   : {transform_fps:.2f} fps")
    print(f"end-to-end speed  : {total_fps:.2f} fps")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MSPE-V5 video perceptual scrambling"
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
        "--core",
        type=int,
        default=DEFAULT_CORE,
        help=(
            "low-frequency square side length. "
            f"Default: {DEFAULT_CORE}"
        ),
    )
    parser.add_argument(
        "--gain-y",
        type=float,
        default=DEFAULT_GAIN_Y,
        help=(
            "luma headroom gain. "
            f"Default: {DEFAULT_GAIN_Y}"
        ),
    )
    parser.add_argument(
        "--gain-c",
        type=float,
        default=DEFAULT_GAIN_C,
        help=(
            "chroma headroom gain. "
            f"Default: {DEFAULT_GAIN_C}"
        ),
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
            "FFmpeg encoder CRF. This is an encoding/test setting, "
            "not an MSPE security constant. Default: 18"
        ),
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help=(
            "FFmpeg encoder preset. This controls speed/compression only. "
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
        core=args.core,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
