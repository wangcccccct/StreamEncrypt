#!/usr/bin/env python3
"""
MSPE-V4 Video
==============

High-throughput video-oriented implementation of the MSPE-V3 transform.

Key design choices:
- Video file in / video file out.
- No audio processing. Output contains video only.
- FFmpeg handles decode/encode.
- Python operates directly on raw YUV420p planes.
- Y is full resolution; Cb/Cr are half width and half height.
- Spectral masks are generated once per resolution and cached.
- scipy.fft DCT/IDCT uses multiple CPU workers.
- Frames are streamed; the whole video is never loaded into memory.

Requirements:
    pip install numpy scipy
    ffmpeg and ffprobe must be installed and available on PATH

Examples:
    python mspe_v4_video.py encrypt input.mp4 encrypted.mp4 "my-key"
    python mspe_v4_video.py decrypt encrypted.mp4 restored.mp4 "my-key"

Faster encoding:
    python mspe_v4_video.py encrypt input.mp4 encrypted.mp4 "my-key" \
        --preset ultrafast --crf 20

Higher quality:
    python mspe_v4_video.py encrypt input.mp4 encrypted.mp4 "my-key" \
        --preset veryfast --crf 16

Important:
This is an experimental perceptual scrambling transform. It is not intended
as a substitute for standard authenticated cryptography such as AES-GCM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.fft import dctn, idctn


# Same transform parameters as MSPE-V3.
GAIN_Y = np.float32(0.42)
GAIN_CB = np.float32(0.28)
GAIN_CR = np.float32(0.28)

LABEL_Y = b"Y"
LABEL_CB = b"Cb"
LABEL_CR = b"Cr"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames_hint: int | None
    duration: float | None


def normalize_key(key: str) -> bytes:
    """Normalize arbitrary UTF-8 text to a fixed 32-byte master key."""
    return hashlib.sha256(key.encode("utf-8")).digest()


def parse_rate(value: str) -> float:
    """Parse FFmpeg rational frame rate such as '30000/1001'."""
    if not value or value == "0/0":
        return 30.0
    try:
        return float(Fraction(value))
    except Exception:
        return float(value)


def probe_video(path: str) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames:format=duration",
        "-of", "json",
        path,
    ]

    try:
        raw = subprocess.check_output(cmd)
    except FileNotFoundError:
        raise SystemExit("ffprobe was not found. Install FFmpeg first.")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffprobe failed for {path}") from exc

    data = json.loads(raw)
    streams = data.get("streams", [])
    if not streams:
        raise SystemExit("No video stream found.")

    s = streams[0]
    width = int(s["width"])
    height = int(s["height"])

    if width % 2 or height % 2:
        raise SystemExit(
            f"YUV420p requires even dimensions, got {width}x{height}."
        )

    fps = parse_rate(s.get("avg_frame_rate", "30/1"))

    nb_frames = s.get("nb_frames")
    frames_hint = int(nb_frames) if nb_frames and nb_frames.isdigit() else None

    duration_raw = data.get("format", {}).get("duration")
    duration = float(duration_raw) if duration_raw else None

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frames_hint=frames_hint,
        duration=duration,
    )


def _mix64(z: np.ndarray) -> np.ndarray:
    """
    Vectorized 64-bit mixing.
    This is used only to generate a deterministic spectral sign pattern.
    """
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z


def generate_sign_mask(
    height: int,
    width: int,
    master_key: bytes,
    label: bytes,
) -> np.ndarray:
    """
    Generate the MSPE +/-1 frequency-domain mask.

    Frequency coordinate -> deterministic sign.
    DC is preserved so the average channel level is not inverted.
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

    z = _mix64(z)

    # int8 saves mask memory; NumPy promotes during multiply.
    mask = np.where(
        (z & np.uint64(1)) == 0,
        np.int8(1),
        np.int8(-1),
    )

    mask[0, 0] = 1
    return mask


class MSPEV4:
    """
    Cached, video-oriented MSPE transform.

    Masks:
      Y  : H x W
      Cb : H/2 x W/2
      Cr : H/2 x W/2

    This is substantially cheaper than the older RGB/YCbCr 4:4:4 prototype.
    """

    def __init__(
        self,
        width: int,
        height: int,
        master_key: bytes,
        workers: int,
    ) -> None:
        self.width = width
        self.height = height
        self.cw = width // 2
        self.ch = height // 2
        self.workers = workers

        self.mask_y = generate_sign_mask(
            height, width, master_key, LABEL_Y
        )
        self.mask_cb = generate_sign_mask(
            self.ch, self.cw, master_key, LABEL_CB
        )
        self.mask_cr = generate_sign_mask(
            self.ch, self.cw, master_key, LABEL_CR
        )

        self.y_size = width * height
        self.c_size = self.cw * self.ch
        self.frame_size = self.y_size + 2 * self.c_size

    def _encrypt_plane(
        self,
        plane_u8: np.ndarray,
        mask: np.ndarray,
        gain: np.float32,
    ) -> np.ndarray:
        # Convert to float32 and center around 128.
        x = plane_u8.astype(np.float32)
        x -= np.float32(128.0)

        # overwrite_x=True allows scipy.fft to reuse input memory where possible.
        coeff = dctn(
            x,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        coeff *= mask

        scrambled = idctn(
            coeff,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        scrambled *= gain
        scrambled += np.float32(128.0)

        np.rint(scrambled, out=scrambled)
        np.clip(scrambled, 0.0, 255.0, out=scrambled)

        return scrambled.astype(np.uint8)

    def _decrypt_plane(
        self,
        plane_u8: np.ndarray,
        mask: np.ndarray,
        gain: np.float32,
    ) -> np.ndarray:
        x = plane_u8.astype(np.float32)
        x -= np.float32(128.0)
        x /= gain

        coeff = dctn(
            x,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        coeff *= mask

        recovered = idctn(
            coeff,
            type=2,
            norm="ortho",
            workers=self.workers,
            overwrite_x=True,
        )

        recovered += np.float32(128.0)

        np.rint(recovered, out=recovered)
        np.clip(recovered, 0.0, 255.0, out=recovered)

        return recovered.astype(np.uint8)

    def transform_frame(
        self,
        frame: bytes,
        decrypt: bool,
    ) -> bytes:
        if len(frame) != self.frame_size:
            raise ValueError(
                f"Expected {self.frame_size} raw bytes, got {len(frame)}"
            )

        raw = np.frombuffer(frame, dtype=np.uint8)

        y0 = 0
        cb0 = self.y_size
        cr0 = self.y_size + self.c_size

        y = raw[y0:cb0].reshape(self.height, self.width)
        cb = raw[cb0:cr0].reshape(self.ch, self.cw)
        cr = raw[cr0:].reshape(self.ch, self.cw)

        if decrypt:
            oy = self._decrypt_plane(y, self.mask_y, GAIN_Y)
            ocb = self._decrypt_plane(cb, self.mask_cb, GAIN_CB)
            ocr = self._decrypt_plane(cr, self.mask_cr, GAIN_CR)
        else:
            oy = self._encrypt_plane(y, self.mask_y, GAIN_Y)
            ocb = self._encrypt_plane(cb, self.mask_cb, GAIN_CB)
            ocr = self._encrypt_plane(cr, self.mask_cr, GAIN_CR)

        # byte joins avoid constructing a temporary HxWx3 array.
        return b"".join((oy.tobytes(), ocb.tobytes(), ocr.tobytes()))


def read_exact(stream, size: int) -> bytes:
    """
    Read exactly one raw video frame.
    Returns b'' only at clean EOF.
    """
    chunks = []
    remaining = size

    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return b""
            raise EOFError("Unexpected EOF in the middle of a raw frame.")
        chunks.append(chunk)
        remaining -= len(chunk)

    if len(chunks) == 1:
        return chunks[0]
    return b"".join(chunks)


def build_decoder(input_path: str) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-nostdin",
        "-i", input_path,
        "-map", "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt", "yuv420p",
        "-f", "rawvideo",
        "pipe:1",
    ]

    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1024 * 1024,
        )
    except FileNotFoundError:
        raise SystemExit("ffmpeg was not found. Install FFmpeg first.")


def build_encoder(
    output_path: str,
    info: VideoInfo,
    codec: str,
    crf: int,
    preset: str,
    ffmpeg_threads: int,
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

    if ffmpeg_threads > 0:
        cmd += ["-threads", str(ffmpeg_threads)]

    # MP4/MOV benefits from faststart. Harmless for common MP4 use.
    suffix = Path(output_path).suffix.lower()
    if suffix in {".mp4", ".mov", ".m4v"}:
        cmd += ["-movflags", "+faststart"]

    cmd.append(output_path)

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
    ffmpeg_threads: int,
    progress_every: int,
) -> None:
    info = probe_video(input_path)

    master_key = normalize_key(key)

    # scipy.fft convention: workers=-1 means all CPUs.
    transform = MSPEV4(
        width=info.width,
        height=info.height,
        master_key=master_key,
        workers=workers,
    )

    decoder = build_decoder(input_path)
    encoder = build_encoder(
        output_path,
        info,
        codec,
        crf,
        preset,
        ffmpeg_threads,
    )

    assert decoder.stdout is not None
    assert encoder.stdin is not None

    decrypt = mode == "decrypt"

    frame_count = 0
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
            output_frame = transform.transform_frame(
                frame,
                decrypt=decrypt,
            )
            transform_seconds += time.perf_counter() - t0

            encoder.stdin.write(output_frame)
            frame_count += 1

            if progress_every > 0 and frame_count % progress_every == 0:
                elapsed = time.perf_counter() - wall_start
                fps = frame_count / elapsed if elapsed else 0.0
                print(
                    f"\rframes={frame_count}  end-to-end={fps:.2f} fps",
                    end="",
                    flush=True,
                )

    except BrokenPipeError:
        pass
    finally:
        try:
            encoder.stdin.close()
        except Exception:
            pass

    decoder_rc = decoder.wait()
    encoder_rc = encoder.wait()

    decoder_err = (
        decoder.stderr.read().decode("utf-8", errors="replace")
        if decoder.stderr
        else ""
    )
    encoder_err = (
        encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.stderr
        else ""
    )

    if decoder_rc != 0:
        raise SystemExit(
            "FFmpeg decoder failed:\n" + decoder_err
        )

    if encoder_rc != 0:
        raise SystemExit(
            "FFmpeg encoder failed:\n" + encoder_err
        )

    wall_seconds = time.perf_counter() - wall_start
    transform_fps = (
        frame_count / transform_seconds
        if transform_seconds
        else 0.0
    )
    end_to_end_fps = (
        frame_count / wall_seconds
        if wall_seconds
        else 0.0
    )

    if progress_every > 0:
        print()

    print(f"mode             : {mode}")
    print(f"input            : {input_path}")
    print(f"output           : {output_path}")
    print(f"resolution       : {info.width}x{info.height}")
    print(f"source fps       : {info.fps:.3f}")
    print(f"frames processed : {frame_count}")
    print(f"transform time   : {transform_seconds:.3f} s")
    print(f"wall time        : {wall_seconds:.3f} s")
    print(f"transform speed  : {transform_fps:.2f} fps")
    print(f"end-to-end speed : {end_to_end_fps:.2f} fps")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MSPE-V4 video perceptual encryption/decryption. "
            "Processes video only; audio is intentionally dropped."
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
        default=-1,
        help=(
            "scipy.fft CPU workers. "
            "-1 uses all available CPUs. Default: -1"
        ),
    )
    parser.add_argument(
        "--codec",
        default="libx264",
        help="FFmpeg video encoder. Default: libx264",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Encoder CRF. Default: 18",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="Encoder preset. Default: veryfast",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help=(
            "FFmpeg encoder threads; 0 lets FFmpeg decide. Default: 0"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=30,
        help="Print progress every N frames; 0 disables. Default: 30",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_video(
        mode=args.mode,
        input_path=str(input_path),
        output_path=str(output_path),
        key=args.key,
        workers=args.workers,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        ffmpeg_threads=args.ffmpeg_threads,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
