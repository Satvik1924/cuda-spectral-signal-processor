#!/usr/bin/env python3
import argparse
import math
import os
import random
import struct
import zlib


def write_png_rgb(path: str, width: int, height: int, rgb_bytes: bytes) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    row_stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb_bytes[y * row_stride : (y + 1) * row_stride])

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


def generate_signals(output_dir: str, count: int, num_samples: int, sample_rate: float) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(2026)

    for idx in range(count):
        f0 = 220.0 + (idx % 16) * 25.0       # Fundamental instrument pitch (220-595 Hz)
        f_harm = f0 * 2.0                    # 2nd harmonic
        f_jammer = 6200.0 + (idx % 5) * 150.0  # High-frequency interference tone
        noise_std = 0.35

        path = os.path.join(output_dir, f"signal_{idx:04d}.csv")
        with open(path, "w") as f:
            for n in range(num_samples):
                t = n / sample_rate
                clean = (
                    0.80 * math.sin(2.0 * math.pi * f0 * t)
                    + 0.40 * math.sin(2.0 * math.pi * f_harm * t)
                )
                jammer = 0.55 * math.sin(2.0 * math.pi * f_jammer * t)
                val = clean + jammer + rng.gauss(0.0, noise_std)
                f.write(f"{val:.6f}\n")


def load_csv_1d(path: str):
    vals = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                vals.append(float(s))
    return vals


def render_waveform_comparison(before_vals, after_vals, out_png: str) -> None:
    width, height = 960, 480
    canvas = bytearray([255] * (width * height * 3))

    def set_px(x: int, y: int, r: int, g: int, b: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            off = (y * width + x) * 3
            canvas[off] = r
            canvas[off + 1] = g
            canvas[off + 2] = b

    def draw_line(x0: int, y0: int, x1: int, y1: int, r: int, g: int, b: int) -> None:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            set_px(x0, y0, r, g, b)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # Draw divider and center zero-axes
    for x in range(width):
        set_px(x, height // 2, 180, 180, 180)
        set_px(x, height // 4, 230, 230, 230)
        set_px(x, (3 * height) // 4, 230, 230, 230)

    window_len = min(400, len(before_vals), len(after_vals))
    # Top panel: Noisy Input Signal (red)
    prev_x, prev_y = 0, height // 4
    for i in range(window_len):
        x = int((i / (window_len - 1)) * (width - 1))
        y = int((height // 4) - max(-1.8, min(1.8, before_vals[i])) * 55)
        if i > 0:
            draw_line(prev_x, prev_y, x, y, 210, 45, 45)
        prev_x, prev_y = x, y

    # Bottom panel: GPU Filtered Output Signal (blue)
    prev_x, prev_y = 0, (3 * height) // 4
    for i in range(window_len):
        x = int((i / (window_len - 1)) * (width - 1))
        y = int(((3 * height) // 4) - max(-1.8, min(1.8, after_vals[i])) * 55)
        if i > 0:
            draw_line(prev_x, prev_y, x, y, 25, 95, 210)
        prev_x, prev_y = x, y

    write_png_rgb(out_png, width, height, bytes(canvas))


def render_spectrogram_png(spec_csv: str, out_png: str) -> None:
    rows = []
    max_val = 1e-6
    with open(spec_csv, "r") as f:
        for line in f:
            vals = [float(v) for v in line.strip().split(",") if v]
            if vals:
                rows.append(vals)
                max_val = max(max_val, max(vals))

    num_frames = len(rows)
    num_bins = len(rows[0])
    scale_x = 16
    width = num_frames * scale_x
    height = num_bins
    canvas = bytearray(width * height * 3)

    for f_idx in range(num_frames):
        for b_idx in range(num_bins):
            norm = min(1.0, rows[f_idx][b_idx] / max_val)
            # Viridis-inspired color ramp
            r = int(255 * min(1.0, max(0.0, 1.5 * norm - 0.3)))
            g = int(255 * math.sin(norm * math.pi))
            b = int(255 * (1.0 - norm * 0.85))
            y = height - 1 - b_idx
            for dx in range(scale_x):
                off = (y * width + f_idx * scale_x + dx) * 3
                canvas[off] = r
                canvas[off + 1] = g
                canvas[off + 2] = b

    write_png_rgb(out_png, width, height, bytes(canvas))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "visualize"], required=True)
    parser.add_argument("--input_dir", default="input_data")
    parser.add_argument("--output_dir", default="output_data")
    parser.add_argument("--artifact_dir", default="artifacts")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--sample_rate", type=float, default=16000.0)
    args = parser.parse_args()

    if args.mode == "generate":
        generate_signals(args.input_dir, args.count, args.samples, args.sample_rate)
        print(f"Generated {args.count} noisy CSV signals ({args.samples} samples each) in '{args.input_dir}/'.")
    elif args.mode == "visualize":
        os.makedirs(args.artifact_dir, exist_ok=True)
        before = load_csv_1d(os.path.join(args.input_dir, "signal_0000.csv"))
        after = load_csv_1d(os.path.join(args.output_dir, "signal_0000.csv"))
        render_waveform_comparison(
            before, after, os.path.join(args.artifact_dir, "signal_0000_waveform_before_after.png")
        )
        render_spectrogram_png(
            os.path.join(args.output_dir, "signal_0000_spectrogram.csv"),
            os.path.join(args.artifact_dir, "signal_0000_spectrogram.png"),
        )
        print(f"Rendered before/after waveform and spectrogram PNGs in '{args.artifact_dir}/'.")
