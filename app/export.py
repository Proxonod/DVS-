"""Headless export module for the DVS viewer.

This script provides a command-line interface to render and export
event streams to video files without starting the GUI.  It uses the
same pipeline and filters as the interactive application but operates
entirely off‑screen.  The rendered frames are streamed to an FFmpeg
process with parameters chosen according to the requested codec.

Example usage:

.. code-block:: console

    python -m app.export --input samples/recording.raw \
        --duration 5 --view-fps 60 \
        --out out/test_ffv1.mkv --codec ffv1

Supported codecs:

* **ffv1** - lossless Matroska using FFV1 level 3
* **x264-lossless** - visually lossless H.264
* **x265-lossless** - visually lossless H.265

Note that the Metavision SDK must be installed to read RAW files.  If
the SDK is missing, this script will raise a ``RuntimeError``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .core.pipeline import Pipeline
from .filters.registry import create_filter
from .io.metavision_reader import MetavisionReader, default_sample_path, is_metavision_raw


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DVS recordings to video")
    parser.add_argument(
        "--input",
        type=str,
        default=default_sample_path(),
        help="Path to a RAW recording file",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration of the export in seconds",
    )
    parser.add_argument(
        "--view-fps",
        type=float,
        default=60.0,
        help="Frame rate of the output video",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output file path (e.g. out.mkv)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="ffv1",
        choices=["ffv1", "x264-lossless", "x265-lossless"],
        help="Video codec to use",
    )
    return parser.parse_args(argv)


def build_ffmpeg_command(width: int, height: int, fps: float, out_path: str, codec: str) -> list[str]:
    """Construct the FFmpeg command for the given parameters.

    Parameters
    ----------
    width, height:
        Frame dimensions.
    fps:
        Frame rate.
    out_path:
        Output file path.
    codec:
        One of ``"ffv1"``, ``"x264-lossless"`` or ``"x265-lossless"``.

    Returns
    -------
    list[str]
        The command line arguments for invoking ``ffmpeg``.
    """
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output without asking
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",  # read video from stdin
    ]
    if codec == "ffv1":
        cmd += [
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-coder",
            "1",
            "-context",
            "1",
            "-g",
            "1",
            "-slices",
            "16",
            "-slicecrc",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "matroska",
            out_path,
        ]
    elif codec == "x264-lossless":
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            "-movflags",
            "+faststart",
            out_path,
        ]
    elif codec == "x265-lossless":
        cmd += [
            "-c:v",
            "libx265",
            "-x265-params",
            "lossless=1",
            "-pix_fmt",
            "yuv444p",
            "-movflags",
            "+faststart",
            out_path,
        ]
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    return cmd


def export_stream(reader: MetavisionReader, pipeline: Pipeline, duration_s: float, fps: float, out_path: str, codec: str) -> None:
    """Render a stream and write it to a video file via FFmpeg.

    Parameters
    ----------
    reader:
        An open ``MetavisionReader`` providing event slices.
    pipeline:
        A configured pipeline with filters and colour mapping.
    duration_s:
        Desired duration of the output in seconds.
    fps:
        Frames per second for the output video.
    out_path:
        Destination file path.
    codec:
        Codec string; passed to :func:`build_ffmpeg_command`.
    """
    width = reader.metadata.width
    height = reader.metadata.height
    frame_interval_us = int(1e6 / fps)
    max_time_us = int(duration_s * 1e6)
    ffmpeg_cmd = build_ffmpeg_command(width, height, fps, out_path, codec)
    # Start FFmpeg process
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Event accumulation and rendering loop
    current_buffer: list[np.ndarray] = []
    accumulated_time = 0
    last_frame_time = 0
    try:
        for events in reader:
            if events.size == 0:
                continue
            # Append to buffer
            current_buffer.append(events)
            # Update accumulated time based on event timestamps
            # Use the last event's timestamp in this slice
            slice_end_time = int(events["t"][-1])
            if last_frame_time == 0:
                last_frame_time = slice_end_time
            # Process frames until we exceed frame_interval
            while slice_end_time - last_frame_time >= frame_interval_us:
                # Concatenate events up to current time
                all_events = (
                    np.concatenate(current_buffer) if len(current_buffer) > 1 else current_buffer[0]
                )
                # Filter events that occurred before last_frame_time + frame_interval
                cutoff_time = last_frame_time + frame_interval_us
                frame_mask = all_events["t"] < cutoff_time
                frame_events = all_events[frame_mask]
                # Keep remaining events for next frame
                remaining = all_events[~frame_mask]
                current_buffer = [remaining] if remaining.size > 0 else []
                last_frame_time = cutoff_time
                # Run through the pipeline
                state = pipeline.process_events(frame_events)
                frame = pipeline.get_frame(state)
                # Write raw frame data to ffmpeg
                proc.stdin.write(frame.tobytes())  # type: ignore[union-attr]
                accumulated_time = cutoff_time
                if accumulated_time >= max_time_us:
                    break
            if accumulated_time >= max_time_us:
                break
    finally:
        # Flush and close FFmpeg
        try:
            if proc.stdin:
                proc.stdin.close()
        finally:
            proc.wait()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    # Validate input file
    input_path = args.input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not is_metavision_raw(input_path):
        raise ValueError(f"Unsupported input file: {input_path}")
    # Create reader
    reader = MetavisionReader.from_raw(input_path)
    # Set up pipeline with a couple of basic filters
    pipeline = Pipeline()
    # Add a background activity filter and a time surface for demonstration
    try:
        baf = create_filter("BackgroundActivityFilter")
        pipeline.add_filter(baf)
    except Exception:
        pass
    try:
        refractory = create_filter("RefractoryFilter")
        pipeline.add_filter(refractory)
    except Exception:
        pass
    try:
        ts = create_filter("TimeSurface")
        pipeline.add_filter(ts)
    except Exception:
        pass
    pipeline.reset(reader.metadata.width, reader.metadata.height)
    # Perform export
    export_stream(reader, pipeline, args.duration, args.view_fps, args.out, args.codec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
