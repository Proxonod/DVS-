"""Headless export module for the DVS viewer.

This script provides a command-line interface to render and export
event streams to video files without starting the GUI.  It uses the
same pipeline and filters as the interactive application but operates
entirely off‑screen.  Frames are rendered in memory and fed into an
OpenCV ``VideoWriter`` instance, mirroring the approach used by the
main GUI exporter so no external FFmpeg binary is required.

Example usage:

.. code-block:: console

    python -m app.export --input samples/recording.raw \
        --duration 5 --view-fps 60 \
        --out out/test_ffv1.mkv --codec ffv1

Supported codecs:

* **ffv1** - lossless Matroska using FFV1 level 3
* **mp4** - H.264-compatible MP4 using OpenCV's ``mp4v`` encoder

Note that the Metavision SDK must be installed to read RAW files.  If
the SDK is missing, this script will raise a ``RuntimeError``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from app.core.pipeline import Pipeline
from app.filters.registry import create_filter
from app.io.metavision_reader import (
    MetavisionReader,
    default_sample_path,
    is_metavision_raw,
)


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
        choices=["ffv1", "mp4"],
        help="Video codec to use",
    )
    return parser.parse_args(argv)


def _create_video_writer(out_path: str, codec: str, fps: float, frame_size: tuple[int, int]):
    """Initialise an OpenCV ``VideoWriter`` for the requested codec."""

    try:
        import cv2
    except Exception as exc:  # pragma: no cover - OpenCV optional in CI
        raise RuntimeError(
            "OpenCV mit VideoWriter-Unterstützung ist erforderlich, konnte jedoch nicht importiert werden."
        ) from exc

    lower = codec.lower()
    if lower == "ffv1":
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    elif lower in {"mp4", "x264-lossless"}:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif lower == "x265-lossless":
        # OpenCV builds rarely ship with HEVC; guide the user to supported codecs
        raise RuntimeError("HEVC/x265-Export wird nicht unterstützt. Bitte 'mp4' oder 'ffv1' verwenden.")
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(
            f"Videoausgabe konnte nicht initialisiert werden (Pfad oder Codec ungültig?): {out_path}"
        )
    return writer


def export_stream(reader: MetavisionReader, pipeline: Pipeline, duration_s: float, fps: float, out_path: str, codec: str) -> None:
    """Render a stream and write it to a video file via OpenCV.

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
        Codec string understood by :func:`_create_video_writer`.
    """
    width = reader.metadata.width
    height = reader.metadata.height
    frame_interval_us = int(1e6 / fps)
    max_time_us = int(duration_s * 1e6)
    frame_size = (int(width), int(height))
    writer = _create_video_writer(out_path, codec, fps, frame_size)
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
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                # OpenCV expects BGR order
                writer.write(frame[..., ::-1])
                accumulated_time = cutoff_time
                if accumulated_time >= max_time_us:
                    break
            if accumulated_time >= max_time_us:
                break
    finally:
        writer.release()


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
