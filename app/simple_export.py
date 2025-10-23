"""Minimal command-line export tool for DVS recordings.

This script prompts the user for an input RAW file (if not supplied via
arguments), allows optional custom colours for positive/negative events
and writes the rendered result to either an FFV1 Matroska file or an
MP4 (H.264) file.  It reuses the existing export pipeline so the
rendering matches the main application while avoiding any GUI
components.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from app.core.pipeline import Pipeline, hex_to_rgb
from app.export import export_stream
from app.filters.registry import create_filter
from app.io.metavision_reader import MetavisionReader, is_metavision_raw


def _parse_colour(value: str | None, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return an RGB tuple for ``value`` or ``fallback`` if ``None``.

    Accepted formats are ``"#RRGGBB"`` or ``"R,G,B"``.
    """

    if value is None:
        return fallback
    value = value.strip()
    if not value:
        return fallback
    if value.startswith("#"):
        return hex_to_rgb(value)
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Invalid colour specification: {value}")
    rgb = tuple(int(p) for p in parts)
    if any(c < 0 or c > 255 for c in rgb):
        raise ValueError("Colour components must be between 0 and 255")
    return rgb  # type: ignore[return-value]


def _build_pipeline(width: int, height: int) -> Pipeline:
    pipeline = Pipeline()
    for name in ("BackgroundActivityFilter", "RefractoryFilter", "TimeSurface"):
        try:
            filt = create_filter(name)
        except Exception:
            continue
        pipeline.add_filter(filt)
    pipeline.reset(width, height)
    return pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple DVS export tool")
    parser.add_argument("--input", type=str, help="Path to a Metavision RAW file")
    parser.add_argument("--out", type=str, required=True, help="Destination video file")
    parser.add_argument(
        "--codec",
        type=str,
        choices=["ffv1", "mp4"],
        default="ffv1",
        help="Output format (Matroska/FFV1 or MP4/H.264)",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--fps", type=float, default=60.0, help="Output frame rate")
    parser.add_argument("--pos-colour", type=str, help="Positive event colour (#RRGGBB or R,G,B)")
    parser.add_argument("--neg-colour", type=str, help="Negative event colour (#RRGGBB or R,G,B)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input
    if not input_path:
        input_path = input("Pfad zur RAW-Datei: ").strip()
    if not input_path:
        raise SystemExit("Es wurde keine Eingabedatei angegeben.")

    raw_path = Path(input_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Input file not found: {raw_path}")
    if not is_metavision_raw(str(raw_path)):
        raise ValueError(f"Unsupported input file: {raw_path}")

    reader = MetavisionReader.from_raw(str(raw_path))
    pipeline = _build_pipeline(reader.metadata.width, reader.metadata.height)

    # Apply custom colours if provided
    pipeline.pos_colour = _parse_colour(args.pos_colour, pipeline.pos_colour)
    pipeline.neg_colour = _parse_colour(args.neg_colour, pipeline.neg_colour)

    codec = args.codec
    export_stream(reader, pipeline, args.duration, args.fps, args.out, codec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
