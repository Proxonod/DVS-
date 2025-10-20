# DVS C++ Pipeline

This directory contains a high-performance C++ reimplementation of the
Python DVS processing pipeline.  It provides a small processing library,
a command-line utility and unit tests covering the core behaviour of the
filters.

## Features

- Zero-copy event buffers using `std::vector`.
- SIMD-friendly loops and explicit saturation arithmetic for rendering.
- Background activity suppression, per-pixel refractory period and time
  surface filters.
- Pipeline compositor producing RGB frames.
- CSV-based CLI that writes PPM images for rapid inspection.

## Building

```
mkdir build && cd build
cmake ..
cmake --build .
```

To run the tests:

```
ctest
```

## Command-line usage

```
./dvs_cli <width> <height> <events.csv> <output.ppm>
```

The CSV file must contain rows of `x,y,t,p` values where `t` is the
timestamp in microseconds.  The output frame is written as a binary PPM
image that can be viewed with standard image viewers.
