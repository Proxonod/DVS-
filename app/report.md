# Prophesee DVS Desktop App – Research and Design

## Objective

This project aims to create a **plug‑and‑play desktop application** that can read recordings from Prophesee event cameras (DVS), stream live data from a connected camera and display the events in real time.  The app will provide a modular filter pipeline (noise suppression, temporal filters, artistic effects) with adjustable parameters, allow the user to scrub through recordings at different playback speeds and frame‑rate caps, customise event colours, and export the visible result to lossless or visually lossless video files.  The primary technologies are Python 3.10/3.11, the Metavision SDK, PySide6 for the GUI and FFmpeg for video export.

## Input sources

### Reading `.raw` files

Prophesee’s recordings are stored in `.raw` files.  The Metavision SDK documentation explains that these files consist of a **header** followed by **event data**.  The header is made up of ASCII lines beginning with `%` and contains key‑value pairs such as the recording date, sensor geometry and encoding format【814703913299290†L99-L123】.  After the header, the event stream is stored in binary using encoding formats such as EVT2 or EVT3【814703913299290†L208-L217】.  When opening a `.raw` file through the SDK, an index file may be generated (`file.raw.tmp_index`) to speed up seeks【814703913299290†L292-L304】.

The Python SDK offers two high‑level classes for reading event files: **`Camera`** (in `metavision_sdk_stream`) and **`EventsIterator`** (in `metavision_core`).  Opening a file using `EventsIterator` yields an iterator that returns **NumPy structured arrays**, each containing fields for timestamp (`t`), pixel coordinates (`x`, `y`) and polarity (`p`)【699379564843265†L292-L307】.  These arrays can be sliced and masked with standard NumPy operations, which makes them suitable for batch processing and filtering.  `EventsIterator` can stream events by a time window (`delta_t`) or by a fixed number of events, and defaults to 10 ms slices【699379564843265†L294-L302】.  Alternatively, `RawReader` can load a specified number of events or a fixed time slice from either a file or a live device【699379564843265†L324-L347】.

### Live camera capture and bias files

For live streaming, the SDK exposes the same event iterator classes by opening a camera instead of a file.  To adjust sensor noise characteristics and dynamic range, event cameras provide **bias parameters** (contrast thresholds, low‑pass/high‑pass filters, refractory period, etc.).  These can be tuned through the HAL API; for example, the Python method `device.get_i_ll_biases().set("bias_diff_on", value)` can set a single bias【569327891620415†L784-L790】.  Older `.bias` files are plain‑text lists of `value % bias_name` pairs; the documentation suggests converting them to camera settings JSON files using Metavision Studio or parsing them manually【569327891620415†L742-L756】.  Our application will detect `.bias` files in the sample folder, parse the values and apply them to the live device on startup.

## Event representation

Each event is represented by a tuple `(t, x, y, p)` where `t` is the timestamp (microseconds), `(x, y)` is the pixel coordinate on the sensor and `p` is the polarity (`0` for negative events, `1` for positive events).  The sensor geometry can be read from the RAW header (keywords `height` and `width`)【814703913299290†L124-L137】.  Events are inherently asynchronous and can be processed in small time slices (1–5 ms) for responsive rendering.

## Noise filtering and time‑domain representations

### Background Activity Filter (BAF)

Event sensors produce a background of spurious events (“background activity”).  A widely used method to remove isolated noise is the **Background Activity Filter (BAF)**.  It keeps an array `t_start[x,y]` that stores the timestamp of the last accepted event at each pixel.  For each new event `e_i = (t_i, x_i, y_i, p_i)` the algorithm checks the neighbourhood defined by a spatial radius (e.g. 1 pixel).  If the minimum timestamp in that neighbourhood is within a **time window** (`time_window`) of `t_i` then the event is accepted; otherwise it is discarded【321700183697423†L910-L970】.  The pseudocode from Ekonoja & Trogadas’s study shows the core logic: initialise `t_start` to `–time_window`, and for each event update the local neighbourhood; only events with recent neighbours pass through【321700183697423†L914-L969】.

### Pixel refractory filter

A second filter enforces a **refractory period** per pixel: events occurring within `refractory_us` of a previous event at the same pixel are suppressed.  This simple rule can be implemented using a 2‑D array `last_time[x,y]` storing the timestamp of the last accepted event per pixel.

### Time surface

A **time surface** is a continuous valued image where each pixel’s intensity encodes how recently an event occurred.  The Medium tutorial on event cameras describes a simple implementation: maintain an image `time_surface[y,x]` initialised to zero, iterate through events in a time window `[start_time, end_time]` and for each event assign

```python
time_surface[y, x] = (2*p - 1) * np.exp(-(end_time - t) / tau)
```

where `tau` is a decay constant【373245208989025†L389-L413】.  This representation emphasises recent events and allows easy rendering as a grey‑scale or colour image.  The time surface can be computed separately for positive and negative polarities when `polarity_separate=True`.

### Other filters

Additional filters can be built on top of the basic event stream:

* **Event rate map**: counts events in each pixel over a sliding window and optionally normalises by the maximum count; applying a Gaussian blur (`sigma`) can produce smooth rate maps.
* **Temporal median filter**: maintains the last `min_hits` events for each pixel within a window and selects the median timestamp or discards outliers.
* **Edge accumulation**: integrates events over a longer window and applies a high‑pass exponential decay (`e^{-dt/\alpha}`) to emphasise edges and motions.
* **Motion direction map**: divides the frame into cells (`cell_size`×`cell_size`), estimates event flow directions in each cell (e.g. via histograms of event displacement vectors) and displays directional colours or arrows.
* **Artistic effects**: motion trails (decay of intensities), glow/bloom (spatial blur of event intensities), posterisation/palette mapping (quantising intensity to discrete levels), halftone or CRT scanlines, temporal jitter/glitch (randomly perturbing event timestamps or dropping events), kaleidoscope or polar warp (coordinate transforms), Voronoi overlays, colour‑lookup tables, region‑of‑interest masks, hot‑pixel suppression and clustering (e.g. DBSCAN) for highlighting event clusters.

## Video export

To export the composited view to video, the app will stream the rendered frames into an FFmpeg process.  Two target formats will be provided:

* **Lossless FFV1 (Matroska)** – recommended for archival.  A typical FFmpeg command uses version 3 of FFV1 with range coding and slice CRC to ensure data integrity:

  ```bash
  ffmpeg -y -f rawvideo -pix_fmt rgb24 -s WxH -r <fps> -i - \
         -c:v ffv1 -level 3 -coder 1 -context 1 \
         -g 1 -slices 16 -slicecrc 1 \
         -pix_fmt rgb24 -f matroska output.mkv
  ```

  Level 3, coder 1 and context 1 correspond to the recommended FFV1 parameters【718784362810110†L112-L116】.

* **Visually lossless MP4** – uses H.264 or H.265 in lossless mode.  For H.264 the recommended settings are `-c:v libx264 -preset slow -crf 0 -pix_fmt yuv444p -movflags +faststart`; for H.265 use `-c:v libx265 -x265-params lossless=1 -pix_fmt yuv444p -movflags +faststart`.  These settings produce large files but maintain visual fidelity.

Still images can be exported as PNG files using the current colour mapping and filter stack.

## User interface and experience

The GUI will be built with **PySide6** and designed to be high‑DPI aware.  A top bar will provide buttons to open a file, start/stop the camera, play/pause, change playback speed (0.1×–8×), adjust the view frame rate cap (24, 30, 60, 120 fps or custom), toggle looping, take screenshots and start exports.  A side panel will list all active filters; users can enable/disable, reorder filters by dragging and adjust parameters live.  A timeline bar with scrubbing shows the current time, allows stepping ±5 ms and displays event rates as a sparkline.  Keyboard shortcuts (space = play/pause, J/K/L to halve/double speed, arrow keys to step, F for fullscreen, R to reset pipeline) offer quick control.

Rendering will use either **OpenGL** through PyOpenGL or a fast QImage path backed by NumPy.  Frames are double‑buffered to avoid tearing.  Separate threads will handle I/O (reading events), processing (filter pipeline) and UI rendering; communication occurs through queues and pre‑allocated buffers.  This design targets 60–120 fps display rates with low latency.

## Data model and presets

The internal event representation is a NumPy structured array with fields `x`, `y`, `t` (microseconds) and `p` (polarity).  Helper functions detect `.raw` and `.bias` files and auto‑open a sample recording.  The filter pipeline will be managed by a **Pipeline** class that maintains an ordered list of filter instances derived from the following base class:

```python
class BaseFilter:
    name: str
    def params(self) -> dict: ...
    def set_params(self, **kwargs) -> None: ...
    def reset(self) -> None: ...
    def process(self, events: np.ndarray, frame_state: dict) -> dict: ...
```

Each filter’s `process` method receives a slice of events and can update its internal state and output a dictionary containing processed events and optional frame overlays (e.g. rate maps).  Filters will be registered via a registry so the UI can instantiate them dynamically.

The application will support **presets**: JSON files listing the active filters, their parameters and the chosen polarity colours.  A global configuration file stores defaults for window sizes, playback speeds, view FPS caps and other limits.  When exporting a video, metadata including the input file path, filter stack and colours will be embedded as JSON metadata in the container.

## Testing and reliability

Unit tests (via `pytest`) will cover core filters such as the background activity filter, refractory logic and time surface decay to ensure they behave as expected under controlled synthetic inputs.  A smoke test will load a sample recording, toggle a few filters, export a short clip and verify that the number of exported frames matches the expected duration times the view FPS.  Structured logging and non‑blocking error reporting will make diagnostics easy.

## Conclusion

By combining the Metavision SDK for event streaming【699379564843265†L294-L307】, spatiotemporal filtering techniques such as the background activity filter【321700183697423†L914-L970】 and time surfaces【373245208989025†L389-L413】, and robust video export via FFmpeg【718784362810110†L112-L116】, the proposed desktop application will enable researchers and artists to visualise, manipulate and record neuromorphic data in a flexible, high‑performance environment.  The modular design allows new filters and visual effects to be added easily, while the preset system ensures reproducible workflows.  The next step is to translate these design decisions into code and build the full application.