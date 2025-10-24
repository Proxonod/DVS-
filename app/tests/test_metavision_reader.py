import numpy as np
import pytest

from app.io.metavision_reader import MetavisionReader, StreamMetadata


def test_ensure_sensor_size_recovers_dimensions_without_losing_events():
    events = np.zeros(5, dtype=[("x", np.int16), ("y", np.int16)])
    events["x"] = np.array([0, 1, 2, 3, 4])
    events["y"] = np.array([0, 1, 0, 2, 3])

    iterator = iter([events])
    reader = MetavisionReader(iterator, StreamMetadata(width=None, height=None))

    dims = reader.ensure_sensor_size()

    assert dims == (int(events["x"].max()) + 1, int(events["y"].max()) + 1)
    assert reader.metadata.width == dims[0]
    assert reader.metadata.height == dims[1]

    chunk = next(reader)
    # The chunk we receive must be the same object that ensure_sensor_size inspected
    assert chunk is events
    with pytest.raises(StopIteration):
        next(reader)


def test_ensure_sensor_size_noop_when_dimensions_known():
    events = np.zeros(2, dtype=[("x", np.int16), ("y", np.int16)])
    events["x"] = np.array([0, 0])
    events["y"] = np.array([1, 1])

    iterator = iter([events])
    reader = MetavisionReader(iterator, StreamMetadata(width=640, height=480))

    dims = reader.ensure_sensor_size()

    assert dims == (640, 480)
    chunk = next(reader)
    assert chunk is events
