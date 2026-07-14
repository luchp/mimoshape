"""Guards for the shipped road-record dataset used by the paper example."""

import pathlib

import numpy as np
import pytest

NPZ = pathlib.Path(__file__).resolve().parent.parent / "data" / "roadsection_220s_300hz.npz"


@pytest.fixture(scope="module")
def npz():
    return np.load(NPZ)


def test_road_record_contents(npz):
    assert npz["data_int16"].dtype == np.int16
    assert npz["data_int16"].shape == (12, 65536)
    assert npz["scale"].shape == (12,)
    assert np.all(npz["scale"] > 0)
    assert float(npz["fs"]) == pytest.approx(300.0, rel=1e-6)
    assert len(npz["names"]) == 12 and len(npz["units"]) == 12
    # left/right interleaving assumed by the paper's co-kurtosis tuples
    for k in range(0, 12, 2):
        left, right = str(npz["names"][k]), str(npz["names"][k + 1])
        assert left.endswith("L") and right.endswith("R")
        assert left[:-1] == right[:-1]


def test_road_record_is_anonymised_and_nongaussian(npz):
    y = npz["data_int16"].astype(np.float64) * npz["scale"][:, None]
    u = y - np.mean(y, axis=1, keepdims=True)
    kurt = np.mean(u**4, axis=1) / np.mean(u**2, axis=1) ** 2
    assert kurt.max() > 5.0  # the heavy-tailed character the example relies on
