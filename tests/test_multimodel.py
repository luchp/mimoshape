"""Multimodel tests: section models, merge strategies, end-to-end synthesis."""

import numpy as np
import pytest

from mimoshape import estimate, moments, multimodel


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def record(rng):
    """Two correlated channels whose variance doubles halfway (non-stationary)."""
    n = 4096
    base = rng.standard_normal(n)
    y0 = base + 0.3 * rng.standard_normal(n)
    y1 = 0.7 * base + 0.5 * rng.standard_normal(n)
    y = np.vstack([y0, y1])
    y[:, n // 2 :] *= 2.0
    return y


def test_moment_tuples_default_counts():
    tuples = multimodel.moment_tuples(3)
    assert tuples.count((0, 0, 0)) == 1 and (2, 2, 2, 2) in tuples
    assert (0, 0, 1, 1) in tuples and (1, 1, 2, 2) in tuples
    assert len(tuples) == 3 + 3 + 3  # skew + kurt + pair co-kurt


def test_moment_tuples_coskewness_only():
    tuples = multimodel.moment_tuples(2, skewness=False, kurtosis=False, cokurtosis=False, coskewness=True)
    assert tuples == [(0, 0, 1), (0, 1, 1)]


def test_best_shift_recovers_position(rng):
    block = rng.standard_normal((2, 256))
    tail = block[:, 40:72]
    assert multimodel.best_shift(tail, block) == 40


def test_merge_crossfade_length_and_untouched_head(rng):
    blocks = [rng.standard_normal((2, 256)) for _ in range(3)]
    fade = 32
    out = multimodel.merge_crossfade(blocks, fade)
    assert out.shape == (2, 3 * 256 - 2 * fade)
    np.testing.assert_array_equal(out[:, : 256 - fade], blocks[0][:, : 256 - fade])


def test_estimate_section_models(record):
    tuples = [(0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 1, 1)]
    models = multimodel.estimate_section_models(record, 4, tuples)
    assert len(models) == 4
    for s, model in enumerate(models):
        assert model.H.shape == (2, 2, 513)  # default nfft = section length 1024
        section = record[:, s * 1024 : (s + 1) * 1024]
        np.testing.assert_allclose(model.variance, np.var(section, axis=1), rtol=1e-12)
        np.testing.assert_allclose(estimate.synthesis_variance(model.H), model.variance, rtol=1e-12)
        assert [t.indices for t in model.targets] == tuples
    # the variance step is visible across sections
    assert np.all(models[3].variance > 3 * models[0].variance)


def test_estimate_section_models_nfft_rounds_down(record):
    models = multimodel.estimate_section_models(record[:, :3000], 2, [], nfft=None)
    assert models[0].H.shape == (2, 2, 1024 // 2 + 1)  # 1500 -> 1024


def test_estimate_section_models_errors(record):
    with pytest.raises(ValueError, match="shorter than nfft"):
        multimodel.estimate_section_models(record, 4, [], nfft=2048)
    bad = record.copy()
    bad[1, :1024] = 5.0
    with pytest.raises(ValueError, match="constant channel"):
        multimodel.estimate_section_models(bad, 4, [])


def test_synthesize_multimodel_rejects_bad_merge():
    with pytest.raises(ValueError, match="merge must be one of"):
        multimodel.synthesize_multimodel([], merge="overlap")


def test_synthesize_multimodel_crossfade_tracks_sections(record, rng):
    tuples = [(0, 0, 0, 0), (1, 1, 1, 1)]
    models = multimodel.estimate_section_models(record, 2, tuples, nfft=512)
    done = []
    result = multimodel.synthesize_multimodel(
        models, merge="crossfade", max_time=10.0, rng=rng,
        progress=lambda d, t: done.append((d, t)),
    )
    assert len(result.blocks) == 2
    fade = 512 // 16  # block length is nfft, not the section length
    assert result.merged.shape == (2, 2 * 512 - fade)
    assert done == [(1, 2), (2, 2)]
    # each block reproduces its own section: variance step and moments
    for block, model in zip(result.blocks, result.models):
        np.testing.assert_allclose(np.var(block, axis=1), model.variance, rtol=0.15)
    for ach, model in zip(result.achieved, result.models):
        for t in model.targets:
            assert ach[t.indices] == pytest.approx(t.value, rel=0.1)


def test_synthesize_multimodel_c1_and_zero_joints(record, rng):
    models = multimodel.estimate_section_models(record, 2, [], nfft=512)
    std = np.sqrt(models[0].variance)

    result = multimodel.synthesize_multimodel(
        models, merge="c1", endpoint_weight=1e4, max_time=10.0, rng=rng
    )
    assert result.merged.shape == (2, 2 * 512)
    head, slope = multimodel.head_state(result.blocks[0])
    next_head, next_slope = multimodel.head_state(result.blocks[1])
    np.testing.assert_allclose(next_head, head, atol=0.05 * std.max())
    np.testing.assert_allclose(next_slope, slope, atol=0.05 * std.max())

    result = multimodel.synthesize_multimodel(
        models, merge="zero", endpoint_weight=1e4, max_time=10.0, rng=rng
    )
    for block in result.blocks:
        h, s = multimodel.head_state(block)
        np.testing.assert_allclose(h, 0.0, atol=0.05 * std.max())
        np.testing.assert_allclose(s, 0.0, atol=0.05 * std.max())
