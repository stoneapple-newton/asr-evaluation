"""Unit tests for eval.metrics.diar_metrics."""
import pytest

from eval.metrics.diar_metrics import (
    _fallback_der,
    der_score,
    pyannote_available,
    speaker_count_delta,
)


def _seg(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


class TestFallbackDER:
    def test_perfect_match(self):
        segs = [_seg(0, 5, "A"), _seg(5, 10, "B")]
        assert _fallback_der(segs, segs) == pytest.approx(0.0)

    def test_empty_ref(self):
        assert _fallback_der([], [_seg(0, 5, "A")]) == pytest.approx(0.0)

    def test_all_missed(self):
        ref = [_seg(0, 10, "A")]
        hyp: list = []
        assert _fallback_der(ref, hyp) == pytest.approx(1.0)

    def test_speaker_confusion(self):
        ref = [_seg(0, 10, "A")]
        hyp = [_seg(0, 10, "B")]  # same span, wrong speaker
        der = _fallback_der(ref, hyp)
        # All 10s overlap but wrong speaker → 10/10 = 1.0
        assert der == pytest.approx(1.0)

    def test_partial_coverage(self):
        ref = [_seg(0, 10, "A")]
        hyp = [_seg(0, 5, "A")]   # correct speaker, half coverage
        der = _fallback_der(ref, hyp)
        # 5s missed out of 10s total → 0.5
        assert der == pytest.approx(0.5)


class TestDerScore:
    def test_identical_segs(self):
        segs = [_seg(0, 5, "A"), _seg(5, 10, "B")]
        score = der_score(segs, segs)
        # pyannote or fallback: perfect should be 0
        assert score == pytest.approx(0.0, abs=0.05)

    def test_empty_ref_returns_zero(self):
        assert der_score([], [_seg(0, 5, "A")]) == pytest.approx(0.0)

    def test_returns_float(self):
        ref = [_seg(0, 10, "A")]
        hyp = [_seg(0, 10, "A")]
        assert isinstance(der_score(ref, hyp), float)


class TestSpeakerCountDelta:
    def test_equal(self):
        ref = [_seg(0, 1, "A"), _seg(1, 2, "B")]
        hyp = [_seg(0, 1, "X"), _seg(1, 2, "Y")]
        assert speaker_count_delta(ref, hyp) == 0

    def test_too_many(self):
        ref = [_seg(0, 1, "A")]
        hyp = [_seg(0, 1, "A"), _seg(1, 2, "B"), _seg(2, 3, "C")]
        assert speaker_count_delta(ref, hyp) == 2

    def test_too_few(self):
        ref = [_seg(0, 1, "A"), _seg(1, 2, "B"), _seg(2, 3, "C")]
        hyp = [_seg(0, 1, "A")]
        assert speaker_count_delta(ref, hyp) == -2


class TestPyannoteAvailable:
    def test_returns_bool(self):
        result = pyannote_available()
        assert isinstance(result, bool)
