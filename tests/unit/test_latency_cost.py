"""Unit tests for eval.metrics.latency_cost."""
import pytest

from eval.metrics.latency_cost import (
    CostConfig,
    gpu_cost,
    llm_cost,
    rtf,
    summarize_latency,
    total_pipeline_cost,
)


class TestRTF:
    def test_real_time(self):
        assert rtf(10.0, 10.0) == pytest.approx(1.0)

    def test_faster_than_real_time(self):
        assert rtf(5.0, 10.0) == pytest.approx(0.5)

    def test_zero_audio_no_crash(self):
        result = rtf(5.0, 0.0)
        assert result > 0  # divides by epsilon, not zero

    def test_zero_processing(self):
        assert rtf(0.0, 10.0) == pytest.approx(0.0)


class TestGPUCost:
    def test_one_hour(self):
        assert gpu_cost(3600.0, 2.0) == pytest.approx(2.0)

    def test_half_hour(self):
        assert gpu_cost(1800.0, 4.0) == pytest.approx(2.0)

    def test_zero_rate(self):
        assert gpu_cost(3600.0, 0.0) == pytest.approx(0.0)


class TestLLMCost:
    def test_basic(self):
        cost = llm_cost(1000, 500, 0.003, 0.015)
        assert cost == pytest.approx(0.003 + 0.0075)

    def test_zero_tokens(self):
        assert llm_cost(0, 0, 0.003, 0.015) == pytest.approx(0.0)


class TestTotalPipelineCost:
    def test_gpu_only(self):
        result = total_pipeline_cost(3600.0, 1.0)
        assert result["gpu_cost_usd"] == pytest.approx(1.0)
        assert result["llm_cost_usd"] == pytest.approx(0.0)
        assert result["total_cost_usd"] == pytest.approx(1.0)

    def test_both_components(self):
        result = total_pipeline_cost(
            3600.0, 1.0,
            tokens_in=1000, tokens_out=500,
            price_per_1k_in=0.003, price_per_1k_out=0.015,
        )
        assert result["total_cost_usd"] == pytest.approx(1.0 + 0.003 + 0.0075)


class TestCostConfig:
    def test_roundtrip(self):
        cfg = CostConfig(gpu_hourly_rate_usd=2.5, llm_price_per_1k_in=0.003, notes="test")
        d = cfg.to_dict()
        cfg2 = CostConfig.from_dict(d)
        assert cfg2.gpu_hourly_rate_usd == pytest.approx(2.5)
        assert cfg2.notes == "test"

    def test_defaults(self):
        cfg = CostConfig()
        assert cfg.gpu_hourly_rate_usd == pytest.approx(0.0)


class TestSummarizeLatency:
    def test_empty(self):
        result = summarize_latency([], [])
        assert result["mean_rtf"] is None

    def test_basic(self):
        result = summarize_latency([5.0, 10.0], [10.0, 10.0])
        assert result["mean_rtf"] == pytest.approx(0.75)
        assert result["max_rtf"] == pytest.approx(1.0)
        assert result["total_audio_sec"] == pytest.approx(20.0)
