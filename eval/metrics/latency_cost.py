"""Real-time factor (RTF) and cost estimation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field


def rtf(processing_sec: float, audio_sec: float) -> float:
    """Real-time factor = processing_time / audio_duration. <1.0 is faster than real time."""
    return processing_sec / max(1e-9, audio_sec)


def gpu_cost(processing_sec: float, gpu_hourly_rate_usd: float) -> float:
    """Compute GPU cost in USD from elapsed seconds and an hourly rate."""
    return (processing_sec / 3600.0) * gpu_hourly_rate_usd


def llm_cost(
    tokens_in: int,
    tokens_out: int,
    price_per_1k_in: float,
    price_per_1k_out: float,
) -> float:
    """Compute LLM cost in USD given token counts and per-1K prices."""
    return (tokens_in / 1000.0) * price_per_1k_in + (tokens_out / 1000.0) * price_per_1k_out


def total_pipeline_cost(
    processing_sec: float,
    gpu_hourly_rate_usd: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    price_per_1k_in: float = 0.0,
    price_per_1k_out: float = 0.0,
) -> dict:
    """Return a breakdown of GPU + LLM costs for one audio sample."""
    gpu = gpu_cost(processing_sec, gpu_hourly_rate_usd)
    llm = llm_cost(tokens_in, tokens_out, price_per_1k_in, price_per_1k_out)
    return {
        "gpu_cost_usd": gpu,
        "llm_cost_usd": llm,
        "total_cost_usd": gpu + llm,
    }


@dataclass
class CostConfig:
    """Pricing assumptions stored per-run for reproducible cost tracking."""

    gpu_hourly_rate_usd: float = 0.0
    llm_price_per_1k_in: float = 0.0
    llm_price_per_1k_out: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "gpu_hourly_rate_usd": self.gpu_hourly_rate_usd,
            "llm_price_per_1k_in": self.llm_price_per_1k_in,
            "llm_price_per_1k_out": self.llm_price_per_1k_out,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CostConfig":
        return cls(
            gpu_hourly_rate_usd=float(d.get("gpu_hourly_rate_usd", 0.0)),
            llm_price_per_1k_in=float(d.get("llm_price_per_1k_in", 0.0)),
            llm_price_per_1k_out=float(d.get("llm_price_per_1k_out", 0.0)),
            notes=str(d.get("notes", "")),
        )


def summarize_latency(elapsed_secs: list[float], audio_secs: list[float]) -> dict:
    """Aggregate RTF stats across samples."""
    if not elapsed_secs or not audio_secs:
        return {"mean_rtf": None, "max_rtf": None, "total_audio_sec": None}
    rtfs = [rtf(e, a) for e, a in zip(elapsed_secs, audio_secs)]
    return {
        "mean_rtf": sum(rtfs) / len(rtfs),
        "max_rtf": max(rtfs),
        "total_audio_sec": sum(audio_secs),
        "total_processing_sec": sum(elapsed_secs),
    }
