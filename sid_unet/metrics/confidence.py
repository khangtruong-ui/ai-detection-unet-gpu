"""
Confidence Interval (CI) utilities for segmentation and classification evaluation metrics.

Supports:
- Exact two-sided Student's t confidence intervals using Chan's parallel online variance/Welford accumulation.
- Wilson score confidence intervals for binary classification accuracy proportions.
- Deterministic bootstrap confidence intervals for aggregate classification metrics (macro F1, AUROC).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _normal_z_quantile(prob: float) -> float:
    """Inverse standard normal CDF approximation (Acklam / Abramowitz & Stegun) as scipy-free fallback."""
    # Exact common values first
    if abs(prob - 0.975) < 1e-9:
        return 1.959963984540054
    if abs(prob - 0.95) < 1e-9:
        return 1.6448536269514722
    if abs(prob - 0.995) < 1e-9:
        return 2.5758293035489004
    # Bisection over erfc for arbitrary probability in (0.5, 1.0)
    lo, hi = 0.0, 10.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < prob:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def critical_t_value(df: int, confidence_level: float = 0.95) -> float:
    """Return two-sided critical value t_{alpha/2, df} (falling back to normal z when scipy is unavailable)."""
    alpha_half = 0.5 * (1.0 + float(confidence_level))
    if df <= 0:
        return _normal_z_quantile(alpha_half)
    try:
        from scipy.stats import t as student_t
        val = float(student_t.ppf(alpha_half, df=df))
        if math.isfinite(val):
            return val
    except Exception:
        pass
    return _normal_z_quantile(alpha_half)


def clamp_interval(
    value: float,
    bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> float:
    """Clamp scalar value to [low, high] when bounds are provided."""
    if bounds is None:
        return float(value)
    lo, hi = bounds
    return float(min(max(float(value), lo), hi))


def compute_mean_confidence_interval(
    mean: float,
    m2: float,
    count: int,
    confidence_level: float = 0.95,
    bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> Dict[str, float]:
    """
    Compute two-sided Student's t confidence interval from online Welford (mean, m2, count).

    Returns:
        Dict with keys: mean, low, high, margin, std, sem, n, confidence_level.
    """
    n = int(count)
    mean_clamped = clamp_interval(mean, bounds)
    if n < 2:
        return {
            "mean": mean_clamped,
            "low": mean_clamped,
            "high": mean_clamped,
            "margin": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "n": float(n),
            "confidence_level": float(confidence_level),
        }

    var = max(0.0, float(m2) / float(n - 1))
    std = math.sqrt(var)
    sem = std / math.sqrt(float(n))
    t_crit = critical_t_value(df=n - 1, confidence_level=confidence_level)
    margin = t_crit * sem
    low = clamp_interval(mean - margin, bounds)
    high = clamp_interval(mean + margin, bounds)
    return {
        "mean": mean_clamped,
        "low": low,
        "high": high,
        "margin": float(margin),
        "std": float(std),
        "sem": float(sem),
        "n": float(n),
        "confidence_level": float(confidence_level),
    }


def compute_wilson_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Compute Wilson score confidence interval for a binomial proportion."""
    n = int(total)
    if n <= 0:
        return {
            "mean": 0.0,
            "low": 0.0,
            "high": 0.0,
            "margin": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "n": 0.0,
            "confidence_level": float(confidence_level),
        }

    p = float(successes) / float(n)
    z = _normal_z_quantile(0.5 * (1.0 + float(confidence_level)))
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p + z2 / (2.0 * float(n))) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * float(n))) / float(n))) / denom
    low = clamp_interval(center - half, (0.0, 1.0))
    high = clamp_interval(center + half, (0.0, 1.0))
    std = math.sqrt(max(0.0, p * (1.0 - p)))
    return {
        "mean": float(p),
        "low": low,
        "high": high,
        "margin": float(0.5 * (high - low)),
        "std": float(std),
        "sem": float(std / math.sqrt(float(n))),
        "n": float(n),
        "confidence_level": float(confidence_level),
    }


def compute_bootstrap_confidence_interval(
    point_estimate: float,
    metric_fn: Callable[[np.ndarray], Optional[float]],
    n_samples: int,
    confidence_level: float = 0.95,
    n_bootstrap: int = 300,
    seed: int = 42,
    bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> Dict[str, float]:
    """Compute deterministic percentile bootstrap CI for aggregate sample-level metrics."""
    n = int(n_samples)
    est = clamp_interval(point_estimate, bounds)
    if n < 2 or n_bootstrap <= 0:
        return {
            "mean": est,
            "low": est,
            "high": est,
            "margin": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "n": float(n),
            "confidence_level": float(confidence_level),
        }

    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        v = metric_fn(idx)
        if v is not None and math.isfinite(float(v)):
            vals.append(float(v))

    if len(vals) < 2:
        return {
            "mean": est,
            "low": est,
            "high": est,
            "margin": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "n": float(n),
            "confidence_level": float(confidence_level),
        }

    arr = np.asarray(vals, dtype=np.float64)
    alpha = 1.0 - float(confidence_level)
    low_q = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
    high_q = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    low = clamp_interval(low_q, bounds)
    high = clamp_interval(high_q, bounds)
    std = float(np.std(arr, ddof=1))
    return {
        "mean": est,
        "low": low,
        "high": high,
        "margin": float(0.5 * (high - low)),
        "std": std,
        "sem": std,
        "n": float(n),
        "confidence_level": float(confidence_level),
    }


class OnlineVarianceTracker:
    """Tracks count, running sum, mean, and Welford/Chan M2 for multiple named metrics."""

    def __init__(self, metric_names: Sequence[str]):
        self.metric_names = list(metric_names)
        self.count: int = 0
        self.sums: Dict[str, float] = {k: 0.0 for k in self.metric_names}
        self.means: Dict[str, float] = {k: 0.0 for k in self.metric_names}
        self.m2s: Dict[str, float] = {k: 0.0 for k in self.metric_names}

    def merge_batch(
        self,
        batch_count: int,
        batch_sums: Dict[str, float],
        batch_m2s: Dict[str, float],
    ) -> None:
        """Merge batch sufficient statistics into global state using Chan's parallel formula."""
        nb = int(batch_count)
        if nb <= 0:
            return
        na = self.count
        new_count = na + nb
        for k in self.metric_names:
            b_sum = float(batch_sums.get(k, 0.0))
            b_m2 = float(batch_m2s.get(k, 0.0))
            b_mean = b_sum / float(nb)
            self.sums[k] += b_sum
            if na == 0:
                self.means[k] = b_mean
                self.m2s[k] = b_m2
            else:
                delta = b_mean - self.means[k]
                self.means[k] = self.sums[k] / float(new_count)
                self.m2s[k] += b_m2 + (delta * delta) * (float(na) * float(nb) / float(new_count))
        self.count = new_count

    def update_single(self, values: Dict[str, float]) -> None:
        """Update with a single sample observation via Welford's step."""
        self.count += 1
        inv_n = 1.0 / float(self.count)
        for k in self.metric_names:
            x = float(values.get(k, 0.0))
            self.sums[k] += x
            delta = x - self.means[k]
            self.means[k] += delta * inv_n
            delta2 = x - self.means[k]
            self.m2s[k] += delta * delta2

    def ci_for(
        self,
        metric_name: str,
        confidence_level: float = 0.95,
        bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
    ) -> Dict[str, float]:
        """Compute Student's t CI dictionary for one tracked metric."""
        mean = self.sums[metric_name] / float(self.count) if self.count > 0 else 0.0
        return compute_mean_confidence_interval(
            mean=mean,
            m2=self.m2s[metric_name],
            count=self.count,
            confidence_level=confidence_level,
            bounds=bounds,
        )


def attach_ci_fields(
    target_dict: Dict[str, Any],
    ci_map: Dict[str, Dict[str, float]],
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Populate flat CI keys (e.g., iou_ci_low, iou_ci_high, iou_ci95_low, iou_ci95_high, ...)
    and structured confidence_intervals dict onto target_dict cleanly.
    """
    pct = int(round(float(confidence_level) * 100.0))
    target_dict["confidence_level"] = float(confidence_level)
    structured: Dict[str, Dict[str, float]] = dict(target_dict.get("confidence_intervals") or {})

    for metric_key, ci in ci_map.items():
        structured[metric_key] = ci
        target_dict[f"{metric_key}_ci_low"] = float(ci["low"])
        target_dict[f"{metric_key}_ci_high"] = float(ci["high"])
        target_dict[f"{metric_key}_ci_margin"] = float(ci["margin"])
        target_dict[f"{metric_key}_std"] = float(ci["std"])
        target_dict[f"{metric_key}_ci{pct}_low"] = float(ci["low"])
        target_dict[f"{metric_key}_ci{pct}_high"] = float(ci["high"])
        target_dict[f"{metric_key}_ci{pct}_margin"] = float(ci["margin"])

    target_dict["confidence_intervals"] = structured
    return target_dict
