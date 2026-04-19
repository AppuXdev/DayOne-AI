from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "query_log.jsonl"


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


def _extract_values(records: Iterable[dict], key: str) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _format_summary(label: str, values: List[float]) -> str:
    if not values:
        return f"{label}: no data"
    values = sorted(values)
    p50 = median(values)
    p95 = _percentile(values, 95)
    p99 = _percentile(values, 99)
    return (
        f"{label}: count={len(values)} p50={p50:.1f}ms "
        f"p95={p95:.1f}ms p99={p99:.1f}ms"
    )


def _read_records(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def main() -> None:
    records = _read_records(LOG_PATH)
    if not records:
        print(f"No query log data found at {LOG_PATH}")
        return

    latency_ms = _extract_values(records, "latency_ms")
    ttft_ms = _extract_values(records, "ttft_ms")

    print(_format_summary("latency_ms", latency_ms))
    print(_format_summary("ttft_ms", ttft_ms))


if __name__ == "__main__":
    main()
