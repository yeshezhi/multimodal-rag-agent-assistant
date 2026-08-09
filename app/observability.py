import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ObservabilityStore:
    """Small local-first query audit log for demos and offline evaluation."""

    def __init__(self, data_dir: Path, max_events: int = 500) -> None:
        self.path = data_dir / "query_events.jsonl"
        self.max_events = max_events
        self._lock = threading.RLock()

    def record(self, event: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self._lock:
            events = self._read_events()
            events.append(event)
            events = events[-self.max_events :]
            self.path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in events),
                encoding="utf-8",
            )

    def summary(self, recent_limit: int = 8) -> dict[str, Any]:
        with self._lock:
            events = self._read_events()
        total = len(events)
        answered = sum(event.get("outcome") == "answered" for event in events)
        refused = sum(event.get("outcome") == "refused" for event in events)
        latency_values = [event["latency_ms"] for event in events if "latency_ms" in event]
        return {
            "total_queries": total,
            "answered_queries": answered,
            "refused_queries": refused,
            "average_latency_ms": round(sum(latency_values) / len(latency_values), 1)
            if latency_values
            else None,
            "recent": list(reversed(events[-recent_limit:])),
        }

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events


def load_evaluation_report(data_dir: Path) -> dict[str, Any] | None:
    path = data_dir / "evaluation_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
