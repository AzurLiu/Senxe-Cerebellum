"""Auditable export and replay of dose-matched feedback schedules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.contact_skill import ContactFeedbackEvent


FeedbackSchedule = dict[int, tuple[ContactFeedbackEvent, ...]]


def _load_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported yoked feedback schedule schema")
    return payload


def load_feedback_schedule(path: str | Path) -> FeedbackSchedule:
    payload = _load_payload(path)
    episodes = payload.get("episodes")
    if not isinstance(episodes, Mapping):
        raise ValueError("yoked feedback schedule must contain episodes")
    return {
        int(episode): tuple(
            ContactFeedbackEvent.from_mapping(event) for event in events
        )
        for episode, events in episodes.items()
    }


def load_feedback_schedule_metadata(path: str | Path) -> dict[str, Any]:
    payload = _load_payload(path)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("yoked feedback metadata must be a mapping")
    return dict(metadata)


def save_feedback_schedule(
    path: str | Path,
    schedule: Mapping[int, Sequence[ContactFeedbackEvent | Mapping[str, Any]]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    episodes: dict[str, list[dict[str, Any]]] = {}
    for episode, events in schedule.items():
        episodes[str(int(episode))] = [
            event.to_dict()
            if isinstance(event, ContactFeedbackEvent)
            else dict(event)
            for event in events
        ]
    payload = {
        "schema_version": 1,
        "metadata": dict(metadata or {}),
        "episodes": episodes,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def event_for_step(
    schedule: FeedbackSchedule,
    episode: int,
    step: int,
) -> ContactFeedbackEvent:
    events = schedule.get(int(episode), ())
    if 0 <= int(step) < len(events):
        return events[int(step)]
    return ContactFeedbackEvent(
        kind="yoked_missing",
        valence=0,
        strength=0.0,
        distance_delta_m=0.0,
        force_n=0.0,
    )
