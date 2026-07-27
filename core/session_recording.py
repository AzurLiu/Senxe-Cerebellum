"""CL session recording plus synchronized project telemetry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SessionRecordingConfig:
    enabled: bool = False
    file_location: str | None = None
    file_suffix: str | None = "senxe"
    include_raw_samples: bool = True


class CLSessionRecorder:
    """Small lifecycle wrapper around ``Neurons.record`` and ``DataStream``."""

    def __init__(
        self,
        neurons: Any,
        config: SessionRecordingConfig | None = None,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self.neurons = neurons
        self.config = config or SessionRecordingConfig()
        self.attributes = dict(attributes or {})
        self.recording: Any | None = None
        self.control_stream: Any | None = None

    def start(self) -> None:
        if not self.config.enabled or self.recording is not None:
            return
        kwargs: dict[str, Any] = {
            "include_spikes": True,
            "include_stims": True,
            "include_raw_samples": self.config.include_raw_samples,
            "include_data_streams": True,
        }
        if self.config.file_location:
            location = Path(self.config.file_location).expanduser()
            location.mkdir(parents=True, exist_ok=True)
            kwargs["file_location"] = str(location)
        if self.config.file_suffix:
            kwargs["file_suffix"] = self.config.file_suffix
        self.control_stream = self.neurons.create_data_stream(
            "senxe_control",
            attributes=self.attributes,
        )
        self.recording = self.neurons.record(**kwargs)

    def append(self, payload: Mapping[str, Any]) -> None:
        if self.control_stream is None:
            return
        timestamp = int(self.neurons.timestamp())
        serialized = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
        )
        self.control_stream.append(timestamp, serialized)

    def stop(self) -> None:
        if self.recording is None:
            return
        stop = getattr(self.recording, "stop", None)
        if callable(stop):
            stop()
        else:
            close = getattr(self.recording, "close", None)
            if callable(close):
                close()
        self.recording = None
        self.control_stream = None

    def __enter__(self) -> "CLSessionRecorder":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
