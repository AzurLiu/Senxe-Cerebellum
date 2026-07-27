"""Stable hashes for the exact local experiment implementation."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess

from core.channel_map import (
    DEFAULT_CONTACT_CHANNEL_MAP,
    ContactChannelMap,
    channel_map_metadata,
)
from core.learning_protocol import default_protocol_path


def project_provenance(
    channel_map: ContactChannelMap = DEFAULT_CONTACT_CHANNEL_MAP,
    *,
    protocol_id: str = "senxe_contact_skill_v2",
) -> dict[str, str]:
    """Return runtime-source, protocol, and Git revision identifiers."""

    root = Path(__file__).resolve().parent.parent
    protocol_path = default_protocol_path()
    runtime_paths = [
        root / "run_ablation_benchmark.py",
        root / "senxe_demo_robosuite.py",
        root / "pyproject.toml",
        root / "requirements.txt",
        protocol_path,
        root / "config" / "contact_generalization.json",
        *sorted((root / "core").glob("*.py")),
    ]
    code_digest = hashlib.sha256()
    for path in runtime_paths:
        relative = path.relative_to(root).as_posix()
        code_digest.update(relative.encode("utf-8"))
        code_digest.update(b"\0")
        code_digest.update(path.read_bytes())
        code_digest.update(b"\0")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = "unavailable"
    package_versions = {"python": platform.python_version()}
    for distribution in (
        "cl-sdk",
        "gymnasium",
        "mujoco",
        "numpy",
        "robosuite",
    ):
        try:
            package_versions[distribution] = version(distribution)
        except PackageNotFoundError:
            package_versions[distribution] = "unavailable"
    map_metadata = channel_map_metadata(channel_map)
    protocol_registry = json.loads(protocol_path.read_text(encoding="utf-8"))
    matching_protocols = [
        protocol
        for protocol in protocol_registry["protocols"]
        if str(protocol["protocol_id"]) == protocol_id
    ]
    if len(matching_protocols) != 1:
        raise ValueError(f"unknown protocol_id for provenance: {protocol_id}")
    protocol_payload = json.dumps(
        matching_protocols[0],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "code_hash": code_digest.hexdigest(),
        "protocol_hash": hashlib.sha256(
            protocol_payload
        ).hexdigest(),
        "protocol_id": protocol_id,
        "git_commit": git_commit,
        "runtime_versions": json.dumps(
            package_versions,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "channel_map_version": str(
            map_metadata["channel_map_version"]
        ),
        "channel_map_hash": str(map_metadata["channel_map_hash"]),
        "channel_map": json.dumps(
            map_metadata["channel_map"],
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
