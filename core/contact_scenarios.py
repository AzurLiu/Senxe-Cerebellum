"""Physical train/held-out perturbations for contact-skill generalization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ContactScenario:
    scenario_id: str
    split: str
    peg_offset_xy_m: tuple[float, float]
    peg_yaw_deg: float
    friction_scale: float

    def __post_init__(self) -> None:
        if self.split not in {"train", "heldout"}:
            raise ValueError("scenario split must be train or heldout")
        if len(self.peg_offset_xy_m) != 2:
            raise ValueError("peg_offset_xy_m must contain x and y")
        if self.friction_scale <= 0:
            raise ValueError("friction_scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_contact_scenarios(
    path: str | Path,
) -> dict[str, tuple[ContactScenario, ...]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = tuple(
        ContactScenario(
            scenario_id=str(item["scenario_id"]),
            split=str(item["split"]),
            peg_offset_xy_m=tuple(
                float(value) for value in item["peg_offset_xy_m"]
            ),
            peg_yaw_deg=float(item["peg_yaw_deg"]),
            friction_scale=float(item["friction_scale"]),
        )
        for item in payload["scenarios"]
    )
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("contact scenario identifiers must be unique")
    return {
        split: tuple(
            scenario for scenario in scenarios if scenario.split == split
        )
        for split in ("train", "heldout")
    }


def default_contact_scenario_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "config"
        / "contact_generalization.json"
    )


class ContactScenarioSchedule:
    """Deterministically select train or held-out physics per episode."""

    def __init__(
        self,
        scenarios: dict[str, Sequence[ContactScenario]],
    ) -> None:
        self.scenarios = {
            split: tuple(values) for split, values in scenarios.items()
        }
        if not self.scenarios.get("train"):
            raise ValueError("at least one train scenario is required")
        if not self.scenarios.get("heldout"):
            raise ValueError("at least one heldout scenario is required")

    def select(
        self,
        episode_index: int,
        *,
        frozen_evaluation: bool,
    ) -> ContactScenario:
        split = "heldout" if frozen_evaluation else "train"
        values = self.scenarios[split]
        return values[int(episode_index) % len(values)]


class RoboSuiteContactPerturbation:
    """Apply scenario parameters to real MuJoCo model state after reset."""

    def __init__(
        self,
        raw_env: Any,
        *,
        peg_body_name: str = "peg1",
        nut_geom_prefix: str = "SquareNut_g",
    ) -> None:
        self.raw_env = raw_env
        self.model = raw_env.sim.model
        self.peg_body_name = peg_body_name
        self.nut_geom_prefix = nut_geom_prefix
        self._peg_body_id = _body_id(self.model, peg_body_name)
        self._nut_geom_ids = tuple(
            geom_id
            for geom_id in range(int(self.model.ngeom))
            if _geom_name(self.model, geom_id).startswith(nut_geom_prefix)
            and not _geom_name(self.model, geom_id).endswith("_visual")
        )
        baseline_key = "_senxe_contact_physics_baseline"
        baseline = getattr(raw_env, baseline_key, None)
        if baseline is None:
            baseline = {
                "peg_pos": np.asarray(
                    self.model.body_pos[self._peg_body_id],
                    dtype=float,
                ).copy(),
                "peg_quat": np.asarray(
                    self.model.body_quat[self._peg_body_id],
                    dtype=float,
                ).copy(),
                "friction": {
                    geom_id: np.asarray(
                        self.model.geom_friction[geom_id],
                        dtype=float,
                    ).copy()
                    for geom_id in self._nut_geom_ids
                },
            }
            setattr(raw_env, baseline_key, baseline)
        self._base_peg_pos = np.asarray(baseline["peg_pos"]).copy()
        self._base_peg_quat = np.asarray(baseline["peg_quat"]).copy()
        self._base_friction = {
            int(geom_id): np.asarray(value).copy()
            for geom_id, value in baseline["friction"].items()
        }
        self.active_scenario: ContactScenario | None = None

    def apply(self, scenario: ContactScenario) -> None:
        # RoboSuite may rebuild the MuJoCo model during a hard reset. Always
        # resolve the live model before applying a held-out physics scenario.
        self.model = self.raw_env.sim.model
        self._peg_body_id = _body_id(self.model, self.peg_body_name)
        position = self._base_peg_pos.copy()
        position[:2] += np.asarray(scenario.peg_offset_xy_m, dtype=float)
        self.model.body_pos[self._peg_body_id] = position
        yaw_radians = np.deg2rad(scenario.peg_yaw_deg)
        yaw_quaternion = np.array([
            np.cos(yaw_radians / 2.0),
            0.0,
            0.0,
            np.sin(yaw_radians / 2.0),
        ])
        self.model.body_quat[self._peg_body_id] = _quaternion_multiply(
            yaw_quaternion,
            self._base_peg_quat,
        )
        for geom_id, baseline in self._base_friction.items():
            friction = baseline.copy()
            friction[0] *= scenario.friction_scale
            self.model.geom_friction[geom_id] = friction
        self.raw_env.sim.forward()
        self.active_scenario = scenario


def _body_id(model: Any, name: str) -> int:
    if hasattr(model, "body"):
        return int(model.body(name).id)
    return int(model.body_name2id(name))


def _geom_name(model: Any, geom_id: int) -> str:
    if hasattr(model, "geom"):
        return str(model.geom(geom_id).name or "")
    return str(model.geom_id2name(geom_id) or "")


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])
    return result / max(float(np.linalg.norm(result)), 1e-12)
