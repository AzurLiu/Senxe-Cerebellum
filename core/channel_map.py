"""Hardware-valid, disjoint CL1 channel allocation for the contact skill.

The CL1 records 64 channels, but only 59 are stimulatable.  The confirmatory
contact-skill path assigns those 59 channels as:

    18 sensory + 20 motor readout + 6 feedback + 15 reserve

The default numbers are spatially scattered rather than contiguous.  This
keeps task semantics separate while retaining spare electrodes for a
preregistered, calibration-driven replacement procedure on real cultures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


CHANNEL_COUNT = 64
NON_STIMULATABLE_CHANNELS = frozenset({0, 4, 7, 56, 63})
STIMULATABLE_CHANNELS = tuple(
    channel
    for channel in range(CHANNEL_COUNT)
    if channel not in NON_STIMULATABLE_CHANNELS
)
MOTOR_OUTPUT_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "soften",
    "retract",
)
CHANNEL_MAP_VERSION = "senxe_contact_18_20_6_15_v1"


@dataclass(frozen=True)
class AntagonisticChannelGroup:
    """Positive and negative readout populations for one skill coordinate."""

    positive: tuple[int, ...]
    negative: tuple[int, ...]


@dataclass(frozen=True)
class ContactChannelMap:
    """Complete allocation for the confirmatory contact-skill path."""

    position_channels: tuple[int, ...]
    force_channels: tuple[int, ...]
    phase_channels: tuple[int, ...]
    contact_channels: tuple[int, ...]
    motor_groups: tuple[AntagonisticChannelGroup, ...]
    positive_feedback_channels: tuple[int, ...]
    negative_feedback_channels: tuple[int, ...]
    reserve_channels: tuple[int, ...]
    version: str = CHANNEL_MAP_VERSION

    @property
    def sensory_channels(self) -> tuple[int, ...]:
        return (
            self.position_channels
            + self.force_channels
            + self.phase_channels
            + self.contact_channels
        )

    @property
    def motor_channels(self) -> tuple[int, ...]:
        return tuple(
            channel
            for group in self.motor_groups
            for channel in group.positive + group.negative
        )

    @property
    def feedback_channels(self) -> tuple[int, ...]:
        return (
            self.positive_feedback_channels
            + self.negative_feedback_channels
        )

    def decoder_groups(
        self,
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        return tuple(
            (group.positive, group.negative)
            for group in self.motor_groups
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update({
            "non_stimulatable_channels": sorted(
                NON_STIMULATABLE_CHANNELS
            ),
            "sensory_channel_count": len(self.sensory_channels),
            "motor_channel_count": len(self.motor_channels),
            "feedback_channel_count": len(self.feedback_channels),
            "reserve_channel_count": len(self.reserve_channels),
            "motor_output_names": list(MOTOR_OUTPUT_NAMES),
        })
        return value

    def stable_hash(self) -> str:
        serialized = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


DEFAULT_CONTACT_CHANNEL_MAP = ContactChannelMap(
    # Two signed electrodes per axis; magnitude is carried by pulse amplitude.
    position_channels=(8, 9, 10, 17, 18, 25),
    force_channels=(27, 28, 57, 1, 2, 3),
    # Seven task phases are represented by non-zero three-bit patterns.
    phase_channels=(5, 6, 11),
    contact_channels=(12, 15, 16),
    motor_groups=(
        AntagonisticChannelGroup((41, 42), (50, 51)),  # delta_x
        AntagonisticChannelGroup((13, 14), (45, 46)),  # delta_y
        AntagonisticChannelGroup((29, 30), (59, 60)),  # delta_z
        AntagonisticChannelGroup((32, 33), (49, 58)),  # soften
        AntagonisticChannelGroup((34, 37), (53, 61)),  # retract
    ),
    positive_feedback_channels=(19, 20, 22),
    negative_feedback_channels=(23, 24, 26),
    reserve_channels=(
        21,
        31,
        35,
        36,
        38,
        39,
        40,
        43,
        44,
        47,
        48,
        52,
        54,
        55,
        62,
    ),
)


def validate_contact_channel_map(
    channel_map: ContactChannelMap,
) -> None:
    """Reject invalid counts, overlaps, and hardware-forbidden assignments."""

    if len(channel_map.position_channels) != 6:
        raise ValueError("position_channels must contain six signed channels")
    if len(channel_map.force_channels) != 6:
        raise ValueError("force_channels must contain six signed channels")
    if len(channel_map.phase_channels) != 3:
        raise ValueError("phase_channels must contain three code channels")
    if len(channel_map.contact_channels) != 3:
        raise ValueError("contact_channels must contain three channels")
    if len(channel_map.motor_groups) != len(MOTOR_OUTPUT_NAMES):
        raise ValueError("motor_groups must contain five skill outputs")
    if any(
        len(group.positive) != 2 or len(group.negative) != 2
        for group in channel_map.motor_groups
    ):
        raise ValueError(
            "every motor output must contain two positive and two negative "
            "channels"
        )
    if len(channel_map.positive_feedback_channels) != 3:
        raise ValueError("positive feedback must contain three channels")
    if len(channel_map.negative_feedback_channels) != 3:
        raise ValueError("negative feedback must contain three channels")
    if len(channel_map.reserve_channels) != 15:
        raise ValueError("reserve_channels must contain 15 channels")

    regions = {
        "sensory": channel_map.sensory_channels,
        "motor": channel_map.motor_channels,
        "feedback": channel_map.feedback_channels,
        "reserve": channel_map.reserve_channels,
    }
    for name, channels in regions.items():
        if len(channels) != len(set(channels)):
            raise ValueError(f"{name} channel region contains duplicates")
        invalid = set(channels) - set(STIMULATABLE_CHANNELS)
        if invalid:
            raise ValueError(
                f"{name} channel region contains non-stimulatable channels: "
                f"{sorted(invalid)}"
            )

    region_names = tuple(regions)
    for index, left_name in enumerate(region_names):
        for right_name in region_names[index + 1:]:
            overlap = set(regions[left_name]) & set(regions[right_name])
            if overlap:
                raise ValueError(
                    f"{left_name} and {right_name} channels overlap: "
                    f"{sorted(overlap)}"
                )

    assigned = set().union(*(set(channels) for channels in regions.values()))
    if assigned != set(STIMULATABLE_CHANNELS):
        missing = set(STIMULATABLE_CHANNELS) - assigned
        extra = assigned - set(STIMULATABLE_CHANNELS)
        raise ValueError(
            "channel map must account for every stimulatable channel; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def channel_map_metadata(
    channel_map: ContactChannelMap = DEFAULT_CONTACT_CHANNEL_MAP,
) -> dict[str, Any]:
    validate_contact_channel_map(channel_map)
    return {
        "channel_map_version": channel_map.version,
        "channel_map_hash": channel_map.stable_hash(),
        "channel_map": channel_map.to_dict(),
    }


validate_contact_channel_map(DEFAULT_CONTACT_CHANNEL_MAP)
