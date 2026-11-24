"""The deterministic controller must finish the task on its own.

The biological path only adds a small bounded correction, so nothing can be
learned about it unless the white-box controller already completes the
placement across the perturbation scenarios. Each test pins one reason it
used to fail.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import senxe_demo_robosuite as demo
from core.hybrid_control import (
    NominalControlConfig,
    NominalTaskController,
    TaskPhase,
)


def _observation(**overrides):
    value = {
        "eef_to_nut": np.zeros(3),
        "nut_to_peg": np.array([0.2, 0.0, 0.0]),
        "grasp_confirmed": False,
        "nut_on_peg": False,
        "placement_success": False,
        "nut_height_above_table_m": 0.17,
        "grasp_yaw_error_rad": 0.0,
        "nut_peg_yaw_error_rad": 0.0,
        "nut_level_error_rad": np.zeros(3),
    }
    value.update(overrides)
    return value


def _carrying(controller: NominalTaskController, stage: str = "align"):
    controller.phase = TaskPhase.TRANSPORT
    controller._transport_stage = stage
    return controller


def test_nut_is_carried_above_the_top_of_the_peg():
    cfg = NominalControlConfig()
    peg_half_height = 0.10  # NutAssemblySquare peg; peg_pos is its centre
    assert cfg.transport_clearance_m >= peg_half_height + 0.025
    # The lift must reach the carry height before any sideways motion: the
    # peg top is about 0.13 m above the table.
    assert cfg.lift_height_above_table_m - cfg.lift_tolerance_m >= 0.14


def test_descent_waits_until_the_gripper_is_centred_over_the_handle():
    controller = NominalTaskController()
    hover = NominalControlConfig().hover_height_m

    off_centre = _observation(eef_to_nut=np.array([0.015, 0.0, -hover]))
    controller.propose(off_centre)
    assert controller._approach_stage == "hover"

    centred = _observation(eef_to_nut=np.array([0.002, 0.0, -hover]))
    controller.propose(centred)
    assert controller._approach_stage == "descend"


def test_blocked_descent_goes_back_up_instead_of_pressing_down():
    controller = NominalTaskController(NominalControlConfig(
        descend_timeout_steps=3,
    ))
    hover = controller.config.hover_height_m
    controller.propose(_observation(eef_to_nut=np.array([0.0, 0.0, -hover])))
    assert controller._approach_stage == "descend"

    # Centred but still 6 cm short and not getting closer: a finger is
    # resting on something.
    blocked = _observation(eef_to_nut=np.array([0.001, 0.0, -0.03]))
    for _ in range(2):
        action = controller.propose(blocked)
        assert action[2] < 0.0
    action = controller.propose(blocked)
    assert controller._approach_stage == "hover"
    assert controller.phase is TaskPhase.APPROACH_NUT
    assert action[2] > 0.0
    assert action[-1] == -1.0


def test_jammed_insertion_lifts_and_lines_up_again():
    controller = NominalTaskController(NominalControlConfig(
        insert_timeout_steps=2,
    ))
    controller.phase = TaskPhase.INSERT
    jammed = _observation(
        nut_to_peg=np.array([0.0, 0.0, -0.04]),
        grasp_confirmed=True,
    )
    for _ in range(2):
        assert controller.propose(jammed)[2] < 0.0
    action = controller.propose(jammed)
    assert controller.phase is TaskPhase.TRANSPORT
    assert controller._transport_stage == "align"
    assert action[2] > 0.0
    assert action[-1] == 1.0


def test_command_does_not_fade_to_nothing_just_outside_tolerance():
    controller = _carrying(NominalTaskController())
    cfg = controller.config
    clearance = cfg.transport_clearance_m
    # 6 mm off in x: outside the 4 mm tolerance. A pure proportional
    # command would be 0.036, which barely moves the arm.
    action = controller.propose(_observation(
        nut_to_peg=np.array([0.006, 0.0, -clearance]),
        grasp_confirmed=True,
    ))
    assert np.linalg.norm(action[:3]) >= cfg.min_translation_action
    assert action[0] > 0.0
    # On target the floor does not apply, so the arm can settle.
    from core.hybrid_control import _with_floor

    small = np.array([0.006, 0.0, 0.0])
    settled = _with_floor(
        small,
        error_norm=0.001,
        floor=cfg.min_translation_action,
        floor_error=cfg.floor_error_m,
    )
    assert np.array_equal(settled, small)


def test_tilted_nut_is_levelled_and_blocks_insertion_until_level():
    controller = _carrying(NominalTaskController())
    cfg = controller.config
    aligned = dict(
        nut_to_peg=np.array([0.0, 0.0, -cfg.transport_clearance_m]),
        grasp_confirmed=True,
    )

    tilted = controller.propose(_observation(
        nut_level_error_rad=np.array([0.0, 0.2, 0.0]),
        **aligned,
    ))
    assert controller.phase is TaskPhase.TRANSPORT
    assert tilted[4] > 0.0
    assert tilted[3] == 0.0
    assert abs(tilted[4]) <= cfg.max_level_action

    controller.propose(_observation(**aligned))
    assert controller.phase is TaskPhase.INSERT


def test_arm_stops_pushing_down_once_the_contact_pushes_back():
    cfg = NominalControlConfig()
    pushing_back = np.array([0.0, 0.0, -(cfg.contact_yield_force_n + 1.0)])

    inserting = NominalTaskController()
    inserting.phase = TaskPhase.INSERT
    target = np.array([0.0, 0.0, -0.04])
    free = inserting.propose(_observation(
        nut_to_peg=target, grasp_confirmed=True,
    ))
    assert free[2] < 0.0
    held = inserting.propose(_observation(
        nut_to_peg=target, grasp_confirmed=True, force=pushing_back,
    ))
    assert held[2] == 0.0

    # The yield only applies to the two downward-pressing stages.
    carrying = _carrying(NominalTaskController(), stage="lift")
    lifting = carrying.propose(_observation(
        nut_height_above_table_m=0.05,
        grasp_confirmed=True,
        force=pushing_back,
    ))
    assert lifting[2] > 0.0


def test_levelling_is_not_commanded_before_the_nut_is_held():
    controller = NominalTaskController()
    action = controller.propose(_observation(
        eef_to_nut=np.array([0.1, 0.0, 0.0]),
        nut_level_error_rad=np.array([0.2, 0.2, 0.0]),
    ))
    assert controller.phase is TaskPhase.APPROACH_NUT
    assert action[3] == 0.0 and action[4] == 0.0


def _yaw_quat(yaw_deg: float) -> np.ndarray:
    half = np.radians(yaw_deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def test_level_error_is_the_rotation_that_uprights_the_nut():
    assert np.allclose(demo._level_error_rad(_yaw_quat(40.0)), 0.0)
    half = np.radians(10.0) / 2.0
    pitched = np.array([np.cos(half), 0.0, np.sin(half), 0.0])  # +10° about y
    error = demo._level_error_rad(pitched)
    assert error == pytest.approx([0.0, -np.radians(10.0), 0.0], abs=1e-9)


@pytest.mark.parametrize("nut_yaw_deg", [-170, -100, -60, -10, 20, 80, 130])
def test_placement_yaw_keeps_the_handle_on_the_robot_side(nut_yaw_deg):
    base = np.array([-0.56, 0.0, 0.9])
    nut = np.array([0.23, 0.10, 1.0])
    error = demo._reachable_yaw_error_rad(
        _yaw_quat(nut_yaw_deg),
        _yaw_quat(0.0),
        nut,
        base,
    )
    final_handle_yaw = np.radians(nut_yaw_deg) - error
    # One of the four square-symmetric fits...
    assert (np.degrees(final_handle_yaw) % 90.0) == pytest.approx(
        0.0, abs=1e-6
    ) or (np.degrees(final_handle_yaw) % 90.0) == pytest.approx(
        90.0, abs=1e-6
    )
    # ...and never the one that puts the gripper beyond the peg.
    away = (nut[:2] - base[:2]) / np.linalg.norm(nut[:2] - base[:2])
    outward = np.cos(final_handle_yaw) * away[0] + np.sin(
        final_handle_yaw
    ) * away[1]
    assert outward <= 0.2 + 1e-9


class _SilentNeurons:
    """No spikes at all: the task is done by the deterministic path alone."""

    def __init__(self):
        self.cursor = 0

    def get_frames_per_second(self):
        return 25_000

    def timestamp(self):
        return self.cursor

    def stim(self, channels, design, burst):
        pass

    def get_health(self):
        return np.ones(64)

    def loop(self, **kwargs):
        frames_per_tick = int(round(25_000 / kwargs["ticks_per_second"]))
        ticks = []
        for _ in range(kwargs["stop_after_ticks"]):
            timestamp = self.cursor
            self.cursor += frames_per_tick
            ticks.append(SimpleNamespace(
                timestamp=timestamp,
                frames=np.zeros((frames_per_tick, 64), dtype=np.int16),
                analysis=SimpleNamespace(spikes=[], stims=[]),
            ))
        return iter(ticks)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deterministic_controller_completes_previously_failing_episodes():
    """Episodes 3 and 85 failed every time before: far-side grasp, drooping
    nut, and a carry height level with the peg top."""

    env, raw_env = demo.make_robosuite_env(render=False)
    try:
        agent = demo.CL1Agent(
            env,
            raw_env,
            _SilentNeurons(),
            protocol_id="senxe_contact_skill_v2",
        )
        for episode in (3, 85):
            agent.run_episode(max_steps=demo.MAX_STEPS, ep_num=episode)
            metrics = agent.last_episode_metrics
            assert metrics["episode_success"], (
                f"episode {episode} "
                f"({agent.active_scenario.scenario_id}) did not complete"
            )
            assert agent.nominal_controller.phase is TaskPhase.COMPLETE
    finally:
        env.close()


def test_remaining_offset_is_integrated_until_the_arm_arrives():
    controller = _carrying(NominalTaskController())
    cfg = controller.config
    stuck = _observation(
        nut_to_peg=np.array([0.006, 0.0, -cfg.transport_clearance_m]),
        grasp_confirmed=True,
    )
    commands = [controller.propose(stuck)[0] for _ in range(12)]
    assert all(b >= a for a, b in zip(commands, commands[1:]))
    assert commands[-1] > commands[0]
    # Bounded: the integral never adds more than its cap.
    for _ in range(500):
        last = controller.propose(stuck)
    assert last[0] <= (
        cfg.position_gain * 0.006 + cfg.max_integral_action + 1e-9
    ) or last[0] <= cfg.min_translation_action + cfg.max_integral_action + 1e-9

    # A new stage starts from zero again.
    controller.reset()
    assert np.array_equal(controller._integral, np.zeros(3))


def test_descent_holds_its_height_while_off_centre():
    controller = NominalTaskController()
    hover = controller.config.hover_height_m
    controller.propose(_observation(eef_to_nut=np.array([0.0, 0.0, -hover])))
    assert controller._approach_stage == "descend"

    off_centre = controller.propose(_observation(
        eef_to_nut=np.array([0.012, 0.0, -0.05]),
    ))
    assert off_centre[2] == 0.0
    assert off_centre[0] > 0.0

    centred = controller.propose(_observation(
        eef_to_nut=np.array([0.001, 0.0, -0.05]),
    ))
    assert centred[2] < 0.0


def test_placement_orientation_is_not_changed_while_carrying():
    controller = _carrying(NominalTaskController())
    aligned_xy = np.array([0.05, 0.0, -controller.config.transport_clearance_m])
    options = np.radians([10.0, 100.0, -170.0, -80.0])
    first = controller._committed_yaw_error(
        {
            "nut_peg_yaw_options_rad": options,
            "nut_peg_yaw_option_outward": np.array([-0.9, 0.1, 0.9, -0.1]),
        },
        0.0,
    )
    assert np.degrees(first) == pytest.approx(10.0)
    # Later the stateless rule would prefer a different quarter (the one
    # that was at 10 degrees now looks outward), but the carried nut keeps
    # heading for the placement it started with.
    later = controller._committed_yaw_error(
        {
            "nut_peg_yaw_options_rad": np.radians([4.0, 94.0, -176.0, -86.0]),
            "nut_peg_yaw_option_outward": np.array([0.9, -0.9, 0.9, -0.9]),
        },
        0.0,
    )
    assert np.degrees(later) == pytest.approx(4.0)
    assert aligned_xy is not None

    # Outside carrying phases there is nothing to commit to.
    controller.phase = TaskPhase.APPROACH_NUT
    assert controller._committed_yaw_error(
        {"nut_peg_yaw_options_rad": options}, 0.25,
    ) == 0.25
    assert controller._yaw_error_prev is None


def test_sideways_slip_before_the_peg_top_sends_the_nut_back_to_line_up():
    controller = NominalTaskController()
    cfg = controller.config
    controller.phase = TaskPhase.INSERT
    slipped_high = _observation(
        nut_to_peg=np.array([0.009, 0.0, -0.12]),
        grasp_confirmed=True,
    )
    controller.propose(slipped_high)
    assert controller.phase is TaskPhase.TRANSPORT
    assert controller._transport_stage == "align"

    # Once the nut is down on the peg, the peg guides it.
    controller.phase = TaskPhase.INSERT
    controller._insert_steps = 0
    engaged = _observation(
        nut_to_peg=np.array([0.009, 0.0, -0.05]),
        grasp_confirmed=True,
    )
    controller.propose(engaged)
    assert controller.phase is TaskPhase.INSERT
    assert cfg.insert_engaged_depth_m > 0.05
