from core.contact_scenarios import (
    ContactScenarioSchedule,
    default_contact_scenario_path,
    load_contact_scenarios,
)


def test_contact_scenarios_have_separate_train_and_heldout_physics():
    scenarios = load_contact_scenarios(default_contact_scenario_path())

    assert len(scenarios["train"]) >= 4
    assert len(scenarios["heldout"]) >= 3
    train_ids = {scenario.scenario_id for scenario in scenarios["train"]}
    heldout_ids = {scenario.scenario_id for scenario in scenarios["heldout"]}
    assert train_ids.isdisjoint(heldout_ids)
    assert any(
        scenario.peg_offset_xy_m != (0.0, 0.0)
        for scenario in scenarios["heldout"]
    )


def test_frozen_evaluation_selects_only_heldout_scenarios():
    scenarios = load_contact_scenarios(default_contact_scenario_path())
    schedule = ContactScenarioSchedule(scenarios)

    training = schedule.select(3, frozen_evaluation=False)
    evaluation = schedule.select(3, frozen_evaluation=True)

    assert training.split == "train"
    assert evaluation.split == "heldout"
