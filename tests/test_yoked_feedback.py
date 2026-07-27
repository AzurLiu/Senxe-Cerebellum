from core.contact_skill import ContactFeedbackEvent
from core.yoked_feedback import (
    event_for_step,
    load_feedback_schedule,
    load_feedback_schedule_metadata,
    save_feedback_schedule,
)


def test_yoked_feedback_round_trip_preserves_dose_and_timing(tmp_path):
    event = ContactFeedbackEvent(
        kind="collision",
        valence=-1,
        strength=0.75,
        distance_delta_m=-0.002,
        force_n=18.0,
    )
    path = tmp_path / "schedule.json"

    save_feedback_schedule(
        path,
        {3: [event]},
        metadata={"culture_id": "donor"},
    )
    schedule = load_feedback_schedule(path)

    assert event_for_step(schedule, 3, 0) == event
    assert load_feedback_schedule_metadata(path)["culture_id"] == "donor"
    missing = event_for_step(schedule, 3, 1)
    assert missing.kind == "yoked_missing"
    assert not missing.active
