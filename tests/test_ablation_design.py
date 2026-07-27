from types import SimpleNamespace

import pytest

import run_ablation_benchmark as benchmark
from core.yoked_feedback import save_feedback_schedule
from run_ablation_benchmark import CONDITIONS, counterbalanced_condition_order


def test_counterbalanced_orders_are_complete_and_rotate():
    names = {condition["name"] for condition in CONDITIONS}
    first = counterbalanced_condition_order(0, seed=9)
    second = counterbalanced_condition_order(1, seed=9)

    assert set(first) == names
    assert set(second) == names
    assert second == first[1:] + first[:1]
    assert "yoked_feedback" in first


def test_real_yoked_condition_requires_versioned_independent_donor(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(benchmark, "RECORD_CL_SESSION", True)
    args = SimpleNamespace(
        culture_id="recipient",
        condition="yoked_feedback",
        allow_hardware_crossover=False,
        yoked_feedback=None,
        output=tmp_path / "results.csv",
    )

    with pytest.raises(RuntimeError, match="independent donor"):
        benchmark._validate_hardware_scope(args, simulator=False)

    path = tmp_path / "donor.json"
    provenance = benchmark.project_provenance()
    save_feedback_schedule(
        path,
        {},
        metadata={
            "culture_id": "donor",
            "condition": "contact_skill",
            "protocol_id": benchmark.BENCHMARK_PROTOCOL_ID,
            **provenance,
        },
    )
    args.yoked_feedback = path
    benchmark._validate_hardware_scope(args, simulator=False)


def test_ablation_csv_refuses_duplicate_evidence_rows(tmp_path):
    path = tmp_path / "results.csv"
    row = {field: "" for field in benchmark.CSV_FIELDS}
    row.update({
        "RunID": "culture:condition",
        "Condition": "contact_skill",
        "Episode": 0,
    })

    benchmark._append_rows(path, [row])
    with pytest.raises(RuntimeError, match="duplicate"):
        benchmark._append_rows(path, [row])


def test_ablation_schema_records_per_episode_electrical_dose():
    assert {
        "StimSafetyLimits",
        "StimAbsChargeNc",
        "EpisodeStimCalls",
        "EpisodeStimChannelPulses",
        "EpisodeStimAbsChargeNc",
    } <= set(benchmark.CSV_FIELDS)


def test_project_provenance_hashes_runtime_and_protocol_sources():
    provenance = benchmark.project_provenance()

    assert len(provenance["code_hash"]) == 64
    assert len(provenance["protocol_hash"]) == 64
    assert len(provenance["channel_map_hash"]) == 64
    assert provenance["channel_map_version"]
    assert provenance["git_commit"]


def test_protocol_provenance_hash_is_profile_specific():
    confirmatory = benchmark.project_provenance(
        protocol_id="senxe_contact_skill_v2",
    )
    application = benchmark.project_provenance(
        protocol_id="senxe_contact_skill_application_v1",
    )

    assert confirmatory["protocol_hash"] != application["protocol_hash"]
    assert confirmatory["code_hash"] == application["code_hash"]


def test_real_run_rejects_reusing_culture_for_another_condition(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(benchmark, "RECORD_CL_SESSION", True)
    path = tmp_path / "results.csv"
    row = {field: "" for field in benchmark.CSV_FIELDS}
    row.update({
        "RunID": "culture_1:baseline",
        "CultureID": "culture_1",
        "Condition": "baseline_only",
        "Episode": 0,
    })
    benchmark._append_rows(path, [row])
    args = SimpleNamespace(
        culture_id="culture_1",
        condition="contact_skill",
        allow_hardware_crossover=False,
        yoked_feedback=None,
        output=path,
    )

    with pytest.raises(RuntimeError, match="already appears"):
        benchmark._validate_hardware_scope(args, simulator=False)
