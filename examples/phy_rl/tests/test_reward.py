"""Verifier behavior that guards the RL reward contract."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "shared"))

from physics_rlvr_common import (
    Answer,
    validate_answer,
    validate_release_state,
    validate_training_policy,
    verify_prediction,
)


def _answer(label: str, value: str, unit: str | None = None) -> Answer:
    return Answer(
        label=label,
        value=value,
        unit=unit,
        answer_type="numeric",
        verifier="numeric",
        atol=0.005,
        rtol=1e-6,
    )


def _completion(*outputs: tuple[str, str, str | None]) -> str:
    payload = [
        {"label": label, "value": value, "unit": unit}
        for label, value, unit in outputs
    ]
    return f"Reasoning omitted. <final>{json.dumps(payload)}</final>"


def test_multipart_reward_is_per_requested_output() -> None:
    answers = [_answer("acceleration", "-4.27", "m/s^2"), _answer("tension", "12.67", "N")]

    assert verify_prediction(_completion(("acceleration", "-4.27", "m/s^2")), answers) == 0.5
    assert verify_prediction(
        _completion(("acceleration", "-4.27", "m/s^2"), ("tension", "12.67", "N")), answers
    ) == 1.0


def test_extra_or_duplicate_output_labels_receive_zero() -> None:
    answers = [_answer("a", "1"), _answer("b", "2")]

    assert verify_prediction(_completion(("a", "1", None), ("b", "2", None), ("guess", "3", None)), answers) == 0.0
    duplicate = '<final>[{"label":"a","value":"1","unit":null},{"label":"a","value":"2","unit":null}]</final>'
    assert verify_prediction(duplicate, answers) == 0.0


def test_unit_conversion_requires_matching_dimensions() -> None:
    answers = [_answer("distance", "1000", "m")]

    assert verify_prediction(_completion(("distance", "1", "km")), answers) == 1.0
    assert verify_prediction(_completion(("distance", "1", "s")), answers) == 0.0
    assert verify_prediction(_completion(("distance", "1000", None)), answers) == 0.0
    optical_power = [_answer("power", "2", "1/m")]
    assert verify_prediction(_completion(("power", "2", "dptr")), optical_power) == 1.0
    assert verify_prediction(_completion(("power", "2", "D")), optical_power) == 0.0


def test_symbolic_unit_conversion_preserves_scale() -> None:
    answer = Answer(label="length", value="x", unit="m", answer_type="symbolic", verifier="sympy")
    assert verify_prediction(_completion(("length", "100 x", "cm")), [answer]) == 1.0
    assert verify_prediction(_completion(("length", "x", "cm")), [answer]) == 0.0
    assert verify_prediction(_completion(("length", "x", "not-a-unit")), [answer]) == 0.0
    capacitor = Answer(label="ratio", value=r"\frac{Ut^2}{d(2d+gt^2)}", unit="kg/C", answer_type="symbolic", verifier="sympy")
    assert verify_prediction(_completion(("ratio", r"\frac{U t^2}{2d^2+g d t^2}", "kg/C")), [capacitor]) == 1.0
    assert verify_prediction(_completion(("ratio", r"\frac{U t^2}{2d^2-g d t^2}", "kg/C")), [capacitor]) == 0.0


def test_mixed_notation_preserves_powers_and_rejects_wrong_physics() -> None:
    answer = Answer(label="pressure", value=r"2 * \nu * \mu * v^{2} / V",
                    unit="Pa", answer_type="symbolic", verifier="sympy")
    assert verify_prediction(_completion(("pressure", r"2 * \nu * \mu * v**2 / V", "Pa")), [answer]) == 1.0
    assert verify_prediction(_completion(("pressure", r"4 * \nu * \mu * v**2 / V", "Pa")), [answer]) == 0.0
    assert verify_prediction(_completion(("pressure", r"-2 * \nu * \mu * v**2 / V", "Pa")), [answer]) == 0.0
    assert verify_prediction(_completion(("pressure", r"2 * \nu * \mu * v**2 / V", "N")), [answer]) == 0.0
    assert verify_prediction(_completion(("pressure", "__import__('os').system('echo unsafe')", "Pa")), [answer]) == 0.0
    decay = Answer(label="decay", value=r"\exp(-t / \tau)", unit=None, answer_type="symbolic", verifier="sympy")
    assert verify_prediction(_completion(("decay", "exp(-t/tau)", None)), [decay]) == 1.0
    charge = Answer(label="charge", value="e*x", unit="C", answer_type="symbolic", verifier="sympy")
    assert verify_prediction(_completion(("charge", "e*x", "C")), [charge]) == 1.0


def test_square_root_equivalence_requires_explicit_domain() -> None:
    answer = Answer(label="period", value=r"2 * \pi * \sqrt{R_1^3 / (G * M)}",
                    unit="s", answer_type="symbolic", verifier="sympy")
    equivalent = r"2 * \pi * R_1 * \sqrt{R_1 / (G * M)}"
    assert verify_prediction(_completion(("period", equivalent, "s")), [answer]) == 0.0
    positive = Answer(**{**answer.__dict__, "assumptions": ["R_1 > 0", "G > 0", "M > 0"]})
    assert verify_prediction(_completion(("period", equivalent, "s")), [positive]) == 1.0
    assert verify_prediction(_completion(("period", r"2 * \pi * (R_1**3 / (G * M))**(1/2)", "s")), [positive]) == 1.0
    assert verify_prediction(_completion(("period", "-" + equivalent, "s")), [positive]) == 0.0
    unconstrained = Answer(label="root", value="sqrt(x**2)", unit=None, answer_type="symbolic", verifier="sympy")
    assert verify_prediction(_completion(("root", "x", None)), [unconstrained]) == 0.0
    contradictory = Answer(**{**unconstrained.__dict__, "assumptions": ["x > 0", "x < 0"]})
    assert validate_answer(contradictory)


def test_symbolic_temperature_conversion_uses_an_offset():
    answer = Answer(label="temperature", value="T + 273.15", unit="K", answer_type="symbolic", verifier="sympy")
    assert verify_prediction(_completion(("temperature", "T", "degC")), [answer]) == 1.0
    assert verify_prediction(_completion(("temperature", "274.15*T", "K")), [answer]) == 0.0
    fahrenheit = Answer(label="temperature", value="5*(T-32)/9", unit="degC", answer_type="symbolic", verifier="sympy")
    assert verify_prediction(_completion(("temperature", "T", "degF")), [fahrenheit]) == 1.0
    angle = Answer(label="angle", value=r"\pi/6", unit="rad", answer_type="symbolic", verifier="sympy")
    assert verify_prediction(_completion(("angle", "30", "degree")), [angle]) == 1.0
    assert verify_prediction(_completion(("angle", "60", "degree")), [angle]) == 0.0


def test_approximation_uses_reviewed_bindings_and_a_declared_tolerance():
    answer = Answer(label="speed", value="2e-6*c", unit="m/s", answer_type="symbolic", verifier="sympy",
                    bindings={"alpha": "1e-6"}, rtol=2e-6)
    exact = "c*(2*alpha+2*alpha**2)/(1+2*alpha+2*alpha**2)"
    assert verify_prediction(_completion(("speed", exact, "m/s")), [answer]) == 1.0
    assert verify_prediction(_completion(("speed", "-" + exact, "m/s")), [answer]) == 0.0
    assert verify_prediction(_completion(("speed", "2*(" + exact + ")", "m/s")), [answer]) == 0.0
    assert verify_prediction(_completion(("speed", "3e-6*c", "m/s")), [answer]) == 0.0
    assert verify_prediction(_completion(("speed", "2e-6*c*x", "m/s")), [answer]) == 0.0
    strict = Answer(**{**answer.__dict__, "rtol": 0})
    assert verify_prediction(_completion(("speed", exact, "m/s")), [strict]) == 0.0
    unspecified = Answer(**{**answer.__dict__, "bindings": {}})
    assert verify_prediction(_completion(("speed", exact, "m/s")), [unspecified]) == 0.0


def test_numeric_target_uses_explicit_rounding_tolerance() -> None:
    answer = Answer(
        label="result",
        value="2.39e-15",
        unit=None,
        answer_type="numeric",
        verifier="numeric",
        atol=1e-18,
        rtol=1e-4,
    )

    assert verify_prediction(_completion(("result", "2.3901e-15", None)), [answer]) == 1.0
    assert verify_prediction(_completion(("result", "2.5e-15", None)), [answer]) == 0.0
    threshold = _answer("coefficient", "2/3", "dimensionless")
    assert verify_prediction(_completion(("coefficient", "0.6666666667", None)), [threshold]) == 1.0
    assert verify_prediction(_completion(("coefficient", "-2/3", None)), [threshold]) == 0.0
    assert verify_prediction(_completion(("coefficient", "4/3", None)), [threshold]) == 0.0


def test_malformed_final_block_receives_zero() -> None:
    answers = [_answer("answer", "42")]

    assert verify_prediction("The answer is 42.", answers) == 0.0
    assert verify_prediction('<final>[{"label":"answer","value":"42","unit":null,}]</final>', answers) == 0.0


def test_latex_scientific_notation_is_checked_numerically() -> None:
    answer = Answer(
        label="value",
        value=r"2.39078 \times 10^{-15}",
        unit=None,
        answer_type="numeric",
        verifier="numeric",
        atol=1e-18,
        rtol=1e-4,
    )

    assert verify_prediction(_completion(("value", r"2.39 \times 10^{-15}", None)), [answer]) == 1.0


def test_malformed_verifier_targets_are_rejected() -> None:
    malformed_unit = Answer(
        label="result",
        value="42",
        unit="not-a-physics-unit",
        answer_type="numeric",
        verifier="numeric",
        atol=0.0,
        rtol=1e-6,
    )
    malformed_value = Answer(
        label="result",
        value="x+",
        unit=None,
        answer_type="numeric",
        verifier="numeric",
        atol=0.0,
        rtol=1e-6,
    )

    assert any("unit" in error for error in validate_answer(malformed_unit))
    assert any("numeric expression" in error for error in validate_answer(malformed_value))
    assert verify_prediction(_completion(("result", "42", "not-a-physics-unit")), [malformed_unit]) == 0.0


def test_training_policy_blocks_benchmarks_and_post_2023_sources() -> None:
    validate_training_policy(source="ipho_open_train", competition="IPhO", year=2023, split="train")

    with pytest.raises(ValueError, match="blocked"):
        validate_training_policy(source="ipho_open_train", competition="IPhO", year=2024, split="train")
    with pytest.raises(ValueError, match="blocked"):
        validate_training_policy(
            source="olympiadbench_physics", competition="OlympiadBench Physics", year=2020, split="train"
        )
    with pytest.raises(ValueError, match="explicit year"):
        validate_training_policy(source="ipho_open_train", competition="IPhO", year=None, split="train")
    for source, competition in [("nbpho_olimpicos", "NBPhO"), ("czech_physics_olympiad", "Czech Physics Olympiad")]:
        validate_training_policy(source=source, competition=competition, year=2023, split="train")
        with pytest.raises(ValueError, match="explicit year"):
            validate_training_policy(source=source, competition=competition, year=None, split="train")
        with pytest.raises(ValueError, match="blocked"):
            validate_training_policy(source=source, competition=competition, year=2024, split="train")
    with pytest.raises(ValueError, match="integer"):
        validate_training_policy(source="ipho_open_train", competition="IPhO", year=True, split="train")
    with pytest.raises(ValueError, match="blocked"):
        validate_training_policy(
            source="estonian_physics_olympiad", competition="Estonian Physics Olympiad", year=2019, split="train"
        )


def test_physics_dataset_is_blocked_as_a_training_source() -> None:
    for source in ["physics_training_release", "desimfj/PHYSICS", "PHYSICS_test",
                   "Darkyy/phy-rl-base", "darkyy_phy_rl_base", "phy_rl_base"]:
        for source_split in ["train", "test", None]:
            with pytest.raises(ValueError, match="blocked"):
                validate_training_policy(
                    source=source, competition="PHYSICS", year=None, split="train",
                    source_split=source_split, source_revision="a" * 64,
                )
    for repository in ["Darkyy/phy-rl-base", "desimfj/PHYSICS"]:
        with pytest.raises(ValueError, match="blocked in source provenance"):
            validate_release_state({"problem_id": "relabeled", "source": "ipho_open_train",
                                    "provenance": {"source_url": f"https://huggingface.co/datasets/{repository}/viewer/default/train"}})


def test_staged_records_cannot_enter_training() -> None:
    validate_release_state({"problem_id": "released", "status": "accepted", "release_status": "ready", "checks": {"units": True}})
    with pytest.raises(ValueError, match="not released"):
        validate_release_state({"problem_id": "pilot", "status": "model_checked", "release_status": "staging_only"})
    with pytest.raises(ValueError, match="review status"):
        validate_release_state({"problem_id": "pilot", "status": "review", "release_status": "ready"})
    with pytest.raises(ValueError, match="failed validation"):
        validate_release_state({"problem_id": "pilot", "release_status": "ready", "checks": {"units": False}})
