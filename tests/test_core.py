from dataclasses import replace
import pytest

from core import SCENARIOS, FAULT_LABELS, evaluate_rules, fingerprint, generate_history, state_payload, sweep_states, validate


def test_each_preset_has_finite_features_and_valid_labels():
    for key, s in SCENARIOS.items():
        assert s.key == key
        assert s.truth in FAULT_LABELS
        payload = state_payload(s.state)
        assert payload['sensors']['shaft_speed_rpm'] > 0
        assert len(generate_history(s.state, s.trend_strength)) == 49


def test_preset_baseline_expected_classifications():
    for key, expected in [
        ('healthy','normal'), ('bearing','bearing_wear'), ('lubrication','lubrication'),
        ('misalignment','misalignment'), ('imbalance','imbalance'), ('sensor','uncertain'),
    ]:
        actual = evaluate_rules(SCENARIOS[key].state)['choice']
        assert actual == expected, (key, actual)


def test_bad_sensor_is_flagged_not_mislabeled_as_defect():
    r = evaluate_rules(SCENARIOS['sensor'].state)
    assert r['flags']
    assert r['choice'] == 'uncertain'


def test_missing_temperature_is_flagged_and_preserved_as_null():
    state = replace(SCENARIOS['bearing'].state, temperature_c=None)
    assert validate(state)
    assert state_payload(state)['sensors']['bearing_temperature_c'] is None


def test_history_is_deterministic_and_ends_with_current_values():
    sc = SCENARIOS['bearing']
    a = generate_history(sc.state, sc.trend_strength, 123)
    b = generate_history(sc.state, sc.trend_strength, 123)
    assert a == b
    assert a[-1]['temperature_c'] == sc.state.temperature_c
    assert a[-1]['vibration_mm_s'] == sc.state.vibration_mm_s
    assert a[0]['minute'] == -120 and a[-1]['minute'] == 0


def test_fingerprint_detects_changed_sensor_or_notes():
    state = SCENARIOS['healthy'].state
    assert fingerprint(state) != fingerprint(replace(state, vibration_mm_s=5))
    assert fingerprint(state) != fingerprint(replace(state, observation='new note'))


def test_sweep_only_alters_one_channel():
    orig = SCENARIOS['healthy'].state
    states = sweep_states(orig, 'temperature_c', [40,50,60])
    assert [s.temperature_c for s in states] == [40,50,60]
    assert all(s.vibration_mm_s == orig.vibration_mm_s for s in states)
    with pytest.raises(ValueError):
        sweep_states(orig, 'rpm', [1000])
