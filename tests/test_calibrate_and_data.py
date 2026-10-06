"""Temperature scaling recovers a known temperature; splits and seeds stay disjoint."""

from __future__ import annotations

import numpy as np
import pytest

from rlcd import calibrate as C
from rlcd import data as D
from rlcd import estimators as E
from rlcd import scoring as S

# ------------------------------------------------------------------- temperature scaling


def overconfident_data(rng, n, k, sharpen):
    """Calibrated logits z*, the labels they generate, and the model's sharpened logits z = s z*."""
    z_star = rng.normal(size=(n, k)) * 1.2
    y = E.sample_categorical(E.softmax(z_star), 1, rng)[:, 0]
    return z_star * sharpen, y


@pytest.mark.parametrize("sharpen", [0.6, 1.0, 1.7, 3.0])
def test_fit_temperature_recovers_the_known_scale(sharpen):
    rng = np.random.default_rng(0)
    z, y = overconfident_data(rng, 60_000, 5, sharpen)
    assert C.fit_temperature(z, y) == pytest.approx(sharpen, rel=0.04)


def test_scaling_lowers_nll_and_ece_and_keeps_accuracy():
    rng = np.random.default_rng(1)
    z, y = overconfident_data(rng, 40_000, 5, 2.5)
    t = C.fit_temperature(z, y)
    before, after = E.softmax(z), C.apply_temperature(z, t)
    assert S.log_loss(after, y) < S.log_loss(before, y)
    assert S.top_label_ece(after, y) < S.top_label_ece(before, y)
    np.testing.assert_array_equal(before.argmax(1), after.argmax(1))


def test_temperature_is_fitted_per_answer_type_and_falls_back_to_pooled():
    rng = np.random.default_rng(2)
    zc, yc = overconfident_data(rng, 20_000, 5, 1.5)
    zn, yn = overconfident_data(rng, 20_000, 2, 3.0)
    zs, ys = overconfident_data(rng, 10, 4, 2.0)  # too few rows: uses the pooled temperature
    scaler = C.TemperatureScaler(min_rows=50).fit(
        {"choice": (zc, yc), "noul": (zn, yn), "score": (zs, ys)}
    )
    assert scaler.temperature("choice") == pytest.approx(1.5, rel=0.06)
    assert scaler.temperature("noul") == pytest.approx(3.0, rel=0.06)
    assert scaler.temperature("score") == scaler.pooled
    assert set(scaler.to_config()["temperature_by_type"]) == {"choice", "noul"}
    probs = scaler.transform(zc[:3], "choice")
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_infinite_padding_logits_get_zero_probability():
    z = np.array([[1.0, 2.0, -np.inf]])
    p = C.apply_temperature(z, 2.0)
    assert p[0, 2] == 0.0 and p.sum() == pytest.approx(1.0)


def test_separable_data_sharpens_to_numerically_zero_nll_and_bad_temperature_is_rejected():
    z = np.array([[10.0, 0.0], [0.0, 10.0]] * 30)
    y = np.array([0, 1] * 30)
    t = C.fit_temperature(z, y)
    assert C.T_MIN <= t < 0.5 and C.mean_nll(z, y, t) < 1e-9
    with pytest.raises(ValueError):
        C.apply_temperature(z, 0.0)


# -------------------------------------------------------------------------------- splits


def make_items(n: int, prefix: str = "q", stem: str = "state number") -> list[D.TypedQuestion]:
    return [
        D.TypedQuestion(
            f"{prefix}{i}", f"{stem} {i}", f"which holds for {i}?", "choice", ("a", "b", "c"), i % 3
        )
        for i in range(n)
    ]


def test_splits_are_deterministic_roughly_the_requested_size_and_cover_everything():
    items = make_items(20_000)
    a, b = D.make_splits(items), D.make_splits(list(reversed(items)))
    assert {k: {q.qid for q in v} for k, v in a.items()} == {
        k: {q.qid for q in v} for k, v in b.items()
    }
    assert sum(len(v) for v in a.values()) == len(items)
    assert len(a["cal"]) == pytest.approx(0.05 * len(items), rel=0.1)
    assert len(a["select"]) == pytest.approx(0.05 * len(items), rel=0.1)
    D.assert_disjoint(a)


def test_adding_items_never_moves_existing_ones_between_splits():
    items = make_items(5_000)
    before = {q.qid: k for k, v in D.make_splits(items).items() for q in v}
    grown = items + make_items(2_000, prefix="mined")
    after = {q.qid: k for k, v in D.make_splits(grown).items() for q in v}
    assert all(after[qid] == k for qid, k in before.items())


def test_a_different_salt_gives_a_different_split():
    items = make_items(2_000)
    a = {q.qid for q in D.make_splits(items, salt="x")["cal"]}
    b = {q.qid for q in D.make_splits(items, salt="y")["cal"]}
    assert a != b


def test_assert_disjoint_catches_reformatted_text_copies_but_passes_clean_splits():
    train = make_items(10)
    D.assert_disjoint({"train": train, "dev": make_items(10, prefix="dev", stem="other")})
    clone = [  # new id, upper-cased with stray whitespace: still the same question
        D.TypedQuestion(
            "other-id", q.state.upper() + "  ", q.question, q.answer_type, q.options, q.gold
        )
        for q in train[:3]
    ]
    with pytest.raises(D.LeakError, match="shared texts"):
        D.assert_disjoint({"train": train, "dev": clone})


def test_assert_disjoint_flags_shared_ids():
    a = make_items(5)
    b = [D.TypedQuestion("q3", "different state", "different?", "noul", ("f", "t"), 0)]
    with pytest.raises(D.LeakError, match="shared ids"):
        D.assert_disjoint({"train": a, "cal": b})


def test_typed_question_validation():
    with pytest.raises(ValueError):
        D.TypedQuestion("x", "s", "q", "essay", ("a", "b"), 0)
    with pytest.raises(ValueError):
        D.TypedQuestion("x", "s", "q", "noul", ("f", "t"), 2)


def test_seed_sets_must_be_disjoint():
    D.SeedSets(train=(0, 1, 2), select=(100, 101), evaluate=(1000, 1001))
    with pytest.raises(D.LeakError, match="train and evaluate"):
        D.SeedSets(train=(0, 1, 2), select=(100,), evaluate=(2, 1000))
    with pytest.raises(D.LeakError, match="select and evaluate"):
        D.SeedSets(train=(0,), select=(100, 101), evaluate=(101,))


def test_halves_are_a_stable_disjoint_even_split_of_the_training_split():
    items = make_items(20_000)
    train = D.make_splits(items)["train"]
    a, b = D.split_halves(train)
    assert len(a) + len(b) == len(train)
    assert len(a) == pytest.approx(len(b), rel=0.05)
    D.assert_disjoint({"A": a, "B": b})
    a2, b2 = D.split_halves(list(reversed(train)))  # independent of file order
    assert {q.qid for q in a} == {q.qid for q in a2} and {q.qid for q in b} == {q.qid for q in b2}
    # halving never touches the calibration or selection splits
    held_out = D.make_splits(items)
    D.assert_disjoint({"A": a, "B": b, "cal": held_out["cal"], "select": held_out["select"]})


def test_halves_do_not_depend_on_the_ninety_five_five_assignment():
    train = D.make_splits(make_items(20_000))["train"]
    a, _ = D.split_halves(train)
    other, _ = D.split_halves(train, salt="another-salt")
    assert {q.qid for q in a} != {q.qid for q in other}
