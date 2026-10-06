"""The torch losses agree with autograd, with each other (A1 = A5), and with the numpy estimators."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rlcd import estimators as E  # noqa: E402
from rlcd import isolated as I  # noqa: E402
from rlcd import torch_batch as TB  # noqa: E402
from rlcd import torch_losses as TL  # noqa: E402
from rlcd import units as U  # noqa: E402

DT = torch.float64


def t(x):
    return torch.tensor(np.asarray(x), dtype=DT)


def grad_of(fn, z0):
    z = t(z0).clone().requires_grad_(True)
    fn(z).backward()
    return z.grad.numpy()


# ------------------------------------------------------------------ batching and readout


def make_unit(qtype: str, n: int, seed: int) -> U.Unit:
    rng = np.random.default_rng(seed)
    n_rows = n if qtype == "score" else 1
    rows = []
    for _ in range(n_rows):
        k = 2 if qtype == "score" else n
        perm = rng.permutation(k).astype(np.int16)
        ids = rng.integers(5, 50, size=int(rng.integers(8, 20))).astype(np.int32)
        rows.append(U.Row(ids, len(ids) - 1, k, perm))
    return U.Unit(f"u{seed}", "r", qtype, "f", "d", n, rows, gold=0)


def test_original_order_undoes_the_shuffle_and_collate_shapes():
    units = [make_unit("choice", 5, 1), make_unit("score", 4, 2), make_unit("noul", 2, 3)]
    batch = TB.collate(units, pad_id=0)
    n_rows = sum(len(u.rows) for u in units)
    assert batch.input_ids.shape[0] == n_rows and batch.input_ids.shape[1] % 64 == 0
    assert batch.spans == [(0, 1), (1, 5), (5, 6)]
    rng = np.random.default_rng(0)
    true_orig = rng.normal(size=(n_rows, 5))
    shown = np.full((n_rows, 255), -np.inf)
    rows = [r for u in units for r in u.rows]
    for b, r in enumerate(rows):
        for pos, orig in enumerate(r.perm.tolist()):
            shown[b, pos] = true_orig[b, orig]  # the model shows option `orig` at position `pos`
    got = TB.original_order(torch.tensor(shown), batch).numpy()
    for b, r in enumerate(rows):
        np.testing.assert_allclose(got[b, : r.nopts], true_orig[b, : r.nopts])


def test_score_readout_is_yes_minus_no_whichever_order_the_two_options_were_shown():
    u = make_unit("score", 3, 4)
    batch = TB.collate([u], 0)
    z_yes = np.array([1.0, -0.5, 2.0])
    z_no = np.array([0.2, 0.7, -1.0])
    shown = np.full((3, 255), -np.inf)
    for b, r in enumerate(u.rows):
        vals = {0: z_no[b], 1: z_yes[b]}  # original option 0 = "no", 1 = "yes"
        for pos, orig in enumerate(r.perm.tolist()):
            shown[b, pos] = vals[orig]
    orig = TB.original_order(torch.tensor(shown), batch)
    (z,) = TL.unit_z(orig, batch, [u])
    np.testing.assert_allclose(z.numpy(), z_yes - z_no)


@pytest.mark.parametrize("qtype", ["choice", "noul", "score"])
def test_logp_matches_the_numpy_readout_and_sums_to_one(qtype):
    z = np.random.default_rng(1).normal(size=5 if qtype != "noul" else 2)
    logp, mass = TL.logp_from_z(qtype, t(z))
    p = logp.exp().numpy()
    np.testing.assert_allclose(p.sum(), 1.0)
    if qtype == "score":
        ref, ref_mass = I.combine(I.yes_probability(z))
        np.testing.assert_allclose(p, ref[0], atol=1e-9)
        assert mass.item() == pytest.approx(ref_mass[0])
    else:
        np.testing.assert_allclose(p, E.softmax(z), atol=1e-9)
        assert mass.item() == 1.0


# -------------------------------------------------------------- gradients: torch vs numpy


def test_supervised_gradient_matches_the_numpy_log_score_gradient():
    rng = np.random.default_rng(2)
    z, gold = rng.normal(size=6), 3
    ana = E.full_score_grad_logits(E.softmax(z)[None], np.array([gold]), "log")[0]
    got = grad_of(lambda zz: TL.supervised_loss("choice", zz, gold), z)
    np.testing.assert_allclose(got, -ana, atol=1e-9)  # the loss descends where the score ascends


@pytest.mark.parametrize("gold", [0, 2, 4])
def test_score_loss_gradient_matches_the_numpy_isolated_gradient(gold):
    l = np.random.default_rng(3).normal(size=5) * 1.5
    ana = I.score_grad_level_logits(l[None], np.array([gold]), "rps")[0]
    got = grad_of(lambda zz: TL.supervised_loss("score", zz, gold), l)
    np.testing.assert_allclose(got, -ana, atol=1e-8)


@pytest.mark.parametrize("qtype,n", [("choice", 6), ("noul", 2), ("score", 5)])
def test_rlcd_full_through_the_analytic_path_has_the_same_gradient_as_supervised_autograd(qtype, n):
    """H2 at the level of one question: the injected analytic gradient equals the autograd gradient of A1."""
    rng = np.random.default_rng(4)
    for gold in range(n):
        z = rng.normal(size=n) * 2
        a1 = grad_of(lambda zz, g=gold: TL.supervised_loss(qtype, zz, g), z)
        a5 = grad_of(lambda zz, g=gold: TL.full_score_loss(qtype, zz, g), z)
        np.testing.assert_allclose(a5, a1, atol=1e-8)


def test_forward_kl_gradient_matches_the_numpy_forward_kl_gradient():
    rng = np.random.default_rng(5)
    z, zr = rng.normal(size=5), rng.normal(size=5)
    ref = torch.log_softmax(t(zr), 0)
    got = grad_of(lambda zz: TL.forward_kl(ref, torch.log_softmax(zz, 0)), z)
    np.testing.assert_allclose(
        got, E.kl_grad_logits(E.softmax(z)[None], E.softmax(zr)[None])[0], atol=1e-9
    )
    assert TL.forward_kl(ref, ref).item() == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------- the sampled estimators: bias, by Monte Carlo


def mc_mean_grad(arm, qtype, z0, gold, k, draws, seed):
    gen = torch.Generator().manual_seed(seed)
    grads = []
    for _ in range(draws):
        z = t(z0).clone().requires_grad_(True)
        TL.sampled_policy_loss(arm, qtype, z, gold, k, gen).backward()
        grads.append(z.grad.numpy().copy())
    g = np.stack(grads)
    return g.mean(0), g.std(0, ddof=1) / np.sqrt(draws)


def test_bandit_gradient_is_the_log_likelihood_gradient_shrunk_by_k_minus_1_over_k():
    z0, gold, k = np.array([1.2, 0.3, -0.4, 0.8, -1.5]), 3, 8
    mean, se = mc_mean_grad("rlcd_bandit", "choice", z0, gold, k, 12000, 0)
    target = -(k - 1) / k * (np.eye(5)[gold] - E.softmax(z0))  # a loss: minus the ascent direction
    assert np.all(np.abs(mean - target) <= 5 * se)


def test_rlvr_gradient_is_the_gradient_of_p_y_not_log_p_y():
    z0, gold, k = np.array([1.2, 0.3, -0.4, 0.8, -1.5]), 3, 8
    p = E.softmax(z0)
    mean, se = mc_mean_grad("rlvr", "choice", z0, gold, k, 12000, 1)
    target = -(k - 1) / k * p[gold] * (np.eye(5)[gold] - p)
    assert np.all(np.abs(mean - target) <= 5 * se)
    # and the bandit target differs from it, so the test has teeth
    assert np.abs(target - (-(k - 1) / k * (np.eye(5)[gold] - p))).max() > 0.1


def test_bandit_on_a_score_question_follows_the_log_score_of_the_normalised_distribution():
    l0, gold, k = np.array([0.8, -0.3, 1.1, -0.9]), 2, 8
    ana = grad_of(lambda zz: TL.log_score_loss(TL.logp_from_z("score", zz)[0], gold), l0)
    mean, se = mc_mean_grad("rlcd_bandit", "score", l0, gold, k, 12000, 2)
    assert np.all(np.abs(mean - (k - 1) / k * ana) <= 5 * se + 1e-9)


# ------------------------------------------------------------------- the combined loss


def test_unit_loss_adds_beta_times_the_forward_kl_and_reports_diagnostics():
    rng = np.random.default_rng(6)
    z = t(rng.normal(size=4)).requires_grad_(True)
    ref = torch.log_softmax(t(rng.normal(size=4)), 0)
    with_kl = TL.unit_loss("supervised", "choice", z, 1, ref, beta=0.5, k=8)
    plain = TL.unit_loss("supervised", "choice", z, 1, ref, beta=0.0, k=8)
    assert with_kl.loss.item() == pytest.approx(plain.loss.item() + 0.5 * with_kl.kl.item())
    assert with_kl.nll.item() == pytest.approx(-torch.log_softmax(z, 0)[1].item())
    assert TL.unit_loss("supervised", "choice", z, 1, None, beta=0.5, k=8).kl.item() == 0.0
    with pytest.raises(ValueError):
        TL.unit_loss("ppo", "choice", z, 1, None, 0.0, 8)


def test_score_mass_is_reported_and_the_mean_deviation_matches_numpy():
    l = np.random.default_rng(7).normal(size=(30, 5))
    masses = [TL.logp_from_z("score", t(row))[1].item() for row in l]
    assert np.mean(np.abs(np.array(masses) - 1)) == pytest.approx(
        I.mass_deviation(I.yes_probability(l))
    )
