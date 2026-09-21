"""The generator, and the counterfactuals that make policy evaluation possible.

Two things are worth pinning here. First, that the process is *coherent* -- an action can only
help, the actions compose the way the policy assumes, and the cause labels partition the late
orders. Second, and more important, that the cause labels are a function of **observed
timestamps only**. That is the claim that makes the cause-aware policy implementable on a real
extract rather than only inside the simulator, so it is checked against the truth table rather
than asserted in a docstring.
"""
from __future__ import annotations

import pytest

from deliveryrisk.data.synthetic import ACTIONS, SyntheticConfig, dataset_summary, generate


def test_frames_are_complete_and_keyed(tiny_data):
    frames = tiny_data.frames
    assert set(frames) == {
        "customers", "sellers", "products", "carriers", "shipping_lanes", "orders",
        "order_items", "order_payments", "order_reviews",
    }
    orders = frames["orders"]
    assert orders.order_id.is_unique
    assert frames["order_items"].order_id.isin(orders.order_id).all()
    assert frames["order_reviews"].order_id.isin(orders.order_id).all()
    # Reviews exist only where a delivery happened.
    delivered = set(orders.loc[orders.order_status == "delivered", "order_id"])
    assert set(frames["order_reviews"].order_id) <= delivered


def test_lifecycle_timestamps_are_ordered(tiny_data):
    o = tiny_data.frames["orders"]
    assert (o.approved_ts >= o.purchase_ts).all()
    done = o.dropna(subset=["pickup_ts", "delivered_ts"])
    assert (done.pickup_ts >= done.approved_ts).all()
    assert (done.delivered_ts >= done.pickup_ts).all()
    # Cancelled orders never ship; in-flight orders have no delivery yet.
    assert o.loc[o.order_status == "cancelled", "delivered_ts"].isna().all()
    assert o.loc[o.order_status == "shipped", "delivered_ts"].isna().all()


def test_actions_can_only_help(tiny_data):
    """No action makes an order later, and `both` is at least as good as either alone."""
    t = tiny_data.truth
    for a in ("nudge", "expedite", "both"):
        assert (t[f"days_over_{a}"] <= t.days_over_none + 1e-9).all(), a
    assert (t.days_over_both <= t.days_over_nudge + 1e-9).all()
    assert (t.days_over_both <= t.days_over_expedite + 1e-9).all()
    assert t.late_both.mean() <= t.late_nudge.mean()
    assert t.late_both.mean() <= t.late_expedite.mean()
    assert t.late_expedite.mean() < t.late_none.mean()


def test_cause_labels_partition_the_late_orders(tiny_data):
    t = tiny_data.truth
    late = t.late_none
    assert (t.handling_fixable & ~late).sum() == 0
    assert (t.transit_fixable & ~late).sum() == 0
    covered = t.handling_fixable | t.transit_fixable | t.unfixable
    assert (covered[late]).all(), "some late order has no cause class"
    assert not (t.unfixable & (t.handling_fixable | t.transit_fixable)).any()


def test_cause_labels_are_computable_from_logged_timestamps(tiny_data):
    """The claim the cause-aware policy rests on.

    ``handling_fixable`` must be reproducible from what an operational extract records -- the
    handling time, the transit time, the promise -- plus two stated operational constants. If it
    could only be derived from latent state, the whole cause-aware branch would be a simulator
    artefact.
    """
    t = tiny_data.truth
    cfg = tiny_data.config
    recomputed_h = t.late_none & (
        (cfg.handling_floor_days + t.transit_days) <= t.slack_days
    )
    recomputed_t = t.late_none & ((t.handling_days + t.transit_floor_days) <= t.slack_days)
    assert (recomputed_h == t.handling_fixable).all()
    assert (recomputed_t == t.transit_fixable).all()


def test_nudge_never_rescues_a_transit_only_order(tiny_data):
    """The asymmetry the whole action catalogue is built on."""
    t = tiny_data.truth
    transit_only = t.late_none & t.transit_fixable & ~t.handling_fixable
    if transit_only.sum() == 0:
        pytest.skip("no transit-only late orders at this scale")
    assert t.loc[transit_only, "late_nudge"].all(), (
        "a seller escalation rescued an order whose delay was entirely in transit"
    )


def test_promise_is_solved_to_the_configured_late_rate():
    cfg = SyntheticConfig(
        n_orders=6000, n_unique_customers=5800, n_sellers=120, n_products=800,
        span_days=300, seed=5, target_late_rate=0.12,
    )
    summary = dataset_summary(generate(cfg).frames)
    assert summary["late_share_delivered"] == pytest.approx(0.12, abs=0.01)


def test_generation_is_deterministic_in_the_seed():
    kw = dict(n_orders=400, n_unique_customers=380, n_sellers=20, n_products=90, span_days=120)
    a = generate(SyntheticConfig(seed=3, **kw))
    b = generate(SyntheticConfig(seed=3, **kw))
    c = generate(SyntheticConfig(seed=4, **kw))
    assert a.frames["orders"].delivered_ts.equals(b.frames["orders"].delivered_ts)
    assert not a.frames["orders"].delivered_ts.equals(c.frames["orders"].delivered_ts)


def test_every_action_has_a_truth_column(tiny_data):
    for a in ACTIONS:
        assert f"late_{a}" in tiny_data.truth.columns
        assert f"days_over_{a}" in tiny_data.truth.columns


def test_censoring_is_a_small_tail_not_a_silent_filter(tiny_data):
    """Right-censoring must not swallow the late orders it is most likely to catch."""
    o = tiny_data.frames["orders"]
    in_flight = (o.order_status == "shipped").mean()
    assert in_flight < 0.05, f"{in_flight:.1%} of orders are still in flight at extract time"
