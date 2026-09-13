# Closed-Loop Credit Card Fraud Detection under Selection Bias

A fraud model that declines a transaction never finds out whether it was right. Declined
authorisations do not settle, so they never charge back, so they never produce a label. Retrain
on next quarter's labels and you are not training on fraud — you are training on **the fraud
your previous model failed to stop**.

This repository builds the closed loop end to end and measures what it does.

> **Headline.** Over 18 monthly retraining cycles on a 590,540-transaction IEEE-CIS-calibrated
> stream, naive retraining loses **20.6 %** of its recall at a fixed 6 % decline budget
> (0.844 → 0.670), with a **50.6 % drawdown** at its worst cycle (0.417). A full-label oracle,
> declining identically, stays flat at 0.844. Approving **2 %** of would-be declines at random
> and training with propensity-aware weights recovers **85 % of the gap** — worth **+31 points
> of recall** in the bottom-decile cycles — for **0.43 %** of fraud value in added losses.
>
> The uncomfortable control: a model that is **never retrained at all** beats the naively
> retrained one by 11.8 points of mean recall. Under censored feedback, retraining is not a
> neutral act.

![recall trajectories](docs/figures/recall_trajectories.png)

Every number here is produced by `python experiments/run_all.py` and written to `results/` and
[`docs/RESULTS.md`](docs/RESULTS.md). Nothing in the documentation is transcribed by hand.

---

## What is actually built

| | |
| --- | --- |
| **A closed-loop simulator** | Declines censor labels; the next model trains on the survivors. Chargebacks mature on a 45-day delay. Thresholds are calibrated on the previous cycle, because an issuer cannot see today's score distribution before deciding on today's traffic. |
| **A counterfactual environment** | An IEEE-CIS-calibrated generator (590,540 transactions, 3.5 % fraud, 13,553 cards, 18 months) with **labels for every transaction, including the ones the policy declined** — the thing no production log and no static benchmark contains. |
| **Point-in-time features** | 49 velocity and aggregate features from streaming accumulators that query before they update. Guaranteed causal, and the guarantee is four executable properties checked by Hypothesis, not a code review. |
| **Bias correction** | Randomised exploration to restore positivity, logged and estimated propensities, IPW with three stabilisers and overlap diagnostics (ESS, positivity violations, max weight share). |
| **Honest evaluation** | Recall at a fixed decline budget against oracle labels on the *whole* cycle; realised P&L; IPS/SNIPS/DR off-policy estimators checked against a truth the simulator actually knows. |

---

## The three findings

### 1. The loop does not decay smoothly. It oscillates, and the mean falls.

Recall does not drift down; it collapses and recovers on a roughly five-cycle period —
0.84 → 0.46 → 0.55 → 0.82 → 0.85 → 0.80 → 0.47 → … The mechanism is a feedback cycle, not a
trend:

**block → the blocked pattern stops producing labels → it ages out of the training window →
the model forgets it → fraud floods back → labels return → the model relearns it → block**

Averaged over the 18 cycles, the naive loop runs at **0.719** against the oracle's **0.844**.
That difference is caused entirely by censoring: the oracle arm declines *identically*, it just
receives the labels anyway.

| arm | day-one | run mean | 10th pct cycle | worst cycle | max drawdown |
| --- | --- | --- | --- | --- | --- |
| `oracle` (all labels) | 0.845 | **0.844** | 0.824 | 0.813 | −3.9 % |
| `frozen` (never retrained) | 0.850 | **0.837** | 0.818 | 0.797 | −6.3 % |
| `explore_ipw_cap50` | 0.845 | **0.823** | 0.787 | 0.774 | −8.4 % |
| `explore_only` | 0.844 | **0.823** | 0.788 | 0.743 | −12.0 % |
| `explore_ipw` | 0.843 | **0.821** | 0.780 | 0.760 | −9.9 % |
| `naive` | 0.844 | **0.719** | 0.464 | **0.417** | **−50.6 %** |

**Every metric an issuer can compute stays green while this happens.** Precision on declines,
chargeback rate on approvals, approval rate, score distribution — all are measured on approved
traffic or on the decisions themselves. Recall against *all* fraud needs labels for declined
transactions, which is precisely what does not exist. The only column that moves is the one no
deployed system can produce.

### 2. The mechanism is visible in what the training set is made of

![positive composition](docs/figures/positive_composition.png)

The oracle's positive class holds a stable archetype mix. The naive loop's swings between 20 %
and 85 % `stealth_small` — the one archetype it cannot catch. At the trough cycles, the model
retrains on a positive class that is almost entirely *the fraud its predecessor missed*, having
never seen a labelled example of the card-testing bursts it used to stop. Then it stops stopping
them.

This is also why the damage is not a data-volume problem. The naive arm has 152–1,309 labelled
positives per cycle against the oracle's 1,366–2,068 — a real shortfall, but the composition
shift is what breaks it, not the count.

![training supply](docs/figures/training_supply.png)

### 3. 2 % exploration buys back most of it — and where you explore matters more than how you weight

A deterministic threshold gives $e(x) = P(\text{label observed} \mid x) \in \{0, 1\}$, so
positivity fails exactly on the decline region and **no estimator recovers it from the logs**.
The data is not missing at random; it is missing by construction. Randomised exploration is not
a heuristic bolted onto the policy — it is the identification strategy.

| arm | run mean | recovered (mean) | recovered (p10 cycles) | gap closed | added fraud loss | ESS |
| --- | --- | --- | --- | --- | --- | --- |
| `explore_ipw_cap50` | 0.823 | +10.4 pts | **+32.2 pts** | **84 %** | 0.49 % | 24 % |
| `explore_ipw_estimated` | 0.823 | +10.4 pts | +32.4 pts | 83 % | 0.48 % | 43 % |
| `explore_only` | 0.823 | +10.4 pts | +32.3 pts | 83 % | 0.49 % | 100 % |
| `explore_ipw_unstable` | 0.823 | +10.3 pts | +32.4 pts | 83 % | 0.41 % | 14 % |
| `explore_ipw` | 0.821 | +10.1 pts | +31.6 pts | 82 % | 0.43 % | 18 % |
| `naive` | 0.719 | — | — | 0 % | 0 % | 100 % |

**The result worth more than the headline: at a 2 % exploration rate, the weighting scheme
barely matters.** Five weighting variants — none, IPW capped at 250, capped at 50, raw
unstabilised $1/e$, and IPW with estimated propensities — span 81.6 % to 83.7 % of the gap
closed, while their effective sample sizes span 14 % to 100 %. The variance the weights add
roughly cancels the bias they remove.

What does the work is **coverage**: having any labelled evidence at all inside the region the
policy blocks. At this rate, explored rows are under 1 % of the training set but carry weights up
to 250, so the re-weighted fit is effectively built on a few hundred points either way. That is
worth knowing before spending a quarter on propensity infrastructure — get the randomisation
shipped first, then tune the weights.

The corollary is that reweighting *should* start paying at higher exploration rates or under
harsher censoring, where the surviving sample is large enough for the bias term to dominate the
variance term. That is testable with this code and is not tested here.

One trap is worth naming, because it cost this project a week of nothing happening: clipping
weights at the 99.5th percentile truncates *precisely* the explored rows the correction exists to
amplify. `clip_quantile` is therefore off by default. See
[`METHODOLOGY.md` §4](docs/METHODOLOGY.md).

#### Where to explore

Given a fixed budget, *which* declines to approve is a real design decision:

| mode | coverage of the decline region | cost | positivity |
| --- | --- | --- | --- |
| `uniform` | full | highest | holds |
| `risk_tapered` | concentrated near the threshold | low | holds only via a floor |
| `amount_capped` | none above the amount cap | lowest | **violated** |
| `cost_aware` ← used | full, stratified by score decile | low | holds |

`risk_tapered` is the appealing trap: the fraud a closed loop forgets fastest sits *deep* in the
decline region, exactly where a taper almost never looks. `cost_aware` spreads the budget evenly
across score deciles and, within each, tilts toward small tickets — a \$12 label teaches about as
much as a \$1,200 one and costs a hundredth as much. The tilt is a known part of the logged
propensity, so IPW corrects for it exactly: cheapness without bias.

![exploration frontier](docs/figures/exploration_frontier.png)

Read that figure horizontally. All four designs recover a similar amount of recall; what
separates them is the bill. At a 2 % rate:

| design | recall recovered | added fraud loss | positivity |
| --- | --- | --- | --- |
| `cost_aware` | +15.7 pts | **0.36 %** | holds |
| `amount_capped` | +16.5 pts | 0.49 % | **violated above the cap** |
| `risk_tapered` | +16.2 pts | 1.25 % | holds only via the floor |
| `uniform` | +15.2 pts | **1.55 %** | holds |

**Cost-aware exploration buys the same recovery as uniform exploration for a quarter of the fraud
losses.** `amount_capped` is competitive on both axes and is the one that gives up identification
entirely — above the amount cap $e(x) = 0$, so that region is unrecoverable no matter what
estimator you point at it. It is in the table because the honest comparison between corrections
is a comparison of *which assumption each one breaks*.

The frontier is also strikingly flat in the rate: 0.5 % exploration already recovers +14.3 points
against +16.5 at 4 %. Diminishing returns set in almost immediately, which again says the binding
constraint is coverage of the blocked region, not sample size within it.

---

## Point-in-time features, and proving it

The leak that matters in fraud work is not target encoding. It is an unshifted
`groupby().rolling('24h')` — a velocity aggregate over a window that includes transactions that
had not happened yet. It is invisible on inspection, because the resulting columns look
completely ordinary.

Its worst victim is the transaction you most want to catch. The *first* authorisation of a
12-transaction card-testing burst gets a card velocity of 12 offline, because the other eleven
are in its forward window. At serving time it has a velocity of zero, because the burst has not
happened yet.

The audit fits one model on two-sided features and scores the same held-out future window twice
— once with features that can see forward, once with the causal features production can actually
compute:

![leakage](docs/figures/leakage.png)

| scenario | PR-AUC | recall @ 6 % | recall on episode openings |
| --- | --- | --- | --- |
| honest (point-in-time) | 0.844 | 0.872 | 0.791 |
| leaky, **as reported offline** | 0.886 | 0.903 | 0.845 |
| leaky, **as served** | 0.823 | 0.861 | 0.783 |

The middle row is what goes in the deck. The bottom row is the *same fitted model*, scoring the
same transactions, with the only difference being that the future is gone: **4.3 points of recall
and 6.3 of PR-AUC evaporate**. And the model it produced is 1.1 points of recall *worse* than the
honest pipeline it was supposed to beat. No amount of retraining recovers that — the signal it
learned to lean on does not exist at decision time.

### The guarantee, and how it is enforced

For a stream ordered by `(ts, transaction_id)`, the feature row for transaction *i* depends only
on transactions strictly before it. There is no `groupby` over the full table anywhere in
`clfraud.features` — features come from streaming accumulators with one rule: **query before
update**.

Four properties, checked by Hypothesis over generated streams rather than one hand-written case:

```
prefix invariance      f(S)[:k] == f(S[:k])          appending the future changes nothing
future independence    mutating row j > i leaves row i bit-identical
order invariance       shuffling the input frame cannot change the output
self-exclusion         no transaction sees itself, including under timestamp ties
```

Three more back them up: the O(1) accumulators are cross-checked against an O(n²) brute-force
definition; flipping every label must not move a single feature value (so accidental target
encoding cannot be introduced without a red test); and the leaky baseline is asserted to *fail*
prefix invariance, so the counterexample cannot be silently "fixed" into meaninglessness.

The tie test earns its place: IEEE-CIS records `TransactionDT` to the second, so simultaneous
transactions on one card are routine, and a closed-interval window would inflate every velocity
feature on exactly the bursts that matter most.

---

## Can you see this coming before it happens?

If a closed loop degrades silently, the practical question is whether a candidate policy can be
scored from the incumbent's logs *before* being shipped. The simulator is the one place that
question has a checkable answer, because it holds the counterfactual labels the estimators are
working around.

Scoring a candidate that declines twice the volume, from logs the 6 %-budget policy produced:

| estimator | mean estimate | truth | mean relative error | 95 % CI covers truth |
| --- | --- | --- | --- | --- |
| IPS | 0.630 | 0.967 | **64 %** | 56 % of cycles |
| SNIPS | 0.884 | 0.967 | **9.3 %** | 67 % of cycles |

IPS is unbiased in expectation and useless in practice at this overlap — its per-cycle estimates
range from 0.04 to 1.66, and a *share* above 1 is not a noisy answer, it is a nonsensical one.
SNIPS is biased and roughly seven times more accurate, because self-normalisation bounds it by
construction. Use SNIPS.

This is also where weight clipping bites hardest: with an exploration floor of 0.004 the correct
weight on an explored row is 250, so a "sensible" cap of 50 discards four fifths of the only
evidence there is about the decline region and biases the estimate down by a factor of several.
`tests/test_metrics_and_ope.py` pins that, along with the case that matters most — **with no
exploration at all, the estimate collapses to under half the truth.** That is not an estimator
failure; it is an identification failure, and no better estimator fixes it.

---

## How bad is it? That depends on how you retrain.

The headline decay is a property of a retraining configuration, not a universal constant, so the
sweep is reported rather than one number:

| training window | decline budget | naive day-one → final | decay | worst cycle | gap closed by exploration |
| --- | --- | --- | --- | --- | --- |
| 3 cycles | 3 % | 0.674 → 0.586 | −13.1 % | 0.555 | 88 % |
| 3 cycles | 6 % | 0.844 → 0.670 | **−20.6 %** | **0.417** | 81 % |
| 6 cycles | 6 % | 0.849 → 0.746 | −12.2 % | 0.626 | 81 % |
| 12 cycles | 6 % | 0.849 → 0.830 | −2.3 % | 0.785 | 67 % |

Both knobs point the same way, for the same reason: **a shorter window and a larger decline
budget each remove more of the evidence the next model needs.** A long memory is the cheapest
partial mitigation available — a 12-cycle window nearly eliminates the decay on its own — but it
trades directly against the ability to track drift, and it is only available at all if the data
platform can afford to keep and reweight a year of history on every retrain.

---

## Quick start

```bash
pip install -e ".[dev,gbdt]"

make test            # 100+ unit and property tests, ~1 min
make smoke           # tiny end-to-end run, well under a minute
make headline        # the full 590K x 18-cycle experiment, ~9 min
make results         # headline + sweeps, regenerates docs/RESULTS.md
```

Or drive it directly:

```bash
clfraud describe  configs/headline.yaml --with-data   # resolved config + dataset marginals
clfraud simulate  configs/headline.yaml               # run every arm
clfraud leak-audit configs/headline.yaml              # point-in-time vs. two-sided window
```

### Running on the real IEEE-CIS data

```bash
./scripts/download_ieee.sh data/raw          # needs a Kaggle API token
clfraud ingest-ieee --raw-dir data/raw --out data/processed/ieee.parquet
```

The feature engine, the leakage audit and the static benchmark all run unchanged on the real
files. **The closed loop cannot** — a real log has no counterfactual label for a declined
transaction, so there is nothing to evaluate a would-be-declined decision against. That gap is
the reason the simulator exists, and saying so plainly is more useful than a simulation dressed
up as a real-data result. `clfraud.data.ieee_cis` documents every entity-key proxy
(`card1+card2+addr1` for the account, and so on) in one place rather than burying it in feature
code.

---

## Layout

```
src/clfraud/
  data/          canonical schema, IEEE-CIS adapter, calibrated generator
  features/      streaming accumulators + the point-in-time builder
  models/        gradient-boosted scorer, isotonic calibration
  bias/          propensity estimation, IPW, overlap diagnostics
  simulator/     decision policy, label maturation, the closed loop
  evaluation/    budgeted metrics, off-policy estimators, leak audit, figures
configs/         headline, ablations, smoke, real-data — one YAML per experiment
experiments/     run_all.py: every stage, resumable, regenerates docs/RESULTS.md
tests/           property-based leakage suite + unit tests
docs/            METHODOLOGY.md (the full write-up), RESULTS.md (generated)
```

---

## What this does not claim

The magnitude of the decay depends on how much of the fraud signal is non-stationary and
label-hungry — in the generator, `hot_cell_share` and `hot_cell_life_months`. That is a genuine
property of real fraud, but its value is not measured here. So `results/headline/sensitivity.csv`
sweeps training-window length against decline budget and reports how the conclusion moves, rather
than presenting one configuration's number as a constant. A single headline figure for "how bad
is the feedback loop" would be a fact about one retraining setup, not about the phenomenon.

Other limits — binary approve/decline with no step-up or review queue, an assumed
false-positive cost rate, a single seed per configuration — are listed in
[`METHODOLOGY.md` §7](docs/METHODOLOGY.md).

## Further reading

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — formal setup, the fraud process, the
  identification argument, experimental design, threats to validity.
- [`docs/RESULTS.md`](docs/RESULTS.md) — every table and figure, regenerated from `results/`.

## License

MIT.
