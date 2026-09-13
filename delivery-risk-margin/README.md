# Delivery Risk Prediction & Margin-Optimised Intervention Policy

A model that flags a late delivery before dispatch has done nothing yet. Someone still has to
decide which flagged orders to act on, and with which action, knowing that an escalation costs
ops time, an express upgrade costs real money, and both are spent on an order that might well
have arrived on time anyway.

This repository builds the whole thing end to end — nine operational tables, point-in-time
features extracted in SQL, calibrated risk and cause models, and a decision layer that maximises
expected contribution — and then measures how much of the value is in each part.

> **Headline.** On a 99,441-order warehouse calibrated to Olist's published marginals, the
> margin-optimised, cause-aware policy is worth **+218 contribution per 1,000 orders** against
> doing nothing (95 % bootstrap **[162, 285]**), spending 147 to do it — a **2.5×** return.
>
> The decile heuristic — flag the riskiest 10 %, nudge them — is worth **+46**. A threshold
> chosen to maximise **F1**, applied with the standard intervention, is worth **−2,728**.
> All three use the same model and the same probabilities: only the rule that turns a
> probability into a decision changes.
>
> Two numbers that reframe the problem. Swapping gradient boosting for logistic regression moves
> the result by **2 %**. And the policy that cuts the late rate *most* — nudge every order,
> −29 % lateness — is the one that loses **457 per 1,000 orders**.

![what each decision rule is worth](docs/figures/policy_value.png)

Every number here is produced by `python experiments/run_all.py` and written to `results/` and
[`docs/RESULTS.md`](docs/RESULTS.md). Nothing in the documentation is transcribed by hand.

---

## What is actually built

| | |
| --- | --- |
| **A nine-table warehouse** | `customers`, `sellers`, `products`, `carriers`, `shipping_lanes`, `orders`, `order_items`, `order_payments`, `order_reviews` — with DDL for SQLite and MySQL 8 ([`sql/schema_mysql.sql`](sql/schema_mysql.sql)). Not a pre-joined CSV: the join decisions are where the point-in-time bugs live. |
| **Point-in-time features, in SQL** | 91 features from window aggregates over a per-entity event stream, strictly causal by construction, [committed as readable SQL](sql/features_point_in_time.sql). The guarantee is four executable properties checked by Hypothesis plus an O(n²) brute-force cross-check — not a code review. |
| **A counterfactual environment** | A congested fulfilment process — seller backlog, carrier capacity, regional incidents — that knows the delivery outcome **under every action**, which is the thing no operational log contains and no A/B test on a live marketplace runs for free. |
| **Calibrated risk, and calibrated cause** | Logistic regression and gradient boosting for `P(late)`, isotonic calibration on a held-out block, and a second model for `P(cause \| late)` over four disjoint classes — because a nudge and an express upgrade act on different halves of the delivery clock. |
| **A decision layer with prices in it** | Per-order margin, a three-term cost of lateness (one term estimated from the reviews table, not assumed), a per-order break-even threshold, four actions, and a Lagrangian allocator that returns the shadow price of the next unit of budget. |
| **Honest evaluation** | Realised contribution against the counterfactual truth, bootstrapped, with an oracle upper bound — so "+218 per 1,000 orders" can be read as "18.5 % of what was there to capture". |

---

## The findings

### 1. The decision layer is worth more than the model

| rule | treated | spend / 1k | late rate | **contribution / 1k** | 95 % CI | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| `oracle` (perfect foresight) | 5 % | 406 | −54 % | **+1,177** | [1093, 1278] | 3.9× |
| `margin_cause_aware` | 11 % | 147 | −6.6 % | **+218** | [162, 285] | 2.5× |
| `margin_risk_only` | 10 % | 134 | −5.6 % | **+197** | [140, 274] | 2.5× |
| `margin_budget_50pct` | 5 % | 73 | −3.8 % | **+183** | [124, 250] | **3.5×** |
| `margin_global_threshold` | 10 % | 131 | −6.0 % | **+162** | [119, 212] | 2.2× |
| `top_10pct_nudge` | 10 % | 120 | −5.8 % | **+46** | [13, 83] | 1.4× |
| `f1_threshold_nudge` | 21 % | 254 | −10.7 % | **+40** | [−3, 92] | 1.2× |
| `youden_threshold_nudge` | 43 % | 511 | −16.9 % | **−73** | [−122, −7] | 0.9× |
| `nudge_everything` | 100 % | 1,200 | −29.0 % | **−457** | [−521, −372] | 0.6× |
| `f1_threshold_both` | 21 % | 3,559 | −31.0 % | **−2,728** | [−2829, −2625] | 0.2× |

Read the last column against the fourth. **Every rule that cuts the late rate hardest destroys
the most value**, because the orders it adds are orders whose lateness was never worth the price
of preventing. The classification-optimal thresholds are not bad at classifying — F1 and Youden
do exactly what they promise. They optimise a quantity nobody is paid on, and they do it while
sitting next to a cost table that would have told them the answer.

Meanwhile the model choice barely registers: logistic regression lands at **+213** against
gradient boosting's **+218** on the same decision rule, a 2 % difference, while moving from the
decile heuristic to the margin rule is worth **4.7×**. Spending the next week on feature
engineering is a defensible choice. Spending it on the threshold is a better one.

### 2. Calibration is load-bearing, and ranking metrics cannot see it

The rule is `intervene when u·p·L > c`. That comparison happens on the **probability** axis, and
every ranking metric is invariant to monotone transforms of the score — so a model can rank
identically and still cross the line on the wrong orders.

| | raw scores | isotonic | change |
| --- | --- | --- | --- |
| `margin_cause_aware` | +203 | **+218** | +7 % |
| `margin_risk_only` | +153 | **+197** | **+29 %** |
| `margin_budget_50pct` | +127 | **+183** | **+44 %** |
| PR-AUC | 0.186 | 0.174 | — |
| capture, top 2 deciles | 0.388 | 0.389 | — |

![calibration](docs/figures/calibration.png)

The bottom two rows are the point. Calibration *slightly worsens* the headline ranking metrics —
isotonic regression is a step function and ties scores together — while being worth up to 44 % of
the money. An offline scoreboard built on PR-AUC would have rejected the step that mattered most.
The uncalibrated booster's calibration slope is **0.77**: its scores are over-spread, which is
what over-intervening on the top decile looks like from the inside.

### 3. There is no such thing as *the* threshold

`c / (u · L)` — the break-even probability for a nudge — computed per order:

| p5 | p25 | median | p75 | p95 | best single threshold |
| --- | --- | --- | --- | --- | --- |
| 0.052 | 0.121 | 0.207 | 0.320 | 0.580 | **0.062** |

![threshold spread](docs/figures/threshold_spread.png)

An **11× spread**, because `L` scales with order value while a nudge costs the same on every
order. The best single global cut-off — chosen by sweeping it against the *same* objective the
per-order rule maximises, so this is the strongest version of the one-threshold argument — sits at
0.062, below the break-even of three quarters of the orders it treats. Constraining the rule to
one number costs **26 %** of the value (+162 against +218).

### 4. Knowing *why* an order is at risk beats knowing how much

A seller escalation compresses handling. An express upgrade compresses transit. They are not
substitutes, and total risk does not say which one an order needs:

![cause mix](docs/figures/cause_mix.png)

The riskiest decile is **20 %** handling-driven; the safest is **67 %**. Risk rank and cause are
close to orthogonal, so a policy assigning by risk alone is spending on the wrong mechanism a
predictable share of the time.

At **equal spend** — both rules swept under the same budget, which is the only version of this
comparison that means anything:

| budget | risk-ranked | cause-aware | gap |
| --- | --- | --- | --- |
| 20 % | +94 | +105 | +12 % |
| 35 % | +118 | +152 | **+29 %** |
| 50 % | +148 | +183 | +23 % |
| 100 % | +197 | +218 | +11 % |

**The gap widens as the budget tightens**, which is the intuitive result and the useful one: when
you can only act on a few hundred orders a week, acting on the right mechanism is most of the
job. The effect is starkest in what each rule *buys*. At a negotiated express rate of a quarter
of list price, the cause-aware rule buys 607 upgrades and earns **+334 per 1,000 orders**; the
risk-ranked rule, at every price tested, buys **zero** expedite-only upgrades — it has no way to
identify an order whose delay is entirely in transit, so the action never wins its argmax.

### 5. The budget frontier bends early

![budget frontier](docs/figures/budget_frontier.png)

**Half the spend buys 84 % of the value** (+183 against +218), at a materially better return
(3.5× against 2.5×). The shadow price — what the next unit of budget earns — starts at 3.75 and
reaches zero by the unconstrained optimum, which is the number to take to whoever owns the cap.

### 6. How wrong can the uplift assumption be?

`ρ`, the probability that an intervention actually delivers the compression it promises, is **not
identified from a do-nothing log**. Nothing in an observational extract says what a nudge would
have achieved. It is an assumption, so it gets a sweep rather than a footnote — the physics stay
fixed and only the belief moves:

![uplift sensitivity](docs/figures/uplift_sensitivity.png)

| believed ÷ true | 0.4× | 0.6× | 0.8× | **1.0×** | 1.25× | 1.5× | 2.0× |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `margin_cause_aware` | +104 | +165 | +194 | **+217** | +230 | +210 | +176 |
| `margin_risk_only` | +87 | +135 | +164 | **+197** | +208 | +209 | +176 |

The policy stays profitable across the whole range, and the curve is **flat on the left and steep
on the right** — under-believing the uplift costs you ceiling, over-believing it costs you money,
because optimism buys interventions that were never going to pay. When the uplift is genuinely
unknown, the cheap error is to assume less than you hope.

---

## Point-in-time features over a relational schema, and proving it

The decision happens at `order_approved_at`: payment cleared, parcel not yet handed to a carrier.
Every feature must be computable *then*. The leak that actually happens is one line:

```python
df["seller_late_rate"] = df.groupby("seller_id").y_late.transform("mean")
```

a whole-table aggregate that, for an order approved in March, includes what that seller did in
April — and includes the order's own outcome. The column looks entirely ordinary.

The audit fits one model per feature mode and scores the same held-out orders twice: once with
features that can see forward, once with the causal features production can actually compute.

| scenario | PR-AUC | capture, top 2 deciles | contribution / 1k |
| --- | --- | --- | --- |
| honest (point-in-time) | 0.174 | 0.389 | **+138** |
| whole-table, **as reported offline** | 0.999 | 1.000 | +686 |
| whole-table, **as served** | 0.105 | 0.218 | **+60** |
| leave-one-out, **as reported offline** | 1.000 | 1.000 | +695 |
| leave-one-out, **as served** | 0.107 | 0.204 | +118 |

![leakage](docs/figures/leakage.png)

The second row is what goes in the deck. The third is the *same fitted model*, scoring the same
orders, with the future removed: it is **worse than the honest pipeline it was supposed to beat**
— 0.218 capture against 0.389 — and it earns **57 % less money**. No amount of retraining recovers
that. The signal it learned to lean on does not exist at decision time.

**The fourth row is the one worth arguing about.** Once someone points out that
`transform('mean')` includes the row's own outcome, the natural fix is to subtract it — a
leave-one-out aggregate. It recovers **nothing**: still 1.000 offline. The dominant leak was never
self-contamination. It is the centred window over a *shared* entity: a carrier's ±30-day late
rate reports the disruption the order is currently sitting inside, measured partly from parcels
that had not been delivered when the decision was made. Leave-one-out cannot touch that, because
the rows doing the damage belong to other orders.

The leak is also largest exactly where an honest builder has least to work with. Point-in-time
capture rises with seller history (0.341 for a seller with fewer than five prior deliveries,
0.411 for one with fifty or more) — the cold-start orders are genuinely harder. The leaky
features report 1.000 for all of them.

### The guarantee, and how it is enforced

Every history feature is a window aggregate over a per-entity event stream, never a `GROUP BY`:

```sql
PARTITION BY entity_key ORDER BY ev_ts RANGE BETWEEN <lo> AND CURRENT ROW
--   ev_ts = ts - EPS for decision rows, ts for outcome / queue-open / queue-close rows
```

A `RANGE` frame includes every *peer* — rows tying on the `ORDER BY` value. Shifting decisions
back by an epsilon puts each decision strictly before every real event at the same instant, so a
decision at *t* sees events at *t* not at all: not another order's delivery confirmed in the same
second, and not its own queue entry. Ties are where this code usually breaks, and they are not
rare — an ERP stamping whole seconds produces thousands a day.

The same builder emits the leaky variants: only the frame moves. That is what makes the table
above a comparison of *definitions* rather than of implementations.

Four properties, checked by Hypothesis over generated streams rather than one hand-written case:

```
prefix invariance      f(S)[:k] == f(S[:k])        appending the future changes nothing
future independence    mutating order j > i leaves order i bit-identical
order invariance       shuffling the input rows cannot change the output
self exclusion         no order sees its own outcome, including under timestamp ties
```

Three more back them up. The window SQL is cross-checked against an O(n²) brute-force
implementation of the same definitions, on every entity and every column. Flipping every label
must not move a single static feature, so accidental target encoding cannot be introduced without
a red test. And the leaky baseline is asserted to **fail** prefix invariance, so it cannot be
quietly "fixed" into meaninglessness while the audit keeps reporting on it.

The features that are hardest to reconstruct after the fact are the queue ones —
`seller_queue_depth` (how many orders that seller owed at the decision instant) and
`seller_queue_mean_age_days` (how long they had been owing them). Both fall out of the same
running sum, `+open_ts` on entry and `−open_ts` on exit, and they are the difference between a
seller with five fresh orders and a seller sitting on five from last week.

---

## Quick start

```bash
pip install -e ".[dev,gbdt]"

make test        # 80+ unit and property tests, a few seconds
make smoke       # tiny end-to-end run, well under a minute
make results     # the full 99,441-order experiment, ~3 min; rewrites docs/RESULTS.md
```

Or drive it directly:

```bash
drisk schema --dialect mysql                # the nine-table DDL
drisk build      configs/headline.yaml      # generate, load, extract features
drisk features   configs/headline.yaml --show-sql   # the point-in-time SQL itself
drisk train      configs/headline.yaml      # ranking, capture, calibration
drisk policy     configs/headline.yaml      # every decision rule against the truth
drisk leak-audit configs/headline.yaml      # point-in-time vs two-sided aggregates
```

### Running on the real Olist data

```bash
./scripts/download_olist.sh data/raw        # needs a Kaggle API token
drisk ingest-olist --raw-dir data/raw --database-url sqlite:///data/olist.db
drisk leak-audit configs/olist.yaml
```

The nine-table load, the feature query, the leakage audit, the models and the calibration
diagnostics all run unchanged on the real files. **The policy comparison cannot** — scoring a
policy needs the delivery outcome under the action it took, and a real extract records one action,
nothing, for every order. That gap is the reason the counterfactual environment exists, and saying
so plainly is worth more than a simulation dressed up as a real-data result.
`deliveryrisk.data.olist` documents every column mapping and both derived tables in one place
rather than burying them in feature code.

---

## Layout

```
src/deliveryrisk/
  data/          the nine-table schema + DDL, the generator, the Olist adapter, the DB handle
  features/      point-in-time SQL, the O(n^2) reference implementation, the builder
  models/        logistic + gradient boosting, isotonic calibration, the cause model
  policy/        margin and lateness costs, the action catalogue, thresholds, budget allocation
  evaluation/    decile capture, counterfactual policy value, the leak audit, figures
sql/             the committed DDL and feature SQL, regenerated by `make sql` and checked by a test
configs/         headline, smoke, ablations, real data — one YAML per experiment
experiments/     run_all.py: every stage, regenerates docs/RESULTS.md and every figure
tests/           property-based point-in-time suite, brute-force cross-check, policy algebra
docs/            METHODOLOGY.md (the full write-up), RESULTS.md (generated)
```

---

## What this does not claim

**The uplift belief is an input, not a finding.** `ρ` is swept from 0.4× to 2× the truth, which
answers "how wrong can the belief be" — not "what if the response differs by segment". A real
deployment gets this number from a randomised holdout, and the holdout is the first thing to
build.

**The cost of lateness is mostly assumed.** One of its three terms — the lift in bad-review
probability caused by a late delivery — is estimated from the reviews table on the training
window. The churn-per-bad-review and repeat-horizon constants are stated, not measured. The
*ordering* of the decision rules is far more robust than the level, because every rule is priced
against the same `L`.

**The closed loop is not simulated.** Intervening changes behaviour: a seller nudged every week
may stop responding to nudges, and then today's policy is training tomorrow's data. Nothing here
models that, and it is the largest gap between this and a deployed system. It is also the subject
of [`closed-loop-fraud/`](../closed-loop-fraud/), which measures what that feedback does to a
model that never sees the labels its own decisions destroyed.

Other limits — one generator, one seed per configuration, a promise engine calibrated to a late
rate rather than fitted to the real promise distribution — are in
[`docs/METHODOLOGY.md` §8](docs/METHODOLOGY.md).

## Further reading

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — the formal setup, the delivery process, the
  point-in-time argument, the cost model, threats to validity.
- [`docs/RESULTS.md`](docs/RESULTS.md) — every table and figure, regenerated from `results/`.

## License

MIT.
