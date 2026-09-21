# Methodology

This document states the problem formally, explains why parts of it need a generator, and records
the design decisions a reader would otherwise have to reverse-engineer from the code.

Every number quoted anywhere in this repository is produced by `python experiments/run_all.py`
and written to `results/` and [`RESULTS.md`](RESULTS.md). Nothing is transcribed by hand.

---

## 1. The problem

A marketplace promises a delivery date at checkout. Some orders miss it. The promise is already
made by the time anything can be done, so the only lever left is what happens between **payment
approval** and **carrier collection** — the pre-dispatch window.

Two levers exist in that window:

* **nudge** — escalate to the seller: this order is at risk, ship it today. Cheap, and it only
  compresses *handling* time.
* **expedite** — buy the express service on the lane. Expensive, and it only compresses *transit*
  time.

Write $p_i = P(\text{late}_i)$, $L_i$ for the cost of order $i$ being late, $c_a(i)$ for the cost
of action $a$ on order $i$, and $u_a(i)$ for the reduction in lateness probability that action
$a$ achieves. The value of taking action $a$ rather than nothing is

$$\Delta_i(a) = u_a(i)\, L_i - c_a(i),$$

and the decision is $a_i^\* = \arg\max_a \Delta_i(a)$, subject to whatever budget exists.

Three things follow immediately, and they are what this repository is about.

**The threshold is per order, not global.** With one action, $\Delta_i(a) > 0$ rearranges to
$p_i > c_a(i) / \big(\tilde u_a\, L_i\big)$, where $\tilde u_a$ is uplift per unit of risk. The
right-hand side is a property of the *order*: a fixed-cost nudge on a high-value order breaks
even at a low probability, and the same nudge on a cheap order does not break even at any
probability worth acting on. RESULTS §5 reports the spread.

**Ranking is not sufficient.** The comparison above is against a number on the probability axis.
ROC-AUC, PR-AUC and decile capture are all invariant to any monotone transform of the score;
the threshold is not. A model can rank perfectly and still cross the line on the wrong orders.

**Risk is not sufficient either.** Because the two actions act on different mechanisms, the
useful quantity is not $p_i$ but the decomposition of $p_i$ into the part a nudge can reach and
the part expediting can reach. Two orders at 30 % risk can need opposite treatments.

---

## 2. The data

### 2.1 Nine tables, not one CSV

`customers`, `sellers`, `products`, `carriers`, `shipping_lanes`, `orders`, `order_items`,
`order_payments`, `order_reviews`. `src/deliveryrisk/data/schema.py` holds the whole schema and
emits DDL for SQLite and MySQL 8; `sql/schema_mysql.sql` is the committed output.

A wide, pre-joined CSV has already made every join decision for you, and the join decisions are
exactly where the point-in-time bugs live. Starting from the normalised tables means the
temporal question — *what did we know at approval time* — has to be answered explicitly for every
feature, because nothing in the schema answers it for you.

Timestamps are epoch seconds stored as doubles, in both dialects. The feature SQL does strict
ordering and interval arithmetic inside window frames; doubles behave identically on both
engines, `DATETIME` semantics do not.

### 2.2 The generator, and why it exists

`src/deliveryrisk/data/synthetic.py` builds a marketplace whose delivery times decompose as

```
delivered = approved + H + T
    H  handling: the seller picks, packs, hands over          a nudge compresses this
    T  transit:  the carrier moves the parcel                 expediting compresses this
    late  <=>  H + T > slack,    slack = promise - approved
```

Neither term is i.i.d. **Seller backlog** (approved-but-not-collected orders) slows the next
handover. **Carrier day-load** above capacity adds transit days to every parcel that carrier
touches that day, so an order's risk depends on other orders it has nothing to do with.
**Incidents** — a depot failure on a (carrier, region) pair for a week or two — make entity
history non-stationary, which is what makes a seller's or lane's past late rate an imperfect
guide to its present. Both congestion terms are computed from the realised order stream and fed
back in a second pass, which is stated here because it is a fixed point solved once, not to
convergence.

The generator exists for one reason: the policy is counterfactual. Scoring it needs
$\text{late}_i(a)$ for the action it took, and delivery is not re-runnable. So the process is
replayed under each action with common random numbers, and `truth` carries
$\text{late}_i(a)$ for all four actions. **It is never loaded into the database**, because no
operational extract contains it.

### 2.3 Calibration to Olist

Scale and marginals track the published Olist Brazilian e-commerce release (99,441 orders;
112,650 order items; 3,095 sellers; 32,951 products; 96,096 unique customers), so the generated
warehouse can be checked against a public reference rather than taken on trust. RESULTS §1 is
that comparison. The promise buffer is *solved* by bisection to hit the configured late rate
rather than hand-tuned, so the headline late rate is reproducible from the config.

Two deliberate departures, both visible in the calibration table:

* **`span_days`** is 730, against Olist's ~730 of order dates but ~180 of dense coverage. Entity
  history features need history; a window shorter than the entity turnover measures cold start,
  not steady state.
* **`mean_promise_days`** lands about 9 % below Olist's. The promise is generated as a rule of
  thumb — advertised handling, a multiple of published transit, a solved buffer — and matching
  both the promise level *and* the late rate *and* the delivery-time distribution exactly would
  require fitting the promise engine to the real data rather than stating it.

### 2.4 Censoring

Cancelled orders (~2 %) never deliver. Orders still moving at the extract are right-censored:
`shipped` status, null `delivered_ts`. Both keep their place in the *event stream* — a cancelled
order still occupied its seller's queue — and both drop out of training and evaluation. The
extract is taken `extract_lag_days` after the last purchase; too short a lag censors the slow
orders preferentially and quietly deflates the observed late rate, which is a real hazard when
building this kind of table from a live warehouse.

---

## 3. Point-in-time features

### 3.1 The guarantee

For an order approved at time $t$, every feature is a function of facts timestamped **strictly
before** $t$, plus the order's own pre-dispatch attributes.

### 3.2 How it is enforced

Not by `GROUP BY`. Every history feature is a window aggregate over one event stream per entity:

| kind | when | what it carries |
| --- | --- | --- |
| 0 decision | `approved_ts` | the row being built |
| 1 outcome | `delivered_ts` | late?, days over, handling days, transit days |
| 2 open | `approved_ts` / `pickup_ts` | a parcel entering the seller's / carrier's queue |
| 3 close | `pickup_ts` / `delivered_ts` | and leaving it |

with

```sql
PARTITION BY entity_key ORDER BY ev_ts RANGE BETWEEN <lo> AND CURRENT ROW
```

and one trick doing the work: `ev_ts = ts - EPS` for decisions, `ts` for everything else.

A `RANGE` frame includes every *peer* — rows whose `ORDER BY` value ties with the current one.
Shifting decisions back by an epsilon puts each decision strictly before every real event at the
same instant, so a decision at $t$ sees events at $t$ not at all: not another order's delivery
confirmed in the same second, and not its own queue entry. Ties are where point-in-time code
usually breaks and they are not rare — an ERP stamping whole seconds produces thousands a day.
`tests/test_point_in_time.py::test_self_exclusion_under_ties` pins both directions of it.

Five entity streams: seller, lane, carrier, category, customer. Seller and carrier also carry a
queue, and the queue features are the ones that are genuinely hard to reconstruct after the fact:
`seller_queue_depth` is how many orders that seller owed at the moment of the decision, and
`seller_queue_mean_age_days` is how long they had been owing them — the difference between five
fresh orders and five they had been sitting on for a week. Both fall out of the same running sum
(`+open_ts` on entry, `-open_ts` on exit).

### 3.3 Shrinkage

Entity late rates are shrunk toward a prior: $(\,n_{\text{late}} + w\pi\,)/(\,n + w\,)$. A seller
with two prior deliveries and one late one has a raw rate of 0.5, and a tree will carve that out
happily. $\pi$ is computed **from the training window only** — a shrinkage target fitted on the
evaluation window is a summary of the future wearing a feature's clothes.

Missing history stays missing. A seller's first order has `seller_mean_handling = NULL`, not the
population median; both model families take NaN natively, and `seller_n_prior = 0` is itself the
signal. Imputing tells the model that a brand-new seller is an average seller, which is the one
thing it is not.

### 3.4 Proving it

Four properties, checked by Hypothesis over generated streams rather than one hand-written case:

```
prefix invariance      f(S)[:k] == f(S[:k])        appending the future changes nothing
future independence    mutating order j > i leaves order i bit-identical
order invariance       the physical row order of the input cannot matter
self exclusion         an order never sees its own outcome, ties included
```

Three more back them up. The window SQL is cross-checked against an O(n²) brute-force
implementation of the same definitions (`features/reference.py`) on every entity and every
column. Flipping every label must not move a single static feature — the cheapest possible test
for accidental target encoding. And the leaky baseline is asserted to *fail* prefix invariance,
so it cannot be silently "fixed" into meaninglessness while the leak audit keeps reporting on it.

---

## 4. Models

**Split.** Chronological, 60/15/25 on `approved_ts`. Never random: every history feature is a
running aggregate, so a random split puts an order's own neighbours in the training set and
reports a number the calendar will never reproduce. The validation block exists so the calibrator
is fitted on data the risk model never saw.

**Two families.** Logistic regression and gradient boosting, on identical features. The linear
model is there to be beaten and to be checked against: when a booster is *enormously* better at
everything, look for the leak before celebrating.

**Calibration.** Isotonic regression fitted on the validation block. Diagnostics are ECE, MCE,
Brier with its reliability/resolution decomposition, and the calibration slope — the slope is
the one to read first, because a slope below 1 means the scores are over-spread, which is the
usual state of an uncalibrated booster and is exactly what over-intervening on the top decile
looks like from the inside.

**The cause model.** A second model answers $P(\text{cause} \mid \text{late})$ over four disjoint
classes: `handling_only` (only a nudge can save it), `transit_only` (only expediting can),
`either`, `neither`. Factorising as
$P(\text{late} \wedge \text{cause}) = P(\text{late})\,P(\text{cause} \mid \text{late})$ keeps the
calibration burden where it belongs: only $P(\text{late})$ is compared against a currency
threshold, so only it has to be calibrated. The conditional model supplies a mix, and a mix is a
ratio.

**The labels are observable.** This is the part that makes the cause-aware policy implementable
on real data rather than only in a simulator. A completed order records handling and transit
separately (`approved → pickup → delivered`), so "would this order have been on time had handling
run at the operational floor" is arithmetic on logged timestamps.
`tests/test_generator.py::test_cause_labels_are_computable_from_logged_timestamps` checks that
claim against the truth table instead of asserting it in prose. What is *not* identified from a
log is whether an intervention would have delivered that compression — see §5.3.

---

## 5. Economics

### 5.1 Margin

$m_i$ = gross margin on the goods (a documented per-category rate) plus a thin share of freight.

### 5.2 The cost of being late

$$L_i = \underbrace{k_{\text{support}}}_{\text{contacts}} \; + \; \underbrace{\pi_v s_v \cdot \text{value}_i}_{\text{goodwill}} \; + \; \underbrace{d \cdot \kappa \cdot h \cdot m_i}_{\text{retention}}$$

Three terms, so each can be argued with separately. The retention term is the one usually
asserted; here $d$ — the lift in bad-review probability caused by lateness — is **estimated from
the data**, on the training window only (`estimate_review_damage`). $\kappa$ (share of bad-review
customers who do not return) and $h$ (future orders a retained customer is worth) remain stated
assumptions, named in the config.

$L_i$ scales with order value. $c_{\text{nudge}}$ does not. That is the whole reason the
threshold is per order.

### 5.3 Uplift, and where the honesty lives

$\rho$ — the probability that an action actually delivers the compression — is **not identified
from a do-nothing log**. Nothing in an observational extract says what a nudge would have
achieved. It is a stated assumption, and a policy is only as good as it.

So: the defaults are what a well-run uplift experiment on this population would report, the
generator knows the true response, and RESULTS §8 sweeps the *believed* value from 0.4× to 2× the
true one and reports realised contribution at each point. A single assumed uplift with no
sensitivity analysis would be a number dressed up as a result.

### 5.4 Prices are inputs too

`expedite_surcharge` is a commercial term, not a model output. At the carrier's published price
the express upgrade turns out to be worth buying for almost nobody, which is a fact about the
price list rather than about the model — so RESULTS §6 sweeps it and shows where the action mix
turns over.

---

## 6. Decision rules

All of them are written against the same expected-value table, so the comparison is between
decision rules and never between implementations.

| rule | what it knows |
| --- | --- |
| `top_k%` | the risk ranking. No economics at all. |
| `f1` / `youden` | a threshold chosen to optimise a classification metric |
| `margin_global_threshold` | one probability cut-off, *chosen on expected contribution* |
| `margin_risk_only` | per-order expected contribution, population cause mix |
| `margin_cause_aware` | per-order expected contribution, per-order cause mix |
| `margin_budget_*` | the same, under a spend cap |
| `oracle` | every counterfactual. Unattainable; it is the denominator. |

Two of these deserve a note on fairness. `margin_global_threshold` is the *strongest* version of
"one threshold for everyone": the cut-off is chosen by sweeping it against the same objective the
per-order rule maximises, so the gap between them is the cost of the constraint and not of a
worse objective. And `margin_risk_only` keeps the population cause mix measured on the training
window — handicapping it further, by letting it assume a nudge rescues any late order, would win
the comparison by misconfiguring the baseline.

**Under a budget**, the continuous relaxation of the knapsack is solved by a single price: pick,
per order, the action maximising $\text{benefit} - (1+\lambda)\,\text{cost}$, and bisect $\lambda$
until the spend meets the cap. Unlike greedy-by-ratio it hands back the shadow price — the return
on the next unit of budget, which is the number worth taking to whoever owns the cap.

---

## 7. Evaluation

Realised contribution per order, against the counterfactual truth:

$$\text{contribution}_i = m_i - c_{a_i}(i) - L_i \cdot \mathbb{1}\{\text{late}_i(a_i)\}$$

reported as the difference against doing nothing, per 1,000 orders, with a percentile bootstrap
over orders. The interval is not decoration: late costs are heavy-tailed, and a difference of a
few units per thousand orders is not worth reporting without one.

Three things this deliberately does not do. It does not score a policy on its own beliefs — a
policy that thinks a nudge is twice as effective as it is looks excellent under its own
expected-value table. It does not reuse the model's probabilities; realised lateness comes from
the truth table. And it reports the oracle, so "the policy earned X per thousand orders" can be
read as "the policy captured Y % of what was there to capture".

---

## 8. Threats to validity

**The uplift belief is an input.** §5.3. Swept, not assumed away — but the sweep holds the true
response fixed, so it answers "how wrong can the belief be", not "what if the response differs by
segment".

**The cost model is assumed apart from one term.** Only $d$ is estimated. $\kappa$ and $h$ are
stated. Contribution scales roughly linearly in $L_i$, so a reader who disagrees with those
constants can scale the headline mentally; the *ordering* of the decision rules is much more
robust than the level, because every rule is priced against the same $L_i$.

**The closed loop is not simulated.** Intervening changes seller behaviour, and a seller who is
nudged weekly may stop responding to nudges. Nothing here models that. It is the single largest
gap between this and a deployed system, and it is the same failure mode as the one in
[`closed-loop-fraud/`](../../closed-loop-fraud/) — the decisions become the data.

**One generator, one seed per configuration.** The parameter sweeps move one axis at a time.

**The real-data path cannot run the policy comparison.** `data/olist.py` loads the real release
into the same nine tables, and the features, the leak audit, the models and the calibration
diagnostics all run unchanged on it. The realised-contribution comparison does not, because the
extract records one action — nothing — for every order. Saying that plainly is worth more than a
simulation dressed up as a real-data result.

**Two of the nine tables are derived on the real path.** Olist has no carrier column, so
`shipping_lanes` is `(seller_state, customer_state)` with the empirical median transit time of
that pair over the training window, and `carriers` is one implicit carrier whose expedite
surcharge is a stated operational price rather than a measurement. Documented in full in
`src/deliveryrisk/data/olist.py`.

---

## 9. Reproducing

```bash
pip install -e ".[dev,gbdt]"
make test          # 80+ unit and property tests, a few seconds
make smoke         # tiny end-to-end run, well under a minute
make results       # the full 99K-order experiment, ~3 minutes; rewrites docs/
```

`results/headline/config.json` is the fully resolved configuration for the committed run.
