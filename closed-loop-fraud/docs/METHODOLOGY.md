# Methodology

This document states the problem formally, explains why it needs a simulator, and records the
design decisions that a reader would otherwise have to reverse-engineer from the code.

---

## 1. The problem

An issuer runs a fraud model $s(\cdot)$ and declines a transaction when $s(x) \geq \tau$.

A declined transaction is never authorised. It never settles, never posts to a statement, and
so **never charges back**. Its fraud label does not arrive late — it does not exist. The only
outcomes the issuer observes are the outcomes of transactions it chose to approve.

Write $A_i \in \{0,1\}$ for the approval decision, $Y_i$ for the true fraud outcome, and
$O_i$ for "the label is observed". Then

$$O_i = A_i = \mathbb{1}\!\left[s(X_i) < \tau\right].$$

Next quarter's model is fit on $\{(X_i, Y_i) : O_i = 1\}$. That set is not a sample of
transactions; it is a sample of *transactions the previous model let through*. Training on it
optimises

$$\mathbb{E}\big[\ell(f(X), Y) \mid O = 1\big] \neq \mathbb{E}\big[\ell(f(X), Y)\big],$$

and the inequality is not small, because $O$ is a deterministic function of a score that was
built to correlate with $Y$ as strongly as possible. The selection is *maximally* adversarial
with respect to the quantity being estimated.

Two consequences compound each cycle:

1. **The positive class degrades into a biased subsample.** The fraud that survives into
   training is the fraud the model failed to catch. Patterns the model catches well contribute
   no positives at all. Retrain, and the model's competence on those patterns erodes — which is
   to say the loop systematically unlearns its own best signals.
2. **Whole regions of feature space go dark.** Inside the decline region *nothing* is labelled:
   not the fraud and not the false positives. A tree model with no data in a region does not
   abstain there, it extrapolates from its neighbours, and the extrapolation drifts toward the
   population base rate. The ranking in the region the policy acts on is the ranking the model
   has the least evidence about.

### Why the dashboards stay green

The failure is silent. Every metric an issuer can actually compute — precision on declines,
chargeback rate on approvals, score distribution, approval rate — is computed on approved
traffic or on the decisions themselves. Recall against *all* fraud requires labels for declined
transactions, which is exactly what does not exist. A closed loop can lose a third of its recall
while every monitored number stays inside its control limits.

---

## 2. Why a simulator

The question is counterfactual: *what would this model have caught, on transactions the policy
refused?* No production log answers it, and neither does a static benchmark — IEEE-CIS included,
since its labels reflect one historical policy's decisions and contain no record of what a
different policy would have seen.

So the environment has to be one where every counterfactual label exists. That is the only
thing the synthetic generator is for. It is not a claim that the data is realistic in every
respect; it is a claim that it reproduces the marginals that matter (volume, fraud rate, amount
distribution, entity cardinality, class separability) while additionally supplying the labels no
real dataset can.

The real dataset still does work here. `clfraud.data.ieee_cis` maps the Kaggle files onto the
same schema so that the feature engine, the leakage audit, and the static model benchmark all
run unchanged on real data. What cannot run on it is the loop itself — and stating that plainly
is more useful than a simulation dressed up as a real-data result.

### Calibration targets

| quantity | IEEE-CIS `train_transaction.csv` | generator |
| --- | --- | --- |
| transactions | 590,540 | 590,540 |
| fraud rate | 3.499 % | 3.5 % |
| distinct `card1` | 13,553 | 13,553 |
| amount median / mean | \$68.77 / \$135.03 | see `results/headline/dataset_calibration.csv` |
| span | 181.8 days | 540 days (18 cycles) |

Span is the deliberate departure. The subject is retraining cycles, so the stream is generated
at 18 months natively. For the real data, `stretch_to_horizon` rescales inter-arrival gaps by a
single constant onto the same horizon — explicit and reversible, rather than a quiet reindex.

### The fraud process

Legit traffic is a non-homogeneous Poisson stream with daily and weekly seasonality, heavy-tailed
card activity, per-card ticket-size distributions, and habitual merchant baskets. Roughly 11 % of
it arrives in short bursts, so *several transactions on one card within the hour* is something
ordinary customers do — without that, card velocity separates the classes outright and there is
no problem left to study.

Fraud arrives as episodes from four archetypes whose mixture drifts across the 18 months:

| archetype | signature | detectability |
| --- | --- | --- |
| `card_testing` | 4–14 small authorisations, many merchants, minutes apart | high (velocity) |
| `ato_large` | 5–20× the card's usual amount, unused device, 02:00–05:00 | high (amount, device) |
| `merchant_breach` | clustered at one compromised acceptor over days or weeks | medium |
| `stealth_small` | in-distribution amount, normal hour, often the card's own device | low |

On top of that, 75 % of fraud episodes are placed inside **hot cells**: (merchant category,
hour-of-day) pairs that rotate every two months. Cells reach fraud rates above 90 % against a
3.5 % base rate, so they are highly learnable — *from recent labels*. That is the point. Knowing
which categories are hot right now is knowledge with a two-month shelf life, and a censored loop
is precisely a mechanism for not acquiring it. It is also the component of the signal that a
handful of well-weighted exploration samples can restore, which is why 2 % of declines buys
disproportionately more than its size suggests.

---

## 3. Point-in-time features

### The guarantee

For a stream ordered by $(\texttt{ts}, \texttt{transaction\_id})$, the feature row for
transaction $i$ depends only on $\{j : (\texttt{ts}_j, \texttt{id}_j) < (\texttt{ts}_i,
\texttt{id}_i)\}$ and on transaction $i$'s own immutable attributes.

### How it is enforced

Not by review. There is no `groupby` over the full table anywhere in `clfraud.features`, so
there is no place for future data to enter. Features come from streaming accumulators
(`CountSumWindow`, `DistinctWindow`, `RecencyTracker`, `WelfordTracker`) with one rule: the
builder **queries before it updates**. A transaction cannot see itself, and it cannot see
anything after it, because nothing after it has been inserted.

Windowed `groupby().rolling()` would be faster. It is also where leakage bugs live — an
off-by-one on a closed interval silently includes the current row, the model looks excellent
offline, and it fails at launch.

### How it is tested

Four executable properties in `tests/test_leakage_properties.py`, checked by Hypothesis over
generated streams rather than on one hand-written example:

| property | statement |
| --- | --- |
| prefix invariance | $f(S)[:k] = f(S[:k])$ — appending the future changes nothing about the past |
| future independence | mutating any transaction after position $i$ leaves row $i$ bit-identical |
| order invariance | shuffling the input frame cannot change the output |
| self-exclusion | no transaction sees itself, **including under timestamp ties** |

Three more back them up: the O(1) accumulators are cross-checked against an O(n²) brute-force
definition of the same quantity; flipping every label must not move a single feature value
(which makes accidental target encoding impossible to introduce without a red test); and
`leaky_aggregate_baseline` is asserted to *fail* prefix invariance, so the counterexample used
in the leakage audit cannot be silently "fixed" into meaninglessness.

Timestamp ties deserve the explicit test. IEEE-CIS records `TransactionDT` to the second, so
simultaneous transactions on one card are routine, and a closed-interval window would inflate
every velocity feature on exactly the bursts that matter most.

### Why absolute time is not a feature

Calendar *shape* (hour of day, day of week) generalises. `days_since_start` only lets a tree
memorise which stretch of the timeline it trained on, which is worth nothing once deployed
forward. It is computed for diagnostics and excluded from the matrix.

---

## 4. The correction

### Identification

Inverse-propensity weighting with $e(x) = P(O = 1 \mid X = x)$ gives

$$\mathbb{E}\!\left[\frac{O}{e(X)}\,\ell(f(X), Y)\right] = \mathbb{E}\big[\ell(f(X), Y)\big]$$

**provided $e(x) > 0$ everywhere** (positivity). A deterministic threshold gives
$e(x) \in \{0, 1\}$, and positivity fails exactly on the decline region — the region that
matters. This is not a statistical inconvenience to be modelled around: the data is not missing
at random, it is missing by construction, and no estimator recovers it from the logs alone.

Randomised exploration is therefore not a heuristic bolted onto the policy. It is the
identification strategy. Approving a small fraction $\epsilon$ of would-be declines makes
$e(x) \geq \epsilon > 0$ everywhere, and the estimand becomes estimable.

### Where to explore

Given a fixed budget, *which* declines to approve is a real design decision with three
competing objectives, and the implemented modes each sacrifice a different one:

| mode | coverage | cost | positivity |
| --- | --- | --- | --- |
| `uniform` | full | highest | holds |
| `risk_tapered` | concentrated near the threshold | low | holds (with a floor) |
| `amount_capped` | full below the cap, **none above** | lowest | **violated above the cap** |
| `cost_aware` | full, stratified by score decile | low | holds |

`risk_tapered` looks appealing and is a trap: the fraud a closed loop forgets fastest sits deep
in the decline region, precisely where a taper almost never looks. `amount_capped` is what an
issuer will actually sign off on and is the one that breaks identification outright.

`cost_aware` is the design the headline result uses. It spreads the budget evenly across score
deciles of the decline region (coverage everywhere, which the taper sacrifices) and, within each
decile, tilts toward small tickets by $u(x) \propto 1/(1 + a(x)/a_0)$ — because a \$12 label
teaches about as much as a \$1,200 one and costs a hundredth as much. The amount tilt is a
*known* part of the logged propensity, so IPW corrects for it exactly; it buys cheapness without
buying bias. A floor keeps $q(x) > 0$ everywhere.

### Variance control

$1/e$ used raw is a variance disaster: at 2 % exploration each explored row carries weight 50,
so a few hundred rows can outvote hundreds of thousands. Three standard stabilisers are
available, and the default choice among them is load-bearing:

- **`stabilise`** — multiply by the marginal observation rate, keeping the weighted sample size
  near the raw one (the tree learner's `min_child_samples` is expressed in counts).
- **`self_normalise`** — divide by the mean weight (Hájek/SNIPS), so the effective learning rate
  does not depend on the exploration rate.
- **`clip_quantile`** — **off by default.** Explored rows are well under 1 % of the training
  set, so *any* upper-quantile clip truncates precisely the rows the correction exists to
  amplify. An earlier version of this project clipped at the 99.5th percentile and measured
  almost no recovery; the correction was being removed before it could act. `max_weight`, tied
  to the propensity floor, does the variance job without that failure mode.

`overlap_diagnostics` reports positivity violations, Kish effective sample size, and the share of
total weight on the heaviest row — the practical failure mode, where one explored transaction
dictates the boundary.

### Known vs. estimated propensities

The logged propensity is exact and should always be preferred: estimation error in $\hat e$
becomes bias in $1/\hat e$, worst where $\hat e$ is smallest. But many legacy stacks record
decisions without recording the randomisation, so `PropensityModel` estimates $\hat e(x)$ from
features — possible because features exist for declined transactions, only labels do not. The
`explore_ipw_estimated` arm quantifies what that costs.

---

## 5. Evaluation

**Recall at a fixed decline budget** is the headline metric throughout. ROC-AUC is close to
useless at a 3.5 % base rate — it is dominated by the ranking of negatives against each other,
and a model can gain AUC while getting worse at the only part of the distribution anyone acts
on. A fixed budget also makes the arms comparable: every arm declines the same volume, so
differences in fraud caught are differences in *ranking quality*, not in appetite.

Metrics are computed against oracle labels on the **entire** cycle, declines included. Business
outcomes are computed from the decisions actually taken.

`exploration_fraud_cost_share` is isolated deliberately. Comparing an exploring arm's total
fraud loss against a non-exploring arm's conflates two opposite effects — exploration lets fraud
through today, and the better model it trains stops more fraud tomorrow. The direct cost of
buying the labels is the first effect alone, and it is the number that has to be small for the
trade to be worth making.

### Thresholds are calibrated on the previous cycle

An issuer cannot see today's score distribution before deciding on today's traffic. So each
cycle's threshold is a quantile of the *previous* cycle's scores. This also means score drift
surfaces as decline-rate drift, which is how the failure actually presents itself on a
production dashboard.

### Label maturation

Chargebacks land 30–90 days after authorisation, so the freshest data is the least usable —
backwards from what drift wants. The default 45-day delay against 30-day cycles costs every
retrain about 1.5 cycles of evidence. `LabelPipelineConfig` also models unreported fraud and
friendly fraud; both are off in the headline run so that censoring is not confounded with label
noise, and both are available for ablation.

---

## 6. Experimental design

All arms share one stream, one feature matrix, one model class, one capacity budget and one
warm-up model. They differ only in which labels come back and how those labels are weighted.

| arm | labels returned | weighting | role |
| --- | --- | --- | --- |
| `oracle` | all, including declines | none | upper bound; isolates censoring from drift |
| `naive` | approvals only | none | what most production stacks do |
| `frozen` | n/a — never retrains | n/a | control: decay here is drift, not the loop |
| `explore_only` | approvals + 2 % of declines | none | separates having labels from using them |
| `explore_ipw` | approvals + 2 % of declines | IPW, capped at 250 | the proposal |
| `explore_ipw_cap50` | same | IPW, capped at 50 | variance ablation: a tighter cap |
| `explore_ipw_unstable` | same | raw $1/e$, no stabiliser | variance ablation: no cap at all |
| `explore_ipw_estimated` | same | IPW with $\hat e$ | cost of estimating the propensity |

A configuration detail worth stating because it destroyed an earlier version of this experiment:
an arm whose YAML omits `exploration:` or `weights:` must get *no* exploration and *no*
weighting. Both dataclasses default to the proposed behaviour — right for a caller constructing
them in code, catastrophic for a control arm that never asked. Before that was fixed, `naive`
and `oracle` were silently running 2 % exploration with IPW, the measured damage was understated,
and nothing failed. `tests/test_config.py` now asserts it for every shipped config.

The `oracle` arm declines **identically** to the others. If it also acted differently, its
advantage would confound "more labels" with "better decisions" and would measure nothing.

Feature construction is policy-independent, so the matrix is built once and shared. That is not
only an optimisation — it is the modelling claim: an issuer computes features for declined
transactions too, and never learns their outcome. Selection acts on labels, not on features,
which is exactly why $P(O \mid X)$ is estimable at all.

### Retraining memory

`train_window_cycles` (hard window) and `recency_half_life_cycles` (exponential decay) are both
implemented, and they fail differently:

- A **hard window** makes the loop *oscillate*. The model blocks a pattern, its labels vanish,
  the pattern falls out of the window in one step, the model forgets it wholesale, fraud floods
  back, labels return, it relearns. Recall sawtooths with a period near the window length.
- **Exponential decay** erodes smoothly, and with a long memory bound it barely erodes at all —
  old evidence is downweighted but never discarded.

The headline configuration uses a 3-cycle hard window because retraining on "the last quarter of
data" is what most teams actually do. `configs/ablation_recency.yaml` runs the alternative, and
`results/*/sensitivity.csv` sweeps window length against decline budget, because the size of the
damage is a property of the retraining configuration and reporting one number as if it were a
constant would be the kind of claim this project exists to argue against.

---

## 7. Threats to validity

- **The generator is a model, not the world.** The magnitude of the decay depends on how much of
  the fraud signal is non-stationary and label-hungry (`hot_cell_share`, `hot_cell_life_months`).
  That parameter is a genuine property of real fraud, but its value is not measured here. The
  sensitivity table is the honest response: report how the conclusion moves, not just the
  conclusion.
- **One decision, one label.** Real stacks have step-up authentication, manual review queues and
  multi-stage rules, all of which leak partial information about declined transactions. The
  binary approve/decline model is a worst case for label availability.
- **The false-positive cost rate is assumed.** 1.5 % of the declined amount is a conservative
  stand-in for lost interchange plus attrition risk. Every cost figure scales linearly in it.
- **Single seed per configuration.** Cycle-level recall is noisy, which is why headline numbers
  are means over three cycles rather than single points. Multi-seed confidence intervals are the
  obvious next step and are not done here.
- **Point-in-time correctness is verified for this feature set.** The property tests constrain
  any feature built from the provided accumulators. A future feature that reaches around them —
  a join against an externally maintained table, say — would need its own proof.
